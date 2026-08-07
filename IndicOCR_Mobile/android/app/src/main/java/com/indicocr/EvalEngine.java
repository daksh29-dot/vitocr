package com.indicocr;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * EvalEngine
 * ==========
 * Mirrors inferFolder.py exactly — runs the ONNX model over a folder of
 * images, decodes CTC output, computes CER / WER / CRR / WRR against
 * ground-truth labels, and returns structured results for export to Excel.
 *
 * Ground-truth format (TSV, same as your training .txt files):
 *   /path/to/image.png\tground_truth_text
 *
 * Two modes:
 *   - runFolder()    : image folder only  → predictions, no metrics
 *   - runEvaluation(): TSV with GT labels → predictions + CER/WER/CRR/WRR
 */
public class EvalEngine implements AutoCloseable {

    // ── Constants — must match VIT_CTC.py training config ──────────────────
    private static final int   IMG_H      = 32;
    private static final int   IMG_W      = 256;
    private static final int   CHANNELS   = 2;     // dual-channel: grayscale + Otsu
    private static final float NORM_MEAN  = 0.5f;
    private static final float NORM_STD   = 0.5f;

    private static final String MODEL_ASSET = "vit_ctc.onnx";
    private static final String LABELMAP_ASSET = "label_map.json";

    // ── Fields ──────────────────────────────────────────────────────────────
    private final Context               context;
    private final OrtEnvironment        env;
    private final OrtSession            session;
    private final Map<Integer, String>  id2char;
    private final int                   blankId;

    // ── Result record ────────────────────────────────────────────────────────
    public static class ImageResult {
        public final String imagePath;
        public final String prediction;
        public final String groundTruth;   // null if no GT provided
        public final float  cer;           // -1 if no GT
        public final float  wer;           // -1 if no GT
        public final long   inferenceMs;

        public ImageResult(String imagePath, String prediction,
                           String groundTruth, float cer, float wer,
                           long inferenceMs) {
            this.imagePath   = imagePath;
            this.prediction  = prediction;
            this.groundTruth = groundTruth;
            this.cer         = cer;
            this.wer         = wer;
            this.inferenceMs = inferenceMs;
        }
    }

    public static class EvalSummary {
        public final List<ImageResult> results;
        public final float avgCER;
        public final float avgWER;
        public final float avgCRR;
        public final float avgWRR;
        public final int   totalImages;
        public final int   skipped;
        public final long  totalMs;

        public EvalSummary(List<ImageResult> results,
                           float avgCER, float avgWER,
                           float avgCRR, float avgWRR,
                           int totalImages, int skipped, long totalMs) {
            this.results     = results;
            this.avgCER      = avgCER;
            this.avgWER      = avgWER;
            this.avgCRR      = avgCRR;
            this.avgWRR      = avgWRR;
            this.totalImages = totalImages;
            this.skipped     = skipped;
            this.totalMs     = totalMs;
        }
    }

    // ── Constructor ──────────────────────────────────────────────────────────
    public EvalEngine(Context ctx) throws OrtException, IOException {
        this.context = ctx.getApplicationContext();
        env     = OrtEnvironment.getEnvironment();
        session = createSession(ctx);
        id2char = loadLabelMap(ctx);
        blankId = findBlankId(id2char);
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Run inference over all images in a folder.
     * Can operate in two modes:
     * 1. Predictions only (hasGT = false)
     * 2. Evaluation with matching .txt files (hasGT = true)
     */
    public EvalSummary runFolder(String folderPathOrUri,
                                 boolean hasGT,
                                 ProgressCallback cb,
                                 CsvResultWriter csvWriter) throws OrtException, IOException {
        
        List<String> imagePaths = new ArrayList<>();
        boolean isUri = folderPathOrUri.startsWith("content://");

        if (isUri) {
            Uri treeUri = Uri.parse(folderPathOrUri);
            String treeId = android.provider.DocumentsContract.getTreeDocumentId(treeUri);
            Uri childrenUri = android.provider.DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, treeId);

            try (android.database.Cursor cursor = context.getContentResolver().query(childrenUri,
                    new String[]{android.provider.DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                                 android.provider.DocumentsContract.Document.COLUMN_MIME_TYPE,
                                 android.provider.DocumentsContract.Document.COLUMN_DISPLAY_NAME},
                    null, null, null)) {
                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        String mime = cursor.getString(1);
                        if (mime != null && mime.startsWith("image/")) {
                            String docId = cursor.getString(0);
                            Uri fileUri = android.provider.DocumentsContract.buildDocumentUriUsingTree(treeUri, docId);
                            imagePaths.add(fileUri.toString());
                        }
                    }
                }
            }
        } else {
            File folder = new File(folderPathOrUri);
            if (!folder.exists()) {
                throw new IOException("❌ Folder does not exist: " + folderPathOrUri);
            }
            File[] files = folder.listFiles(f -> {
                String n = f.getName().toLowerCase();
                return n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg")
                    || n.endsWith(".bmp") || n.endsWith(".tif") || n.endsWith(".tiff")
                    || n.endsWith(".webp");
            });
            if (files != null) {
                Arrays.sort(files);
                for (File f : files) imagePaths.add(f.getAbsolutePath());
            }
        }

        if (imagePaths.isEmpty()) {
            throw new IOException("📂 Empty Folder: No valid images found in: " + folderPathOrUri);
        }

        if (cb != null) cb.onProgress(0, imagePaths.size());

        List<ImageResult> results = new ArrayList<>();
        int skipped = 0;
        int totalWithGT = 0;
        double totalCER = 0, totalWER = 0, totalCRR = 0, totalWRR = 0;
        long totalStart = System.currentTimeMillis();

        int processedCount = 0;
        for (int i = 0; i < imagePaths.size(); i++) {
            if ((cb != null && cb.isCancelled()) || Thread.interrupted()) {
                logTermination();
                break;
            }
            String path = imagePaths.get(i);
            String fileName = isUri ? Uri.parse(path).getLastPathSegment() : new File(path).getName();
            
            if (cb != null) cb.onProgressWithDetails(i + 1, imagePaths.size(), "Processing: " + fileName);

            processedCount++;
            try {
                long t0   = System.currentTimeMillis();
                String pr = inferImage(path);
                long ms   = System.currentTimeMillis() - t0;

                String gt = null;
                float cer = -1f, wer = -1f;

                if (hasGT) {
                    // GT matching logic for URIs would be complex, skipping for now or keeping file-based
                    if (!isUri) {
                        gt = findGroundTruthInFolder(new File(path));
                        if (gt != null) {
                            cer = computeCER(pr, gt);
                            wer = computeWER(pr, gt);
                            totalCER += cer; totalWER += wer;
                            totalCRR += (1 - cer); totalWRR += (1 - wer);
                            totalWithGT++;
                        } else {
                            skipped++;
                        }
                    } else {
                        skipped++;
                    }
                }

                ImageResult res = new ImageResult(path, pr, gt, cer, wer, ms);

                if (csvWriter != null) {
                    csvWriter.writeRow(res, hasGT);
                    if (cb != null) {
                        cb.onProgressWithDetails(csvWriter.getRowsWritten(), imagePaths.size(),
                                "Written: " + fileName);
                    }
                } else {
                    results.add(res);
                }
            } catch (Exception e) {
                skipped++;
                android.util.Log.e("EvalEngine", "Error processing: " + fileName, e);
            }
            if (cb != null) cb.onProgress(i + 1, imagePaths.size());
        }

        if (hasGT && processedCount > 0) {
            long totalMs = System.currentTimeMillis() - totalStart;
            return new EvalSummary(results,
                    totalWithGT > 0 ? (float)(totalCER / totalWithGT) : 0f,
                    totalWithGT > 0 ? (float)(totalWER / totalWithGT) : 0f,
                    totalWithGT > 0 ? (float)(totalCRR / totalWithGT) : 0f,
                    totalWithGT > 0 ? (float)(totalWRR / totalWithGT) : 0f,
                    processedCount, skipped, totalMs);
        }

        long totalMs = System.currentTimeMillis() - totalStart;
        return new EvalSummary(results, 0, 0, 0, 0, processedCount, skipped, totalMs);
    }

    private String findGroundTruthInFolder(File imgFile) {
        String base = imgFile.getAbsolutePath();
        // Try removing extension
        int dot = base.lastIndexOf('.');
        String noExt = (dot > 0) ? base.substring(0, dot) : base;

        String[] possible = { noExt + ".txt", base + ".txt", noExt + ".gt.txt" };
        for (String p : possible) {
            File f = new File(p);
            if (f.exists()) {
                try {
                    java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(f));
                    String line = reader.readLine();
                    reader.close();
                    return (line != null) ? line.trim() : "";
                } catch (IOException ignored) {}
            }
        }
        return null;
    }

    private void logTermination() {
        android.util.Log.i("EvalEngine", "Execution terminated by user.");
    }

    /**
     * Run evaluation with ground-truth TSV.
     * TSV format: image_path\tground_truth_text  (same as your training .txt)
     * Computes CER / WER / CRR / WRR exactly as VIT_CTC.py validate().
     */
    public EvalSummary runEvaluation(String tsvPath,
                                     String imageFolder,
                                     ProgressCallback cb,
                                     CsvResultWriter csvWriter) throws OrtException, IOException {
        List<String[]> samples = parseTSV(tsvPath);
        if (samples.isEmpty())
            return new EvalSummary(new ArrayList<>(), 0,0,0,0, 0, 0, 0);

        if (cb != null) cb.onProgress(0, samples.size());

        List<ImageResult> results = new ArrayList<>();
        int skipped = 0;
        double totalCER = 0, totalWER = 0, totalCRR = 0, totalWRR = 0;
        long totalStart = System.currentTimeMillis();

        for (int i = 0; i < samples.size(); i++) {
            if ((cb != null && cb.isCancelled()) || Thread.interrupted()) {
                logTermination();
                break;
            }
            String imgPath = samples.get(i)[0];
            String gt      = samples.get(i)[1];

            // Resolve relative paths if an image folder is provided
            if (imageFolder != null && !imageFolder.isEmpty()) {
                File imgFile = new File(imgPath);
                if (!imgFile.isAbsolute()) {
                    imgPath = new File(imageFolder, imgPath).getAbsolutePath();
                } else if (!imgFile.exists()) {
                    // Try filename in selected folder
                    imgPath = new File(imageFolder, imgFile.getName()).getAbsolutePath();
                }
            }

            String fileName = new File(imgPath).getName();
            try {
                long t0   = System.currentTimeMillis();
                String pr = inferImage(imgPath);
                long ms   = System.currentTimeMillis() - t0;

                float cer = computeCER(pr, gt);
                float wer = computeWER(pr, gt);
                totalCER += cer;  totalWER += wer;
                totalCRR += (1 - cer); totalWRR += (1 - wer);

                ImageResult res = new ImageResult(imgPath, pr, gt, cer, wer, ms);

                if (csvWriter != null) {
                    csvWriter.writeRow(res, true);
                    if (cb != null) cb.onProgressWithDetails(
                        csvWriter.getRowsWritten(), samples.size(), "Written: " + fileName);
                } else {
                    results.add(res);
                }
            } catch (Exception e) {
                skipped++;
                android.util.Log.e("EvalEngine", "Error evaluating sample " + i, e);
            }
            if (cb != null) cb.onProgress(i + 1, samples.size());
        }

        int n = results.size() > 0 ? results.size()
                : (csvWriter != null ? csvWriter.getRowsWritten() : 0);
        long totalMs = System.currentTimeMillis() - totalStart;
        return new EvalSummary(
                results,
                n > 0 ? (float)(totalCER / n) : 0f,
                n > 0 ? (float)(totalWER / n) : 0f,
                n > 0 ? (float)(totalCRR / n) : 0f,
                n > 0 ? (float)(totalWRR / n) : 0f,
                n, skipped, totalMs);
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            // DO NOT close env here. It's a singleton.
        } catch (Exception ignored) {}
    }

    // ── Inference ────────────────────────────────────────────────────────────

    private String inferImage(String imgPathOrUri) throws OrtException {
        float[] input = preprocessImage(imgPathOrUri);
        return runORT(input);
    }

    private String runORT(float[] inputData) throws OrtException {
        long[] shape = {1, CHANNELS, IMG_H, IMG_W};
        FloatBuffer buf = FloatBuffer.wrap(inputData);
        OnnxTensor tensor = null;
        OrtSession.Result result = null;
        try {
            tensor = OnnxTensor.createTensor(env, buf, shape);
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("image", tensor);

            result = session.run(inputs);
            float[][][] logProbs = (float[][][]) result.get(0).getValue();
            return greedyDecode(logProbs[0]);
        } finally {
            if (result != null) result.close();
            if (tensor != null) tensor.close();
        }
    }

    // ── Preprocessing — mirrors VIT_CTC.py preprocess_image() ch=2 ──────────

    private float[] preprocessImage(String imgPathOrUri) {
        // Decode at a downsampled resolution instead of full camera resolution.
        BitmapFactory.Options boundsOpts = new BitmapFactory.Options();
        boundsOpts.inJustDecodeBounds = true;
        
        try {
            if (imgPathOrUri.startsWith("content://")) {
                try (InputStream is = context.getContentResolver().openInputStream(Uri.parse(imgPathOrUri))) {
                    BitmapFactory.decodeStream(is, null, boundsOpts);
                }
            } else {
                BitmapFactory.decodeFile(imgPathOrUri, boundsOpts);
            }
        } catch (IOException e) {
            throw new RuntimeException("Cannot read image bounds: " + imgPathOrUri, e);
        }

        int inSample = 1;
        int halfH = boundsOpts.outHeight / 2, halfW = boundsOpts.outWidth / 2;
        while ((halfH / inSample) >= IMG_H * 4 && (halfW / inSample) >= IMG_W) {
            inSample *= 2; 
        }
        BitmapFactory.Options decodeOpts = new BitmapFactory.Options();
        decodeOpts.inSampleSize = inSample;
        decodeOpts.inPreferredConfig = Bitmap.Config.RGB_565; // Lighter bitmap configuration

        Bitmap bmp = null;
        try {
            if (imgPathOrUri.startsWith("content://")) {
                try (InputStream is = context.getContentResolver().openInputStream(Uri.parse(imgPathOrUri))) {
                    bmp = BitmapFactory.decodeStream(is, null, decodeOpts);
                }
            } else {
                bmp = BitmapFactory.decodeFile(imgPathOrUri, decodeOpts);
            }
        } catch (IOException e) {
             throw new RuntimeException("Cannot decode: " + imgPathOrUri, e);
        }
        
        if (bmp == null) throw new RuntimeException("Cannot decode: " + imgPathOrUri);

        Bitmap gray = toGrayscale(bmp);
        bmp.recycle();

        int srcW = gray.getWidth(), srcH = gray.getHeight();
        int newH = IMG_H;
        int newW = Math.max(1, (int)(srcW * ((float) newH / srcH)));
        newW     = Math.min(newW, IMG_W);

        Bitmap resized = Bitmap.createScaledBitmap(gray, newW, newH, true);
        gray.recycle(); // ← free immediately

        int[] pixels = new int[newH * newW];
        resized.getPixels(pixels, 0, newW, 0, 0, newW, newH);
        resized.recycle(); // ← free immediately

        // Channel 0: normalised grayscale (white-padded right)
        float[] ch0 = new float[IMG_H * IMG_W];
        java.util.Arrays.fill(ch0, 1.0f); // white padding normalised
        for (int row = 0; row < newH; row++) {
            for (int col = 0; col < newW; col++) {
                int px  = pixels[row * newW + col];
                float l = (Color.red(px)*0.299f
                         + Color.green(px)*0.587f
                         + Color.blue(px)*0.114f) / 255.0f;
                ch0[row * IMG_W + col] = (l - NORM_MEAN) / NORM_STD;
            }
        }

        // Channel 1: Otsu binarised
        float[] ch1 = otsuBinarise(ch0, IMG_H, IMG_W);

        float[] dual = new float[2 * IMG_H * IMG_W];
        System.arraycopy(ch0, 0, dual, 0,             IMG_H * IMG_W);
        System.arraycopy(ch1, 0, dual, IMG_H * IMG_W, IMG_H * IMG_W);
        return dual;
    }

    private float[] otsuBinarise(float[] normData, int H, int W) {
        int[] hist  = new int[256];
        int total   = H * W;
        for (float v : normData) {
            int pix = Math.max(0, Math.min(255,
                      (int)Math.round((v * NORM_STD + NORM_MEAN) * 255)));
            hist[pix]++;
        }
        double sum = 0;
        for (int i = 0; i < 256; i++) sum += i * hist[i];
        double sumB = 0, wB = 0, maxVar = 0;
        int threshold = 0;
        for (int t = 0; t < 256; t++) {
            wB += hist[t]; if (wB == 0) continue;
            double wF = total - wB; if (wF == 0) break;
            sumB += t * hist[t];
            double mB = sumB / wB, mF = (sum - sumB) / wF;
            double var = wB * wF * (mB - mF) * (mB - mF);
            if (var > maxVar) { maxVar = var; threshold = t; }
        }
        final int thr = threshold;
        float[] result = new float[normData.length];
        for (int i = 0; i < normData.length; i++) {
            float raw = (normData[i] * NORM_STD + NORM_MEAN) * 255.0f;
            float bin = raw >= thr ? 255.0f : 0.0f;
            result[i] = (bin / 255.0f - NORM_MEAN) / NORM_STD;
        }
        return result;
    }

    private Bitmap toGrayscale(Bitmap src) {
        Bitmap out = Bitmap.createBitmap(src.getWidth(), src.getHeight(),
                                          Bitmap.Config.RGB_565);
        android.graphics.Canvas canvas = new android.graphics.Canvas(out);
        android.graphics.ColorMatrix cm = new android.graphics.ColorMatrix();
        cm.setSaturation(0);
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setColorFilter(new android.graphics.ColorMatrixColorFilter(cm));
        canvas.drawBitmap(src, 0, 0, paint);
        return out;
    }

    // ── CTC Decode ───────────────────────────────────────────────────────────

    private String greedyDecode(float[][] logProbs) {
        StringBuilder sb = new StringBuilder();
        int prev = -1;
        int blankCount = 0;
        final int SPACE_THRESHOLD = 12;

        for (float[] t : logProbs) {
            int best = argmax(t);
            if (best == blankId) {
                blankCount++;
            } else {
                if (best != prev) {
                    if (blankCount >= SPACE_THRESHOLD && sb.length() > 0) {
                        sb.append(" ");
                    }
                    sb.append(id2char.getOrDefault(best, ""));
                }
                blankCount = 0;
            }
            prev = best;
        }
        return sb.toString().replace("_", " ").trim();
    }

    private int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }

    // ── Metrics — mirrors VIT_CTC.py validate() ─────────────────────────────

    /**
     * Character Error Rate = edit_distance(pred, gt) / len(gt)
     */
    private float computeCER(String pred, String gt) {
        if (gt.isEmpty()) return pred.isEmpty() ? 0f : 1f;
        return (float) editDistance(pred, gt) / gt.length();
    }

    /**
     * Word Error Rate = edit_distance(pred_words, gt_words) / len(gt_words)
     */
    private float computeWER(String pred, String gt) {
        String[] pw = pred.trim().split("\\s+");
        String[] gw = gt.trim().split("\\s+");
        if (gw.length == 0 || (gw.length == 1 && gw[0].isEmpty()))
            return (pw.length == 0 || (pw.length == 1 && pw[0].isEmpty())) ? 0f : 1f;
        return (float) editDistance(pw, gw) / gw.length;
    }

    /** Levenshtein on char arrays */
    private int editDistance(String a, String b) {
        int m = a.length(), n = b.length();
        int[] dp = new int[n + 1];
        for (int j = 0; j <= n; j++) dp[j] = j;
        for (int i = 1; i <= m; i++) {
            int prev = dp[0]; dp[0] = i;
            for (int j = 1; j <= n; j++) {
                int temp = dp[j];
                dp[j] = a.charAt(i-1) == b.charAt(j-1)
                        ? prev
                        : 1 + Math.min(prev, Math.min(dp[j], dp[j-1]));
                prev = temp;
            }
        }
        return dp[n];
    }

    /** Levenshtein on word arrays */
    private int editDistance(String[] a, String[] b) {
        int m = a.length, n = b.length;
        int[] dp = new int[n + 1];
        for (int j = 0; j <= n; j++) dp[j] = j;
        for (int i = 1; i <= m; i++) {
            int prev = dp[0]; dp[0] = i;
            for (int j = 1; j <= n; j++) {
                int temp = dp[j];
                dp[j] = a[i-1].equals(b[j-1])
                        ? prev
                        : 1 + Math.min(prev, Math.min(dp[j], dp[j-1]));
                prev = temp;
            }
        }
        return dp[n];
    }

    // ── TSV parser ───────────────────────────────────────────────────────────

    private List<String[]> parseTSV(String tsvPath) throws IOException {
        List<String[]> out = new ArrayList<>();
        java.io.BufferedReader br = new java.io.BufferedReader(
                new java.io.FileReader(tsvPath));
        String line;
        while ((line = br.readLine()) != null) {
            int tab = line.indexOf('\t');
            if (tab < 0) continue;
            out.add(new String[]{ line.substring(0, tab),
                                  line.substring(tab + 1) });
        }
        br.close();
        return out;
    }

    // ── ORT session ──────────────────────────────────────────────────────────

    private OrtSession createSession(Context ctx) throws OrtException, IOException {
        File modelFile = copyAssetToCache(ctx, MODEL_ASSET);
        try (OrtSession.SessionOptions opts = new OrtSession.SessionOptions()) {
            opts.setIntraOpNumThreads(4);
            opts.setInterOpNumThreads(2);
            return env.createSession(modelFile.getAbsolutePath(), opts);
        }
    }

    // ── Label map ────────────────────────────────────────────────────────────

    private Map<Integer, String> loadLabelMap(Context ctx) throws IOException {
        InputStream is = ctx.getAssets().open(LABELMAP_ASSET);
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384];
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        byte[] bytes = buffer.toByteArray();
        is.close();
        Map<Integer, String> map = new HashMap<>();
        try {
            JSONObject root = new JSONObject(new String(bytes, "UTF-8"));
            JSONObject id2c = root.getJSONObject("id2char");
            Iterator<String> keys = id2c.keys();
            while (keys.hasNext()) {
                String k = keys.next();
                map.put(Integer.parseInt(k), id2c.getString(k));
            }
        } catch (Exception e) {
            throw new IOException("Failed to parse label_map.json: " + e.getMessage());
        }
        return map;
    }

    private int findBlankId(Map<Integer, String> map) {
        for (Map.Entry<Integer, String> e : map.entrySet())
            if ("<BLANK>".equals(e.getValue())) return e.getKey();
        int max = 0;
        for (int k : map.keySet()) if (k > max) max = k;
        return max;
    }

    private File copyAssetToCache(Context ctx, String name) throws IOException {
        File out = new File(ctx.getCacheDir(), name);
        long assetLength = -1;
        try (AssetFileDescriptor afd = ctx.getAssets().openFd(name)) {
            assetLength = afd.getLength();
        } catch (IOException ignored) {
        }
        if (out.exists() && out.length() > 0
                && (assetLength <= 0 || out.length() == assetLength)) {
            return out;
        }
        try (InputStream in = ctx.getAssets().open(name);
             FileOutputStream os = new FileOutputStream(out)) {
            byte[] buf = new byte[8192]; int n;
            while ((n = in.read(buf)) != -1) os.write(buf, 0, n);
        }
        return out;
    }

    // ── Callback interface ───────────────────────────────────────────────────

    public interface ProgressCallback {
        void onProgress(int done, int total);
        default void onProgressWithDetails(int done, int total, String currentItem) {
            onProgress(done, total);
        }
        boolean isCancelled();
    }
}
