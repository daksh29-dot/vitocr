package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * IndicOCREngine
 * ==============
 * Drop-in Android inference engine for the ViT-CTC Indic OCR model.
 *
 * Usage:
 *   IndicOCREngine ocr = new IndicOCREngine(context);
 *   String text = ocr.recognise(bitmap);   // bitmap = word-image crop
 *   ocr.close();
 *
 * Assets expected in  src/main/assets/:
 *   vit_ctc.onnx       (or vit_ctc_int8.onnx for the quantised version)
 *   label_map.json
 *
 * Gradle dependency (add to app/build.gradle):
 *   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
 */
public class IndicOCREngine implements AutoCloseable {

    // ── Constants matching your training config ──────────────────────────────
    private static final int   IMG_H        = 32;
    private static final int   IMG_W        = 256;
    private static final int   CHANNELS     = 2;          // 1=grayscale, 2=dual
    private static final int   SEQUENCE_LEN = 512;        // 8 * 64 patches
    private static final float NORM_MEAN    = 0.5f;
    private static final float NORM_STD     = 0.5f;

    private static final String MODEL_ASSET    = "vit_ctc.onnx";
    private static final String LABELMAP_ASSET = "label_map.json";

    // Caps how many word crops go into one ONNX inference call. Large
    // enough to meaningfully amortize per-call overhead, small enough that
    // a dense page doesn't allocate one huge tensor at once.
    private static final int MAX_BATCH_SIZE = 16;

    // ── Fields ────────────────────────────────────────────────────────────────
    private final OrtEnvironment env;
    private final OrtSession     session;
    private final Map<Integer, String> id2char;
    private final int blankId;

    // ── Constructor ───────────────────────────────────────────────────────────
    public IndicOCREngine(Context ctx) throws OrtException, IOException {
        env     = OrtEnvironment.getEnvironment();
        session = createSession(ctx);
        id2char = loadLabelMap(ctx);
        blankId = findBlankId(id2char);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Recognise text in a word-image crop.
     * @param src Any Bitmap (colour or grayscale). Will be resized internally.
     * @return Recognised Unicode string.
     */
    public String recognise(Bitmap src) throws OrtException {
        if (src == null) return "";
        float[] inputData = preprocessBitmap(src);
        return runInference(inputData);
    }

    /**
     * Recognises many word crops in ONE inference call instead of one call
     * per word. Existing per-word looping code (e.g. recognising every word
     * on a page one at a time) pays fixed per-call overhead N times over;
     * batching amortizes that overhead across all N crops at once, which is
     * where most of the real speed win on a dense page comes from — the
     * model itself is unchanged.
     *
     * Chunked internally at MAX_BATCH_SIZE so a very dense page (100+ words)
     * doesn't allocate one huge tensor and spike memory — several
     * medium-sized batches instead of one giant one or many tiny ones.
     *
     * @param crops Word-image crops, in any order — results come back in the same order.
     * @return Recognised text for each crop, same order and length as the input list.
     */
    public List<String> recogniseBatch(List<Bitmap> crops) throws OrtException {
        List<String> results = new ArrayList<>(crops.size());
        if (crops.isEmpty()) return results;

        for (int start = 0; start < crops.size(); start += MAX_BATCH_SIZE) {
            int end = Math.min(start + MAX_BATCH_SIZE, crops.size());
            results.addAll(runBatch(crops.subList(start, end)));
        }
        return results;
    }

    private List<String> runBatch(List<Bitmap> crops) throws OrtException {
        int n = crops.size();
        int perImageSize = CHANNELS * IMG_H * IMG_W;
        float[] batchData = new float[n * perImageSize];

        for (int i = 0; i < n; i++) {
            Bitmap crop = crops.get(i);
            float[] single = (crop != null) ? preprocessBitmap(crop) : new float[perImageSize];
            System.arraycopy(single, 0, batchData, i * perImageSize, perImageSize);
        }

        return runInferenceBatch(batchData, n);
    }

    /**
     * Recognise text from a JPEG/PNG file path.
     */
    public String recogniseFile(String imagePath) throws OrtException, IOException {
        Bitmap bmp = BitmapFactory.decodeFile(imagePath);
        if (bmp == null) throw new IOException("Cannot decode: " + imagePath);
        return recognise(bmp);
    }

    @Override
    public void close() {
        try { if (session != null) session.close(); } catch (OrtException ignored) {}
        if (env     != null) env.close();
    }

    // ── Preprocessing ─────────────────────────────────────────────────────────

    /**
     * Resize → pad → normalise → return flat float[1 * CHANNELS * IMG_H * IMG_W].
     *
     * Mirrors VIT_CTC.py  preprocess_image():
     *   1. Convert to grayscale
     *   2. Resize height→32, keep aspect ratio, cap width at 256
     *   3. Pad right with white (1.0 before norm → -1.0 after norm)
     *   4. Normalise: pixel = (pixel/255 - 0.5) / 0.5
     */
    private float[] preprocessBitmap(Bitmap original) {
        // Step 0 – Add 3px white padding to each side
        Bitmap src = addWhitePadding(original, 3);

        // Step 1 – resize preserving aspect ratio, height=32, max_width=256
        int srcW = src.getWidth();
        int srcH = src.getHeight();
        int newH = IMG_H;
        int newW = Math.max(1, (int) (srcW * ((float) newH / srcH)));
        newW     = Math.min(newW, IMG_W);

        Bitmap resized = Bitmap.createScaledBitmap(src, newW, newH, true);

        // Step 2 – convert the SMALL bitmap to grayscale
        Bitmap gray = toGrayscale(resized);

        // Cleanup intermediate bitmaps
        if (resized != src) resized.recycle();
        if (src != original) src.recycle();

        // Step 3 – build float array with right-padding
        float[] data = new float[IMG_H * IMG_W];
        // Fill with white (1.0) by default
        java.util.Arrays.fill(data, 1.0f);

        int[] pixels = new int[newH * newW];
        gray.getPixels(pixels, 0, newW, 0, 0, newW, newH);
        gray.recycle();

        for (int row = 0; row < newH; row++) {
            for (int col = 0; col < newW; col++) {
                int px   = pixels[row * newW + col];
                // grayscale value via luminance
                float lum = (Color.red(px) * 0.299f
                           + Color.green(px) * 0.587f
                           + Color.blue(px) * 0.114f) / 255.0f;
                float norm = (lum - NORM_MEAN) / NORM_STD;
                data[row * IMG_W + col] = norm;   // channel 0
            }
        }

        // If CHANNELS == 2, add Otsu-binarised channel (simplified threshold)
        if (CHANNELS == 2) {
            float[] otsu = otsuBinarise(data, IMG_H, IMG_W);
            float[] dual = new float[2 * IMG_H * IMG_W];
            System.arraycopy(data, 0, dual, 0,              IMG_H * IMG_W);
            System.arraycopy(otsu, 0, dual, IMG_H * IMG_W, IMG_H * IMG_W);
            return dual;
        }

        return data;
    }

    /** Otsu global threshold on normalised [-1,1] data, returns binarised [-1,1]. */
    private float[] otsuBinarise(float[] normData, int H, int W) {
        // Convert back to [0,255] for histogram
        int[] hist = new int[256];
        int total  = H * W;
        for (float v : normData) {
            int pix = Math.round((v * NORM_STD + NORM_MEAN) * 255);
            pix = Math.max(0, Math.min(255, pix));
            hist[pix]++;
        }
        // Otsu's method
        double sum = 0;
        for (int i = 0; i < 256; i++) sum += i * hist[i];

        double sumB = 0, wB = 0, maxVar = 0;
        int threshold = 0;
        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;
            double wF = total - wB;
            if (wF == 0) break;
            sumB += t * hist[t];
            double mB  = sumB / wB;
            double mF  = (sum - sumB) / wF;
            double var = wB * wF * (mB - mF) * (mB - mF);
            if (var > maxVar) { maxVar = var; threshold = t; }
        }

        float[] result = new float[H * W];
        for (int i = 0; i < normData.length; i++) {
            float rawPix = (normData[i] * NORM_STD + NORM_MEAN) * 255.0f;
            float bin    = rawPix >= threshold ? 255.0f : 0.0f;
            result[i]    = (bin / 255.0f - NORM_MEAN) / NORM_STD;
        }
        return result;
    }

    private Bitmap toGrayscale(Bitmap src) {
        Bitmap out = Bitmap.createBitmap(src.getWidth(), src.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        android.graphics.ColorMatrix cm = new android.graphics.ColorMatrix();
        cm.setSaturation(0);
        Paint paint = new Paint();
        paint.setColorFilter(new android.graphics.ColorMatrixColorFilter(cm));
        canvas.drawBitmap(src, 0, 0, paint);
        return out;
    }

    /**
     * Adds white padding to all sides of a bitmap.
     */
    private Bitmap addWhitePadding(Bitmap src, int padding) {
        int newW = src.getWidth() + 2 * padding;
        int newH = src.getHeight() + 2 * padding;
        Bitmap out = Bitmap.createBitmap(newW, newH, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        canvas.drawColor(Color.WHITE);
        canvas.drawBitmap(src, padding, padding, null);
        return out;
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    private String runInference(float[] inputData) throws OrtException {
        long[] shape = {1, CHANNELS, IMG_H, IMG_W};

        FloatBuffer buf = FloatBuffer.wrap(inputData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buf, shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image", inputTensor);

        OrtSession.Result result = session.run(inputs);

        // Output is log_softmax [1, T, num_classes]
        float[][][] logProbs = (float[][][]) result.get(0).getValue();

        inputTensor.close();
        result.close();

        return greedyDecode(logProbs[0]);
    }

    /** Same as runInference but for a batch of N pre-stacked, pre-preprocessed crops in one call. */
    private List<String> runInferenceBatch(float[] batchData, int batchSize) throws OrtException {
        long[] shape = {batchSize, CHANNELS, IMG_H, IMG_W};

        FloatBuffer buf = FloatBuffer.wrap(batchData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buf, shape);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("image", inputTensor);

        OrtSession.Result result = session.run(inputs);

        // Output is log_softmax [batchSize, T, num_classes]
        float[][][] logProbs = (float[][][]) result.get(0).getValue();

        inputTensor.close();

        List<String> texts = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            texts.add(greedyDecode(logProbs[i]));
        }

        result.close();
        return texts;
    }

    /** Greedy CTC decode (argmax + collapse repeats + remove blank). */
    private String greedyDecode(float[][] logProbs) {
        StringBuilder sb  = new StringBuilder();
        int prevToken = -1;
        for (float[] timestep : logProbs) {
            int best = argmax(timestep);
            if (best != prevToken && best != blankId) {
                String ch = id2char.getOrDefault(best, "");
                sb.append(ch);
            }
            prevToken = best;
        }
        return sb.toString();
    }

    private int argmax(float[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[best]) best = i;
        }
        return best;
    }

    // ── Session creation ──────────────────────────────────────────────────────

    private OrtSession createSession(Context ctx) throws OrtException, IOException {
        // Copy model from assets to internal storage (ORT needs a file path on Android)
        File modelFile = copyAssetToCache(ctx, MODEL_ASSET);

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // Use NNAPI (Neural Network API) when available – this is the Android
        // ARM Neural Engine accelerator. Falls back to CPU automatically.
        try {
            opts.addNnapi();
            System.out.println("[IndicOCR] NNAPI acceleration enabled");
        } catch (OrtException e) {
            System.out.println("[IndicOCR] NNAPI not available, using CPU");
        }
        opts.setIntraOpNumThreads(4);
        opts.setInterOpNumThreads(2);

        return env.createSession(modelFile.getAbsolutePath(), opts);
    }

    // ── Label map ─────────────────────────────────────────────────────────────

    private Map<Integer, String> loadLabelMap(Context ctx) throws IOException {
        InputStream is  = ctx.getAssets().open(LABELMAP_ASSET);
        byte[] bytes;
        try {
            bytes = readAllBytes(is);
        } finally {
            is.close();
        }
        String json     = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);

        Map<Integer, String> map = new HashMap<>();
        try {
            JSONObject root   = new JSONObject(json);
            JSONObject id2c   = root.getJSONObject("id2char");
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

    private byte[] readAllBytes(InputStream inputStream) throws IOException {
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384];
        while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        return buffer.toByteArray();
    }

    private int findBlankId(Map<Integer, String> map) {
        for (Map.Entry<Integer, String> e : map.entrySet()) {
            if ("<BLANK>".equals(e.getValue())) return e.getKey();
        }
        // Fallback: blank is typically the last id
        int maxId = 0;
        for (int k : map.keySet()) if (k > maxId) maxId = k;
        return maxId;
    }

    // ── Asset helpers ─────────────────────────────────────────────────────────

    private File copyAssetToCache(Context ctx, String assetName) throws IOException {
        File outFile = new File(ctx.getCacheDir(), assetName);
        if (outFile.exists()) return outFile;   // already copied

        try (InputStream in  = ctx.getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) != -1) out.write(buf, 0, n);
        }
        return outFile;
    }
}