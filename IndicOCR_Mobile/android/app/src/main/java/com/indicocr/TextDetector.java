package com.indicocr;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * TextDetector
 * ============
 * Runs the bundled TextBPN segmentation ONNX model and converts its text
 * probability map into word boxes for the existing OCR crop pipeline.
 */
public final class TextDetector implements AutoCloseable {

    private static final String TAG = "TextDetector";
    private static final String MODEL_ASSET = "textbpn_full_64.onnx";

    private static final int MODEL_SIZE = 640;
    private static final int POLY_COUNT = 64;
    private static final int POLY_POINTS = 20;
    private static final float TEXT_THRESHOLD = 0.45f;
    private static final int MIN_BOX_SIDE = 5;
    private static final double MIN_BOX_AREA_RATIO = 0.00002;
    private static final double MAX_BOX_AREA_RATIO = 0.65;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private OrtSession session;
    private boolean isReady;

    public TextDetector(Context context) {
        env = OrtEnvironment.getEnvironment();
        try {
            if (!assetExists(context, MODEL_ASSET)) {
                Log.i(TAG, "TextBPN model asset (" + MODEL_ASSET + ") not found; learned detector inactive.");
                isReady = false;
                return;
            }

            File modelFile = copyAssetToCache(context, MODEL_ASSET);
            try (OrtSession.SessionOptions cpuOpts = createSessionOptions()) {
                session = env.createSession(modelFile.getAbsolutePath(), cpuOpts);
                Log.i(TAG, "TextBPN detector session created with CPU.");
            }

            isReady = true;
            Log.i(TAG, "TextBPN detector ready.");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize TextDetector: " + e.getMessage(), e);
            isReady = false;
        }
    }

    public static boolean isModelBundled(Context ctx) {
        try (InputStream is = ctx.getAssets().open(MODEL_ASSET)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private boolean assetExists(Context context, String fileName) {
        try (InputStream is = context.getAssets().open(fileName)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    public boolean isReady() {
        return isReady;
    }

    private OrtSession.SessionOptions createSessionOptions() throws OrtException {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(2);
        opts.setInterOpNumThreads(1);
        return opts;
    }

    public List<WordSegmenter.WordBox> detect(Bitmap src) {
        if (!isReady || src == null || src.isRecycled()) {
            return new ArrayList<>();
        }

        LetterboxInput input = null;
        OnnxTensor imageTensor = null;
        OnnxTensor initPolysTensor = null;
        OnnxTensor validMaskTensor = null;
        OrtSession.Result result = null;

        try {
            input = preprocess(src);
            imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input.data),
                    new long[]{1, 3, MODEL_SIZE, MODEL_SIZE});
            initPolysTensor = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[POLY_COUNT * POLY_POINTS * 2]),
                    new long[]{POLY_COUNT, POLY_POINTS, 2});
            validMaskTensor = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(new float[POLY_COUNT]),
                    new long[]{POLY_COUNT});

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input", imageTensor);
            inputs.put("init_polys", initPolysTensor);
            inputs.put("valid_mask", validMaskTensor);

            result = session.run(inputs);
            
            // Access the tensor data directly to avoid allocating nested arrays for a 640x640 map.
            OnnxTensor outputTensor = (OnnxTensor) result.get(0);
            java.nio.FloatBuffer buffer = outputTensor.getFloatBuffer();
            float[] fyPreds = new float[buffer.remaining()];
            buffer.get(fyPreds);

            List<Rect> modelBoxes = boxesFromSegmentation(fyPreds, input);
            List<WordSegmenter.WordBox> wordBoxes = buildTextBpnWordBoxes(src, modelBoxes);

            return wordBoxes;
        } catch (Throwable t) {
            Log.e(TAG, "TextBPN detection failed: " + t.getMessage(), t);
            return new ArrayList<>();
        } finally {
            if (result != null) result.close();
            closeQuietly(imageTensor);
            closeQuietly(initPolysTensor);
            closeQuietly(validMaskTensor);
            if (input != null && input.bitmap != null) input.bitmap.recycle();
        }
    }

    private LetterboxInput preprocess(Bitmap src) {
        float scale = Math.min(MODEL_SIZE / (float) src.getWidth(), MODEL_SIZE / (float) src.getHeight());
        int scaledW = Math.max(1, Math.round(src.getWidth() * scale));
        int scaledH = Math.max(1, Math.round(src.getHeight() * scale));
        int padX = (MODEL_SIZE - scaledW) / 2;
        int padY = (MODEL_SIZE - scaledH) / 2;

        Bitmap modelBitmap = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(modelBitmap);
        canvas.drawColor(Color.WHITE);
        
        // Use a Matrix for efficient scaling and translation in one step
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postScale(scale, scale);
        matrix.postTranslate(padX, padY);
        canvas.drawBitmap(src, matrix, new Paint(Paint.FILTER_BITMAP_FLAG));

        int[] pixels = new int[MODEL_SIZE * MODEL_SIZE];
        modelBitmap.getPixels(pixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE);

        float[] data = new float[3 * MODEL_SIZE * MODEL_SIZE];
        int plane = MODEL_SIZE * MODEL_SIZE;
        for (int i = 0; i < pixels.length; i++) {
            int px = pixels[i];
            data[i] = (((px >> 16) & 0xFF) / 255.0f - MEAN[0]) / STD[0];
            data[plane + i] = (((px >> 8) & 0xFF) / 255.0f - MEAN[1]) / STD[1];
            data[(2 * plane) + i] = ((px & 0xFF) / 255.0f - MEAN[2]) / STD[2];
        }

        return new LetterboxInput(data, modelBitmap, scale, padX, padY);
    }

    private List<Rect> boxesFromSegmentation(float[] fyPreds, LetterboxInput input) {
        byte[] maskData = new byte[MODEL_SIZE * MODEL_SIZE];
        int planeSize = MODEL_SIZE * MODEL_SIZE;

        if (fyPreds.length < planeSize) {
            Log.w(TAG, "TextBPN output too small: " + fyPreds.length);
            return new ArrayList<>();
        }

        int channelCount = Math.max(1, fyPreds.length / planeSize);
        if (channelCount < 4) {
            Log.w(TAG, "Unexpected TextBPN channel count: " + channelCount);
        }

        int scoreOffset = chooseSegmentationPlane(fyPreds, Math.min(channelCount, 2), planeSize);
        if (scoreOffset < 0) {
            Log.w(TAG, "TextBPN did not produce a usable text mask.");
            return new ArrayList<>();
        }

        int foregroundPixels = 0;
        for (int i = 0; i < planeSize; i++) {
            float score = normalizeScore(fyPreds[scoreOffset + i]);
            if (score >= TEXT_THRESHOLD) {
                maskData[i] = (byte) 255;
                foregroundPixels++;
            }
        }

        float coverage = foregroundPixels / (float) planeSize;
        if (coverage > MAX_BOX_AREA_RATIO) {
            Log.w(TAG, "TextBPN mask covers " + coverage + " of the page; rejecting oversized mask.");
            return new ArrayList<>();
        }

        Mat mask = new Mat(MODEL_SIZE, MODEL_SIZE, CvType.CV_8UC1);
        mask.put(0, 0, maskData);

        Mat closeKernel = null;
        Mat dilateKernel = null;
        Mat hierarchy = null;
        List<MatOfPoint> contours = new ArrayList<>();

        try {
            closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 3));
            dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(9, 3));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, closeKernel);
            Imgproc.dilate(mask, mask, dilateKernel);

            hierarchy = new Mat();
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } finally {
            if (mask != null) mask.release();
            if (closeKernel != null) closeKernel.release();
            if (dilateKernel != null) dilateKernel.release();
            if (hierarchy != null) hierarchy.release();
        }

        int srcArea = MODEL_SIZE * MODEL_SIZE;
        double minArea = srcArea * MIN_BOX_AREA_RATIO;
        List<Rect> boxes = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            org.opencv.core.Rect cvRect = Imgproc.boundingRect(contour);
            contour.release();
            if (cvRect.width < MIN_BOX_SIDE || cvRect.height < MIN_BOX_SIDE) continue;
            if ((double) cvRect.width * cvRect.height < minArea) continue;

            Rect mapped = mapToSource(cvRect, input);
            if (mapped.width() > 0 && mapped.height() > 0) {
                boxes.add(mapped);
            }
        }

        return sortReadingOrder(boxes);
    }

    private float normalizeScore(float value) {
        if (value >= 0.0f && value <= 1.0f) return value;
        return (float) (1.0 / (1.0 + Math.exp(-value)));
    }

    private int chooseSegmentationPlane(float[] fyPreds, int channelLimit, int planeSize) {
        int bestChannel = -1;
        float bestDistance = Float.MAX_VALUE;
        final float targetCoverage = 0.12f;

        for (int ch = 0; ch < channelLimit; ch++) {
            int offset = ch * planeSize;
            int foreground = 0;
            for (int i = 0; i < planeSize; i++) {
                if (normalizeScore(fyPreds[offset + i]) >= TEXT_THRESHOLD) foreground++;
            }

            float coverage = foreground / (float) planeSize;
            Log.i(TAG, "TextBPN channel " + ch + " mask coverage=" + coverage);
            if (coverage <= 0.0f || coverage > MAX_BOX_AREA_RATIO) continue;

            float distance = Math.abs(coverage - targetCoverage);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestChannel = ch;
            }
        }

        return bestChannel < 0 ? -1 : bestChannel * planeSize;
    }

    private Rect mapToSource(org.opencv.core.Rect box, LetterboxInput input) {
        int left = Math.round((box.x - input.padX) / input.scale);
        int top = Math.round((box.y - input.padY) / input.scale);
        int right = Math.round((box.x + box.width - input.padX) / input.scale);
        int bottom = Math.round((box.y + box.height - input.padY) / input.scale);

        int marginX = Math.max(2, Math.round((right - left) * 0.03f));
        int marginY = Math.max(2, Math.round((bottom - top) * 0.08f));

        return new Rect(
                Math.max(0, left - marginX),
                Math.max(0, top - marginY),
                Math.max(0, right + marginX),
                Math.max(0, bottom + marginY));
    }

    private List<WordSegmenter.WordBox> buildTextBpnWordBoxes(Bitmap src, List<Rect> boxes) {
        List<WordSegmenter.WordBox> candidates = new ArrayList<>();

        for (Rect box : boxes) {
            Rect bounded = boundToSource(box, src);
            if (bounded.width() <= 0 || bounded.height() <= 0) continue;

            // Previous classical oversized-box refinement intentionally disabled:
            // if (bounded.width() > bounded.height() * OVERSIZED_ASPECT_RATIO) {
            //     Bitmap crop = WordSegmenter.crop(src, bounded);
            //     List<WordSegmenter.WordBox> split = WordSegmenter.segment(crop);
            //     ...
            // }

            candidates.add(new WordSegmenter.WordBox(bounded, 0, 1.0f));
        }

        List<Rect> rects = new ArrayList<>();
        for (WordSegmenter.WordBox candidate : candidates) {
            rects.add(candidate.rect);
        }
        return toWordBoxes(sortReadingOrder(rects));
    }

    private Rect boundToSource(Rect box, Bitmap src) {
        return new Rect(
                Math.max(0, Math.min(src.getWidth(), box.left)),
                Math.max(0, Math.min(src.getHeight(), box.top)),
                Math.max(0, Math.min(src.getWidth(), box.right)),
                Math.max(0, Math.min(src.getHeight(), box.bottom)));
    }

    private List<WordSegmenter.WordBox> toWordBoxes(List<Rect> boxes) {
        List<WordSegmenter.WordBox> out = new ArrayList<>();
        int line = -1;
        int lastTop = Integer.MIN_VALUE;
        int lastBottom = Integer.MIN_VALUE;

        for (Rect box : boxes) {
            if (line < 0 || !sameLine(box, lastTop, lastBottom)) {
                line++;
                lastTop = box.top;
                lastBottom = box.bottom;
            } else {
                lastTop = Math.min(lastTop, box.top);
                lastBottom = Math.max(lastBottom, box.bottom);
            }
            out.add(new WordSegmenter.WordBox(box, line, 1.0f));
        }
        return out;
    }

    private List<Rect> sortReadingOrder(List<Rect> boxes) {
        boxes.sort((a, b) -> a.top - b.top);

        List<List<Rect>> lines = new ArrayList<>();
        List<int[]> lineBounds = new ArrayList<>();
        for (Rect box : boxes) {
            int bestLine = -1;
            for (int i = 0; i < lines.size(); i++) {
                int[] bounds = lineBounds.get(i);
                if (sameLine(box, bounds[0], bounds[1])) {
                    bestLine = i;
                    break;
                }
            }

            if (bestLine == -1) {
                List<Rect> line = new ArrayList<>();
                line.add(box);
                lines.add(line);
                lineBounds.add(new int[]{box.top, box.bottom});
            } else {
                lines.get(bestLine).add(box);
                int[] bounds = lineBounds.get(bestLine);
                bounds[0] = Math.min(bounds[0], box.top);
                bounds[1] = Math.max(bounds[1], box.bottom);
            }
        }

        List<Integer> lineOrder = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) lineOrder.add(i);
        lineOrder.sort((a, b) -> lineBounds.get(a)[0] - lineBounds.get(b)[0]);

        List<Rect> ordered = new ArrayList<>();
        for (int i : lineOrder) {
            List<Rect> line = lines.get(i);
            line.sort((a, b) -> a.left - b.left);
            ordered.addAll(line);
        }
        return ordered;
    }

    private boolean sameLine(Rect box, int top, int bottom) {
        int overlap = Math.min(box.bottom, bottom) - Math.max(box.top, top);
        int minHeight = Math.min(box.height(), bottom - top);
        return minHeight > 0 && overlap > minHeight * 0.45f;
    }

    private File copyAssetToCache(Context ctx, String assetName) throws IOException {
        File outFile = new File(ctx.getCacheDir(), assetName);
        long assetLength = -1;
        try (AssetFileDescriptor afd = ctx.getAssets().openFd(assetName)) {
            assetLength = afd.getLength();
        } catch (IOException ignored) {
        }
        if (outFile.exists() && outFile.length() > 0
                && (assetLength <= 0 || outFile.length() == assetLength)) {
            return outFile;
        }

        try (InputStream in = ctx.getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) != -1) {
                out.write(buf, 0, n);
            }
        }
        return outFile;
    }

    private void closeQuietly(OnnxTensor tensor) {
        if (tensor == null) return;
        try {
            tensor.close();
        } catch (Exception ignored) {
        }
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing TextDetector", e);
        }
    }

    private static final class LetterboxInput {
        final float[] data;
        final Bitmap bitmap;
        final float scale;
        final int padX;
        final int padY;

        LetterboxInput(float[] data, Bitmap bitmap, float scale, int padX, int padY) {
            this.data = data;
            this.bitmap = bitmap;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }
    }
}
