package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.util.List;

/**
 * HybridSegmenter
 * ===============
 * Single entry point MainActivity calls instead of WordSegmenter directly.
 * Tries the learned {@link TextDetector} (ONNX DBNet) first; falls back to
 * the classical {@link WordSegmenter} OpenCV pipeline if the model isn't
 * bundled, fails to load, or returns nothing usable for a given page.
 *
 * Construction loads an ONNX session, which does file I/O — build this off
 * the UI thread (same pattern as MainActivity's initOCREngine()).
 */
public final class HybridSegmenter {

    private static final String TAG = "HybridSegmenter";

    /** Minimum boxes a "successful" detector run must produce before we
     *  trust it over falling back — guards against a model that loads fine
     *  but returns a degenerate empty result on a particular page. */
    private static final int MIN_ACCEPTABLE_BOXES = 1;

    private final TextDetector textDetector;
    private boolean useLearnedDetector = true;

    public HybridSegmenter(Context context) {
        this.textDetector = new TextDetector(context);
        if (!textDetector.isReady()) {
            Log.i(TAG, "DBNet model unavailable — using classical WordSegmenter only.");
            useLearnedDetector = false;
        }
    }

    /** Lets a settings toggle force classical-only mode for A/B comparison. */
    public void setUseLearnedDetector(boolean enabled) {
        this.useLearnedDetector = enabled && textDetector.isReady();
    }

    public boolean isUsingLearnedDetector() {
        return useLearnedDetector;
    }

    /**
     * Segments a page/region into word boxes, in reading order.
     * @return boxes plus which detector produced them, for benchmark logging.
     */
    public Result segment(Bitmap src) {
        long start = System.nanoTime();

        if (useLearnedDetector) {
            List<WordSegmenter.WordBox> boxes = textDetector.detect(src);
            Log.i(TAG, "DBNet detector found " + boxes.size() + " boxes");
            if (boxes.size() >= MIN_ACCEPTABLE_BOXES) {
                long elapsedMs = (System.nanoTime() - start) / 1_000_000;
                return new Result(boxes, "dbnet", elapsedMs);
            }
            Log.w(TAG, "DBNet returned no usable boxes, falling back to classical for this page.");
        }

        List<WordSegmenter.WordBox> boxes = WordSegmenter.segment(src);
        Log.i(TAG, "Classical segmenter found " + boxes.size() + " boxes");
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        return new Result(boxes, "classical", elapsedMs);
    }

    public void close() {
        textDetector.close();
    }

    /** Bundles boxes with which pipeline produced them and how long it took —
     *  feed straight into BenchmarkLogger. */
    public static final class Result {
        public final List<WordSegmenter.WordBox> boxes;
        public final String detectorType; // "dbnet" or "classical"
        public final long detectionLatencyMs;

        Result(List<WordSegmenter.WordBox> boxes, String detectorType, long detectionLatencyMs) {
            this.boxes = boxes;
            this.detectorType = detectorType;
            this.detectionLatencyMs = detectionLatencyMs;
        }
    }
}
