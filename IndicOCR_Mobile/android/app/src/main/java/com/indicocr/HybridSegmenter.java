package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * HybridSegmenter
 * ===============
 * Compatibility wrapper used by evaluation/test flows. Segmentation is now
 * TextBPN-only; the older classical path is left commented below for easy
 * inspection, but it is not executed.
 */
public final class HybridSegmenter {

    private static final String TAG = "HybridSegmenter";

    private final TextDetector textDetector;
    private boolean useLearnedDetector = true;

    public HybridSegmenter(Context context) {
        this.textDetector = new TextDetector(context);
        if (!textDetector.isReady()) {
            Log.e(TAG, "TextBPN model unavailable; segmentation disabled.");
            useLearnedDetector = false;
        }
    }

    /** Keeps the existing API, but Page segmentation remains TextBPN-only. */
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

        if (!useLearnedDetector) {
            long elapsedMs = (System.nanoTime() - start) / 1_000_000;
            return new Result(new ArrayList<>(), "textbpn_unavailable", elapsedMs);
        }

        List<WordSegmenter.WordBox> boxes = textDetector.detect(src);
        Log.i(TAG, "TextBPN detector found " + boxes.size() + " boxes");
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        return new Result(boxes, "textbpn", elapsedMs);

        // Previous classical segmentation fallback intentionally disabled:
        // List<WordSegmenter.WordBox> boxes = WordSegmenter.segment(src);
        // Log.i(TAG, "Classical segmenter found " + boxes.size() + " boxes");
        // return new Result(boxes, "classical", elapsedMs);
    }

    public void close() {
        textDetector.close();
    }

    public static final class Result {
        public final List<WordSegmenter.WordBox> boxes;
        public final String detectorType; // "textbpn" or "textbpn_unavailable"
        public final long detectionLatencyMs;

        Result(List<WordSegmenter.WordBox> boxes, String detectorType, long detectionLatencyMs) {
            this.boxes = boxes;
            this.detectorType = detectorType;
            this.detectionLatencyMs = detectionLatencyMs;
        }
    }
}
