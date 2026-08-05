package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

/**
 * TextDetector
 * ============
 * Implements a deep-learned text detector using a DBNet ONNX model.
 * Handles the inference via ONNX Runtime and post-processing via OpenCV.
 *
 * This class is currently a stub because the underlying DBNet model asset 
 * (det_model.onnx) is not bundled in this version. It reports isReady() = false,
 * which triggers Page mode to fall back to the classical OpenCV pipeline.
 */
public final class TextDetector implements AutoCloseable {

    private static final String TAG = "TextDetector";
    private static final String MODEL_ASSET = "det_model.onnx";

    private OrtSession session;
    private OrtEnvironment env;
    private boolean isReady;

    public TextDetector(Context context) {
        try {
            this.env = OrtEnvironment.getEnvironment();
            
            // Check if model exists in assets
            if (assetExists(context, MODEL_ASSET)) {
                Log.i(TAG, "DBNet model found but loader not fully implemented yet.");
                this.isReady = false;
            } else {
                this.isReady = false;
                Log.i(TAG, "DBNet model asset (" + MODEL_ASSET + ") not found; learned detector inactive.");
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize TextDetector: " + e.getMessage());
            this.isReady = false;
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

    /**
     * Detects word boxes in the given bitmap using DBNet.
     * @param src Full page bitmap.
     * @return List of detected boxes in reading order.
     */
    public List<WordSegmenter.WordBox> detect(Bitmap src) {
        // If isReady is false, this should not be called.
        Log.d(TAG, "detect() called on bitmap of size: " + (src != null ? src.getWidth() + "x" + src.getHeight() : "null"));
        return new ArrayList<>();
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (Exception e) {
            Log.e(TAG, "Error closing TextDetector", e);
        }
    }
}
