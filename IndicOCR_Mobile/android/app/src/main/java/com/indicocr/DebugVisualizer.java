package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * DebugVisualizer
 * ===============
 * Utility to draw detected bounding boxes on an image and save it to disk.
 * Used for visual verification of segmentation quality.
 */
public final class DebugVisualizer {

    private static final String TAG = "DebugVisualizer";

    private DebugVisualizer() {}

    /**
     * Draws the provided word boxes onto a copy of the bitmap and saves it to internal storage.
     * @return Absolute path to the saved debug image, or null on failure.
     */
    public static String drawAndSave(Context ctx, Bitmap src, List<WordSegmenter.WordBox> words, String tag) {
        if (src == null || words == null || ctx == null) return null;

        Bitmap debugBmp = null;
        try {
            // DOWNSCALE for debug: No need to save a 12MP debug overlay
            int maxDim = 1200;
            int w = src.getWidth();
            int h = src.getHeight();
            float scale = 1.0f;
            if (Math.max(w, h) > maxDim) {
                scale = (float) maxDim / Math.max(w, h);
                w = Math.round(w * scale);
                h = Math.round(h * scale);
            }

            // OPTIMIZATION: Use RGB_565 to use 50% less memory for the debug overlay
            debugBmp = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);

            Canvas canvas = new Canvas(debugBmp);
            // Draw the scaled original first
            android.graphics.Rect srcRect = new android.graphics.Rect(0, 0, src.getWidth(), src.getHeight());
            android.graphics.Rect dstRect = new android.graphics.Rect(0, 0, w, h);
            canvas.drawBitmap(src, srcRect, dstRect, null);

            Paint boxPaint = new Paint();
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(Math.max(2, w / 400f));
            boxPaint.setColor(Color.RED);

            Paint textPaint = new Paint();
            textPaint.setColor(Color.RED);
            textPaint.setTextSize(Math.max(10, w / 50f));
            textPaint.setFakeBoldText(true);

            for (int i = 0; i < words.size(); i++) {
                WordSegmenter.WordBox wBox = words.get(i);
                if (wBox != null && wBox.rect != null) {
                    // Map box to scaled coordinates
                    float left = wBox.rect.left * scale;
                    float top = wBox.rect.top * scale;
                    float right = wBox.rect.right * scale;
                    float bottom = wBox.rect.bottom * scale;
                    
                    canvas.drawRect(left, top, right, bottom, boxPaint);
                    
                    // Draw word index and line index for deep debugging
                    String label = i + ":" + wBox.lineIndex;
                    canvas.drawText(label, left, Math.max(textPaint.getTextSize(), top - 2), textPaint);
                }
            }

            File cacheDir = ctx.getExternalCacheDir();
            if (cacheDir == null) cacheDir = ctx.getCacheDir();
            
            File dir = new File(cacheDir, "debug_segmentation");
            if (!dir.exists() && !dir.mkdirs()) {
                Log.e(TAG, "Failed to create debug directory");
                return null;
            }

            File file = new File(dir, "debug_" + tag + "_" + System.currentTimeMillis() + ".jpg");
            try (FileOutputStream out = new FileOutputStream(file)) {
                debugBmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
                Log.i(TAG, "Saved debug overlay: " + file.getAbsolutePath());
                return file.getAbsolutePath();
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to create or save debug image: " + e.getMessage());
            return null;
        } finally {
            if (debugBmp != null) debugBmp.recycle();
        }
    }
}
