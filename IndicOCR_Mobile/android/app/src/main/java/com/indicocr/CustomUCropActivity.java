package com.indicocr;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.yalantis.ucrop.UCropActivity;
import com.yalantis.ucrop.view.GestureCropImageView;
import com.yalantis.ucrop.view.OverlayView;
import com.yalantis.ucrop.view.TransformImageView;
import com.yalantis.ucrop.view.UCropView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Field;

/**
 * CustomUCropActivity — fixes the crop box so it:
 *   1. Auto-positions onto the actual text content (not just the whole
 *      image) — detected via the same Otsu-threshold approach as
 *      WordSegmenter, but collapsed to one bounding box instead of
 *      per-word boxes, since this only needs a starting crop rect.
 *   2. Can be dragged freely to any size/position with no grid snap
 *   3. Corner handles reach all the way to image edges
 *
 * If content detection fails or finds nothing confident, it falls back
 * to fitting the whole image (the previous behaviour) — this is always
 * just a starting point the user can still drag/resize afterward.
 *
 * All field names are verified against UCrop source:
 *   OverlayView:       mCropViewRect, mCropRectMinSize, mCropRectCornerTouchAreaLineLength,
 *                      mTouchPointThreshold, mFreestyleCropMode, mShowCropGrid
 *   CropImageView:     mCropRect, mMaxResultImageSizeX/Y, mImageToWrapCropBoundsAnimDuration
 *   TransformImageView: mCurrentImageCorners, mBitmapLaidOut, getViewBitmap() [public]
 *
 * REQUIRES: OpenCV native library already loaded process-wide (MainActivity
 * calls OpenCVLoader.initLocal() before this activity is ever launched, and
 * that load persists for the app's process — no re-init needed here).
 */
public class CustomUCropActivity extends UCropActivity {

    private static final String TAG = "CustomUCrop";

    // ── Retry config ──────────────────────────────────────────────────────
    private static final int POLL_INTERVAL_MS = 80;   // how often to poll for bitmap ready
    private static final int MAX_POLLS        = 40;   // 40 × 80ms = 3.2s max wait

    // ── Content-detection tuning ─────────────────────────────────────────
    private static final double KERNEL_WIDTH_FACTOR  = 0.9;
    private static final double KERNEL_HEIGHT_FACTOR = 0.4;
    private static final int    MIN_KERNEL_W = 6,  MAX_KERNEL_W = 40;
    private static final int    MIN_KERNEL_H = 3,  MAX_KERNEL_H = 15;

    private static final float  PAD_X_RATIO = 0.08f;         // horizontal padding around detected box
    private static final float  PAD_Y_RATIO = 0.18f;         // extra vertical padding — Indic matras/descenders extend further than Latin text
    private static final float  MIN_BOX_PX  = 10f;          // Minimum size to trust a detection

    private final Handler handler = new Handler(Looper.getMainLooper());
    private int pollCount = 0;
    private boolean isAutoPositioned = false;
    private TransformImageView.TransformImageListener originalListener;

    // ─────────────────────────────────────────────────────────────────────

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Ensure the rotation dial is visible by setting the state to ROTATE
        setWidgetState(com.yalantis.ucrop.R.id.state_rotate);
    }

    private void setWidgetState(int stateViewId) {
        try {
            java.lang.reflect.Method method = UCropActivity.class.getDeclaredMethod("setWidgetState", int.class);
            method.setAccessible(true);
            method.invoke(this, stateViewId);
        } catch (Exception e) {
            Log.e(TAG, "Failed to set widget state: " + e.getMessage());
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        pollCount = 0;
        handler.removeCallbacksAndMessages(null);

        UCropView uCropView = findViewById(com.yalantis.ucrop.R.id.ucrop);
        if (uCropView != null) {
            GestureCropImageView imageView = uCropView.getCropImageView();
            // Step 1: apply non-layout overrides immediately
            imageView.post(this::applyStaticOverrides);

            // Step 2: Wrap listener to trigger auto-positioning on load complete
            if (!isAutoPositioned) {
                wrapTransformImageListener(imageView);
            }
        }

        // Fallback polling in case listener doesn't fire or was already complete
        if (!isAutoPositioned) {
            pollUntilBitmapReady();
        }
    }

    private void wrapTransformImageListener(GestureCropImageView imageView) {
        originalListener = (TransformImageView.TransformImageListener) getFieldValue(imageView, "mTransformImageListener");
        imageView.setTransformImageListener(new TransformImageView.TransformImageListener() {
            @Override
            public void onRotate(float currentAngle) {
                if (originalListener != null) originalListener.onRotate(currentAngle);
            }

            @Override
            public void onScale(float currentScale) {
                if (originalListener != null) originalListener.onScale(currentScale);
            }

            @Override
            public void onLoadComplete() {
                if (originalListener != null) originalListener.onLoadComplete();
                if (!isAutoPositioned) {
                    Log.i(TAG, "onLoadComplete -> triggering auto-position");
                    fitCropBoxToImageBounds();
                }
            }

            @Override
            public void onLoadFailure(Exception e) {
                if (originalListener != null) originalListener.onLoadFailure(e);
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        handler.removeCallbacksAndMessages(null);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Step 1 — overrides that don't depend on layout being complete
    // ─────────────────────────────────────────────────────────────────────
    private void applyStaticOverrides() {
        UCropView uCropView = findViewById(com.yalantis.ucrop.R.id.ucrop);
        if (uCropView == null) return;

        GestureCropImageView imageView = uCropView.getCropImageView();
        OverlayView overlayView        = uCropView.getOverlayView();
        if (imageView == null || overlayView == null) return;

        // ── CropImageView / TransformImageView ───────────────────────────
        // No max output size restrictions
        setInt(imageView,   "mMaxResultImageSizeX",              0);
        setInt(imageView,   "mMaxResultImageSizeY",              0);
        // Kill the snap-back animation that pushes image away from box edges
        setLong(imageView,  "mImageToWrapCropBoundsAnimDuration", 0L);

        // ── OverlayView ───────────────────────────────────────────────────
        // Allow crop box to be as small as 0px
        setInt(overlayView, "mCropRectMinSize",                  0);
        Field f = findField(overlayView, "mCropRectMinSize");
        if (f != null) {
            try {
                Log.d(TAG, "mCropRectMinSize = " + f.getInt(overlayView));
            } catch (Exception e) {
                Log.e(TAG, "Couldn't read mCropRectMinSize", e);
            }
        } else {
            Log.e(TAG, "mCropRectMinSize field NOT FOUND");
        }
        // Increase corner touch target so handles are easy to grab at edges
        setInt(overlayView, "mTouchPointThreshold",              80);
        // Keep corner line length reasonable
        setInt(overlayView, "mCropRectCornerTouchAreaLineLength", 60);
        // Enable freestyle (free-form) crop — disabling grid snap
        setInt(overlayView, "mFreestyleCropMode",
               OverlayView.FREESTYLE_CROP_MODE_ENABLE);           // = 1
        // Hide inner grid lines (cleaner UX, less visual noise)
        setBoolean(overlayView, "mShowCropGrid", false);

        // ── FORCE NO ASPECT RATIO ──
        setFloat(imageView, "mTargetAspectRatio", 0f);
        // ── ALLOW UNLIMITED ZOOM OUT ──
        setFloat(imageView, "mMinScale", 0.001f);

        overlayView.setFreestyleCropMode(OverlayView.FREESTYLE_CROP_MODE_ENABLE);
        overlayView.postInvalidate();

        Log.i(TAG, "✅ Static overrides applied");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Step 2 — poll until mBitmapLaidOut = true, then position the crop box
    // ─────────────────────────────────────────────────────────────────────
    private void pollUntilBitmapReady() {
        handler.postDelayed(() -> {
            if (isBitmapLaidOut()) {
                Log.i(TAG, "✅ Bitmap laid out after " + pollCount + " polls — positioning crop box");
                fitCropBoxToImageBounds();
            } else if (pollCount < MAX_POLLS) {
                pollCount++;
                pollUntilBitmapReady();
            } else {
                Log.w(TAG, "⚠️ Bitmap never laid out — positioning anyway");
                fitCropBoxToImageBounds();
            }
        }, POLL_INTERVAL_MS);
    }

    /** Reads the protected mBitmapLaidOut flag from TransformImageView */
    private boolean isBitmapLaidOut() {
        UCropView uCropView = findViewById(com.yalantis.ucrop.R.id.ucrop);
        if (uCropView == null) return false;
        GestureCropImageView imageView = uCropView.getCropImageView();
        if (imageView == null) return false;
        Object val = getFieldValue(imageView, "mBitmapLaidOut");
        return Boolean.TRUE.equals(val);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Step 3 — auto-position the crop box onto detected text content,
    //           falling back to the full image if detection isn't confident
    // ─────────────────────────────────────────────────────────────────────
    private void fitCropBoxToImageBounds() {
        UCropView uCropView = findViewById(com.yalantis.ucrop.R.id.ucrop);
        if (uCropView == null) return;

        GestureCropImageView imageView = uCropView.getCropImageView();
        OverlayView overlayView        = uCropView.getOverlayView();
        if (imageView == null || overlayView == null) return;

        // Ensure we have layout before measuring
        if (imageView.getWidth() <= 0 || imageView.getHeight() <= 0) {
            if (pollCount < MAX_POLLS) {
                pollCount++;
                handler.postDelayed(this::fitCropBoxToImageBounds, POLL_INTERVAL_MS);
            }
            return;
        }

        // Read internal transformation matrix
        Matrix currentMatrix = (Matrix) getFieldValue(imageView, "mCurrentImageMatrix");
        Bitmap bitmap = imageView.getViewBitmap();

        if (currentMatrix == null || bitmap == null) {
            Log.w(TAG, "Matrix or Bitmap null — retrying");
            handler.postDelayed(this::fitCropBoxToImageBounds, 120);
            return;
        }

        Rect contentPx = detectContentBounds(bitmap);
        RectF targetRect;

        if (contentPx != null) {
            Log.i(TAG, "Content detected in pixels: " + contentPx);
            RectF contentRectF = new RectF(contentPx.x, contentPx.y, contentPx.x + contentPx.width, contentPx.y + contentPx.height);
            
            targetRect = new RectF();
            currentMatrix.mapRect(targetRect, contentRectF);

            float padX = targetRect.width()  * PAD_X_RATIO;
            float padY = targetRect.height() * PAD_Y_RATIO;
            targetRect.inset(-padX, -padY);

            float[] corners = (float[]) getFieldValue(imageView, "mCurrentImageCorners");
            if (corners != null && corners.length >= 8) {
                RectF imgBounds = getBounds(corners);
                targetRect.intersect(imgBounds);
            }
            
            Log.i(TAG, "Content-based crop box: " + targetRect);
            applyFinalRect(imageView, overlayView, targetRect, true);
        } else {
            Log.i(TAG, "No content detected — fitting whole image");
            float[] corners = (float[]) getFieldValue(imageView, "mCurrentImageCorners");
            if (corners != null && corners.length >= 8) {
                targetRect = getBounds(corners);
                applyFinalRect(imageView, overlayView, targetRect, false);
            }
        }
    }

    private void applyFinalRect(GestureCropImageView imageView, OverlayView overlayView, RectF targetRect, boolean wasContent) {


        if (targetRect == null || targetRect.width() < MIN_BOX_PX) {
            Log.w(TAG, "applyFinalRect: targetRect invalid");
            return;
        }

        // We use a small delay to ensure we're the LAST ones to touch the state
        handler.postDelayed(() -> {
            Log.i(TAG, "Applying final rect (" + (wasContent ? "content" : "full") + "): " + targetRect);
            
            // 1. Temporarily kill wrap animations so our snap is instant and isn't "fought" by the library
            setLong(imageView, "mImageToWrapCropBoundsAnimDuration", 0L);

            // 2. Inform BOTH views of the new aspect ratio
            float ratio = targetRect.width() / targetRect.height();
            setFloat(imageView, "mTargetAspectRatio", ratio);
            
            // Use reflection for OverlayView because setTargetAspectRatio() internally calls setupCropBounds()
            // which would overwrite our mCropViewRect. We set the field directly.
            setFloat(overlayView, "mTargetAspectRatio", ratio);

            // 3. Update CropImageView's internal rect
            imageView.setCropRect(targetRect);
            
            // 4. Update OverlayView's internal rect (final field, use reflection)
            RectF overlayRect = (RectF) getFieldValue(overlayView, "mCropViewRect");
            if (overlayRect != null) {
                overlayRect.set(targetRect);
            }
            
            overlayView.postInvalidate();
            imageView.postInvalidate();
            
            isAutoPositioned = true;

            if (wasContent) {
                int percent = (int) ((targetRect.width() * targetRect.height()) / 
                                    (imageView.getWidth() * imageView.getHeight()) * 100);
                Toast.makeText(this, "Snapped to " + percent + "% of image", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "No text found - using full image", Toast.LENGTH_SHORT).show();
            }
        }, 200); // Increased to 200ms to be safer
    }

    private RectF getBounds(float[] corners) {
        float minValX = corners[0], maxValX = corners[0];
        float minValY = corners[1], maxValY = corners[1];
        for (int i = 2; i < 8; i += 2) {
            minValX = Math.min(minValX, corners[i]);
            maxValX = Math.max(maxValX, corners[i]);
            minValY = Math.min(minValY, corners[i + 1]);
            maxValY = Math.max(maxValY, corners[i + 1]);
        }
        return new RectF(minValX, minValY, maxValX, maxValY);
    }


    private static CLAHE cachedClahe;

    private static synchronized CLAHE getClahe() {
        if (cachedClahe == null) {
            cachedClahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        }
        return cachedClahe;
    }

    /** CLAHE (local contrast) + Otsu threshold + Adaptive Dilation + contour extent. */
    private Rect detectContentBounds(Bitmap bmp) {
        if (bmp == null) return null;
        
        Mat rgba = null;
        Mat gray = null;
        Mat enhanced = null;
        Mat binary = null;
        Mat kernel = null;
        Mat dilated = null;
        Mat hierarchy = null;
        List<MatOfPoint> contours = new ArrayList<>();
        
        Bitmap detectBmp = null;
        
        try {
            // Downscale for detection (exactly like WordSegmenter) — ensures kernels
            // remain proportional and speeds up threshold/contours significantly.
            double scale = 1.0;
            int maxDim = Math.max(bmp.getWidth(), bmp.getHeight());
            if (maxDim > 1024) {
                scale = 1024.0 / maxDim;
            }

            if (bmp.getConfig() == Bitmap.Config.HARDWARE || scale < 1.0) {
                detectBmp = Bitmap.createScaledBitmap(bmp, (int) (bmp.getWidth() * scale), (int) (bmp.getHeight() * scale), true);
            } else {
                detectBmp = bmp;
            }

            rgba = new Mat();
            Utils.bitmapToMat(detectBmp, rgba);

            gray = new Mat();
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);

            enhanced = new Mat();
            getClahe().apply(gray, enhanced);
            
            Imgproc.GaussianBlur(enhanced, enhanced, new Size(3, 3), 0);

            binary = new Mat();
            Imgproc.threshold(enhanced, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            // Measure typical glyph size to tune dilation
            double medianGlyphHeight = estimateMedianGlyphHeight(binary);
            if (medianGlyphHeight <= 0) medianGlyphHeight = 20;

            int kernelW = clamp((int) Math.round(medianGlyphHeight * KERNEL_WIDTH_FACTOR), MIN_KERNEL_W, MAX_KERNEL_W);
            int kernelH = clamp((int) Math.round(medianGlyphHeight * KERNEL_HEIGHT_FACTOR), MIN_KERNEL_H, MAX_KERNEL_H);
            
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelW, kernelH));
            dilated = new Mat();
            Imgproc.dilate(binary, dilated, kernel);

            hierarchy = new Mat();
            Imgproc.findContours(dilated, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            if (contours.isEmpty()) return null;

            int minX = Integer.MAX_VALUE, maxX = 0;
            int minY = Integer.MAX_VALUE, maxY = 0;
            
            // Edge guard: ignore anything exactly on the boundary
            boolean found = false;
            for (MatOfPoint c : contours) {
                org.opencv.core.Rect r = Imgproc.boundingRect(c);
                
                // Noise filter: ignore very small speckles
                if (r.width < medianGlyphHeight * 0.3 && r.height < medianGlyphHeight * 0.3) continue;
                
                // Ignore the boundary itself if it was detected as a massive contour
                if (r.width > detectBmp.getWidth() * 0.98 && r.height > detectBmp.getHeight() * 0.98) continue;

                minX = Math.min(minX, r.x);
                minY = Math.min(minY, r.y);
                maxX = Math.max(maxX, r.x + r.width);
                maxY = Math.max(maxY, r.y + r.height);
                found = true;
            }

            if (!found) return null;

            // Map back to original bitmap scale
            double inv = 1.0 / scale;
            return new Rect(
                (int) Math.round(minX * inv), (int) Math.round(minY * inv),
                (int) Math.round(maxX * inv), (int) Math.round(maxY * inv)
            );
        } catch (Exception e) {
            Log.e(TAG, "Content detection failed: " + e.getMessage());
            return null;
        } finally {
            if (detectBmp != null && detectBmp != bmp) detectBmp.recycle();
            if (rgba != null) rgba.release();
            if (gray != null) gray.release();
            if (enhanced != null) enhanced.release();
            if (binary != null) binary.release();
            if (kernel != null) kernel.release();
            if (dilated != null) dilated.release();
            if (hierarchy != null) hierarchy.release();
            for (MatOfPoint c : contours) c.release();
        }
    }

    private double estimateMedianGlyphHeight(Mat binary) {
        Mat lightKernel = null;
        Mat lightDilate = null;
        Mat hierarchy = null;
        List<MatOfPoint> glyphContours = new ArrayList<>();
        
        try {
            lightKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            lightDilate = new Mat();
            Imgproc.dilate(binary, lightDilate, lightKernel);

            hierarchy = new Mat();
            Imgproc.findContours(lightDilate, glyphContours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            List<Integer> heights = new ArrayList<>();
            for (MatOfPoint c : glyphContours) {
                org.opencv.core.Rect r = Imgproc.boundingRect(c);
                if (r.height >= 4 && r.height <= 200) heights.add(r.height);
            }
            if (heights.isEmpty()) return -1;
            Collections.sort(heights);
            return heights.get(heights.size() / 2);
        } finally {
            if (lightKernel != null) lightKernel.release();
            if (lightDilate != null) lightDilate.release();
            if (hierarchy != null) hierarchy.release();
            for (MatOfPoint c : glyphContours) c.release();
        }
    }

    private int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Reflection helpers — verified field names only
    // ─────────────────────────────────────────────────────────────────────

    private void setInt(Object obj, String name, int value) {
        Field f = findField(obj, name);
        if (f == null) { Log.w(TAG, "⚠️ int field not found: " + name); return; }
        try { f.set(obj, value); Log.i(TAG, "✅ " + name + " = " + value); }
        catch (Exception e) { Log.e(TAG, "setInt " + name + ": " + e.getMessage()); }
    }

    private void setLong(Object obj, String name, long value) {
        Field f = findField(obj, name);
        if (f == null) { Log.w(TAG, "⚠️ long field not found: " + name); return; }
        try { f.set(obj, value); Log.i(TAG, "✅ " + name + " = " + value); }
        catch (Exception e) { Log.e(TAG, "setLong " + name + ": " + e.getMessage()); }
    }

    private void setFloat(Object obj, String name, float value) {
        Field f = findField(obj, name);
        if (f == null) { Log.w(TAG, "⚠️ float field not found: " + name); return; }
        try { f.set(obj, value); Log.i(TAG, "✅ " + name + " = " + value); }
        catch (Exception e) { Log.e(TAG, "setFloat " + name + ": " + e.getMessage()); }
    }

    private void setBoolean(Object obj, String name, boolean value) {
        Field f = findField(obj, name);
        if (f == null) { Log.w(TAG, "⚠️ boolean field not found: " + name); return; }
        try { f.set(obj, value); Log.i(TAG, "✅ " + name + " = " + value); }
        catch (Exception e) { Log.e(TAG, "setBoolean " + name + ": " + e.getMessage()); }
    }

    private Object getFieldValue(Object obj, String name) {
        Field f = findField(obj, name);
        if (f == null) return null;
        try { return f.get(obj); }
        catch (Exception e) { Log.e(TAG, "getField " + name + ": " + e.getMessage()); return null; }
    }

    /** Walks the full superclass chain to find a declared field */
    private Field findField(Object obj, String name) {
        Class<?> cls = obj.getClass();
        while (cls != null) {
            try {
                Field f = cls.getDeclaredField(name);
                f.setAccessible(true);
                return f;
            } catch (NoSuchFieldException e) {
                cls = cls.getSuperclass();
            }
        }
        return null;
    }
}