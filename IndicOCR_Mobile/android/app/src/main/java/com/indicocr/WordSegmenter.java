package com.indicocr;

import android.graphics.Bitmap;
import android.graphics.Rect;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * WordSegmenter
 * =============
 * Splits a page/region image into individual word-crop bounding boxes,
 * sorted in natural reading order (top-to-bottom lines, left-to-right
 * within each line), ready to feed one-by-one into IndicOCREngine.recognise().
 *
 * v2 changes (fixing real accuracy + speed problems from v1):
 *   1. ADAPTIVE dilation kernel — v1 used a fixed 18x5 kernel guess, which
 *      only worked for one specific font size / camera distance and caused
 *      merged or split words on anything else. v2 runs a lightweight first
 *      pass to measure the actual median glyph height in THIS image, then
 *      sizes the dilation kernel proportionally. Self-tuning per photo.
 *   2. SCALE-RELATIVE noise filtering — v1's MIN_WORD_WIDTH/HEIGHT/AREA were
 *      fixed pixel counts, which silently broke if the photo resolution or
 *      camera distance changed. v2 filters relative to the measured glyph
 *      height instead, so it stays correct across zoom levels.
 *   3. DOWNSCALE BEFORE DETECTION — segmentation doesn't need full camera
 *      resolution to find word boundaries; only the final crops fed to OCR
 *      do. v2 runs OpenCV ops on a capped-resolution copy (faster: threshold
 *      /dilate/contours all scale with pixel count) and maps boxes back to
 *      full-resolution coordinates before cropping, so OCR input quality is
 *      unaffected.
 *   4. RUNNING-BOUNDS LINE GROUPING — v1 compared every new box only against
 *      each line's first box, which could misgroup lines on skewed photos.
 *      v2 expands each line's bounds as boxes are added, tolerating drift.
 *   5. MERGE PASS — a second pass merges same-line boxes with an
 *      abnormally small gap between them, catching residual split-word
 *      cases the dilation step alone didn't fully bridge.
 *   6. OVERSIZED-REGION RE-SPLIT — on dense, justified text, inter-word
 *      spacing can be nearly as tight as intra-word spacing, so even the
 *      adaptive kernel can still merge multiple words into one blob. Any
 *      box far wider than a typical word is re-processed on its own with
 *      a much smaller kernel to split it back into word-sized pieces —
 *      this is what actually prevents multi-word blobs from being fed
 *      into a recognition model that was only ever trained on single
 *      words (the cause of garbled, decode-noise-like output).
 *
 * SETUP REQUIRED (one-time, unchanged from v1):
 *   1. app/build.gradle:  implementation 'org.opencv:opencv:4.9.0'
 *   2. Once, before first use (e.g. top of MainActivity.onCreate()):
 *          if (!org.opencv.android.OpenCVLoader.initLocal()) { ... }
 *
 * USAGE (unchanged):
 *   List<WordSegmenter.WordBox> words = WordSegmenter.segment(pageBitmap);
 *   for (WordSegmenter.WordBox w : words) {
 *       Bitmap crop = WordSegmenter.crop(pageBitmap, w.rect);
 *       String text = ocrEngine.recognise(crop);
 *   }
 */
public final class WordSegmenter {

    /** One detected word region on the source image, in reading order. */
    public static class WordBox {
        public final Rect rect;
        /** Which text line this word belongs to (0-based, top to bottom). */
        public final int lineIndex;
        /** Confidence score (0-1.0), e.g. from a learned detector. Classical is usually 1.0. */
        public final float confidence;

        public WordBox(Rect rect, int lineIndex) {
            this(rect, lineIndex, 1.0f);
        }

        public WordBox(Rect rect, int lineIndex, float confidence) {
            this.rect = rect;
            this.lineIndex = lineIndex;
            this.confidence = confidence;
        }
    }

    // ── Tunable knobs ────────────────────────────────────────────────────────

    /** Detection runs on a copy capped to this size (longest side, px) for speed. Boxes are scaled back to full resolution afterward. */
    private static final int MAX_DETECT_DIMENSION = 1400;

    // Dilation kernel is sized as a multiple of the measured median glyph
    // height, then clamped to a sane range. Increase the WIDTH_FACTOR if
    // words are still coming out split into separate letters; decrease it
    // if separate words are merging into one box.
    private static final double KERNEL_WIDTH_FACTOR  = 0.9;
    private static final double KERNEL_HEIGHT_FACTOR = 0.4;
    private static final int MIN_KERNEL_W = 6,  MAX_KERNEL_W = 40;
    private static final int MIN_KERNEL_H = 3,  MAX_KERNEL_H = 15;

    // Noise filtering, relative to median glyph height (scale-invariant).
    private static final double MIN_WORD_WIDTH_FACTOR  = 0.3;
    private static final double MIN_WORD_HEIGHT_FACTOR = 0.4;
    private static final double MIN_WORD_AREA_FACTOR   = 0.15; // × glyphHeight^2

    // Two boxes are considered "same line" if their vertical overlap is at
    // least this fraction of the shorter box's height.
    private static final double LINE_OVERLAP_RATIO = 0.5;

    // Second-pass merge: same-line boxes closer than this (× glyph height)
    // are merged — catches words dilation didn't fully bridge.
    private static final double MERGE_GAP_FACTOR = 0.35;

    // A box wider than this (× glyph height) is treated as suspicious —
    // almost certainly several words merged together, not one long word —
    // and gets re-processed with a much smaller kernel to split it up.
    private static final double MAX_WORD_WIDTH_FACTOR = 3.5;
    // Re-split kernel is this fraction of the normal kernel width — tight
    // enough to separate words that were nearly touching. Shrinks further
    // on each recursive pass (see splitOversizedBoxRecursive).
    private static final double RESPLIT_KERNEL_FACTOR = 0.4;
    // Safety cap so a pathological image can't recurse forever.
    private static final int MAX_RESPLIT_DEPTH = 3;

    private static CLAHE cachedClahe;

    private WordSegmenter() {}

    private static synchronized CLAHE getClahe() {
        if (cachedClahe == null) {
            cachedClahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        }
        return cachedClahe;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    public static List<WordBox> segment(Bitmap src) {
        double scale = 1.0;
        int maxDim = Math.max(src.getWidth(), src.getHeight());
        if (maxDim > MAX_DETECT_DIMENSION) {
            scale = MAX_DETECT_DIMENSION / (double) maxDim;
        }
        Bitmap detectBmp = scale < 1.0
                ? Bitmap.createScaledBitmap(src, (int) (src.getWidth() * scale), (int) (src.getHeight() * scale), true)
                : src;

        Mat rgba = new Mat();
        Mat gray = new Mat();
        Mat enhanced = new Mat();
        Mat binary = new Mat();
        Mat kernel = null;
        Mat dilated = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();

        try {
            Utils.bitmapToMat(detectBmp, rgba);
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            rgba.release();

            // CLAHE (local contrast enhancement) before thresholding
            getClahe().apply(gray, enhanced);
            gray.release();

            Imgproc.GaussianBlur(enhanced, enhanced, new Size(3, 3), 0);

            Imgproc.threshold(enhanced, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
            enhanced.release();

            // Pass 1: measure actual glyph size in this image (self-tuning).
            double medianGlyphHeight = estimateMedianGlyphHeight(binary);
            if (medianGlyphHeight <= 0) medianGlyphHeight = 20; // fallback if measurement failed

            // Pass 2: dilate with a kernel sized to what was actually measured.
            int kernelW = clamp((int) Math.round(medianGlyphHeight * KERNEL_WIDTH_FACTOR), MIN_KERNEL_W, MAX_KERNEL_W);
            int kernelH = clamp((int) Math.round(medianGlyphHeight * KERNEL_HEIGHT_FACTOR), MIN_KERNEL_H, MAX_KERNEL_H);
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelW, kernelH));
            Imgproc.dilate(binary, dilated, kernel);

            Imgproc.findContours(dilated, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            double minW = medianGlyphHeight * MIN_WORD_WIDTH_FACTOR;
            double minH = medianGlyphHeight * MIN_WORD_HEIGHT_FACTOR;
            double minArea = medianGlyphHeight * medianGlyphHeight * MIN_WORD_AREA_FACTOR;

            List<Rect> boxes = new ArrayList<>();
            for (MatOfPoint c : contours) {
                org.opencv.core.Rect r = Imgproc.boundingRect(c);
                if (r.width < minW || r.height < minH) continue;
                if ((double) r.width * r.height < minArea) continue;
                boxes.add(new Rect(r.x, r.y, r.x + r.width, r.y + r.height));
            }

            List<WordBox> ordered = sortReadingOrder(boxes);
            ordered = mergeCloseWords(ordered, medianGlyphHeight);
            ordered = splitOversizedBoxes(binary, ordered, medianGlyphHeight);
            ordered = scaleToOriginal(ordered, scale);

            return ordered;
        } catch (Exception e) {
            throw e;
        } finally {
            if (detectBmp != src) detectBmp.recycle();
            rgba.release();
            gray.release();
            enhanced.release();
            binary.release();
            if (kernel != null) kernel.release();
            dilated.release();
            hierarchy.release();
            for (MatOfPoint c : contours) c.release();
        }
    }

    /** Crops one detected word region out of the (full-resolution) source bitmap. Bounds-safe. */
    public static Bitmap crop(Bitmap src, Rect box) {
        int x = Math.max(0, box.left);
        int y = Math.max(0, box.top);
        int w = Math.min(box.width(), src.getWidth() - x);
        int h = Math.min(box.height(), src.getHeight() - y);
        if (w <= 0 || h <= 0) return null;
        return Bitmap.createBitmap(src, x, y, w, h);
    }

    // ── Glyph size estimation ────────────────────────────────────────────────

    /** Lightly dilates just enough to solidify strokes, then measures the median contour height across the page. */
    private static double estimateMedianGlyphHeight(Mat binary) {
        Mat lightKernel = null;
        Mat lightDilate = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> glyphContours = new ArrayList<>();
        
        try {
            lightKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Imgproc.dilate(binary, lightDilate, lightKernel);

            Imgproc.findContours(lightDilate, glyphContours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            List<Integer> heights = new ArrayList<>();
            for (MatOfPoint c : glyphContours) {
                org.opencv.core.Rect r = Imgproc.boundingRect(c);
                if (r.height >= 4 && r.height <= 200) heights.add(r.height); // drop extreme outliers
            }
            
            if (heights.isEmpty()) return -1;
            Collections.sort(heights);
            return heights.get(heights.size() / 2);
        } finally {
            if (lightKernel != null) lightKernel.release();
            lightDilate.release();
            hierarchy.release();
            for (MatOfPoint c : glyphContours) c.release();
        }
    }

    // ── Reading order (skew-tolerant) ───────────────────────────────────────

    private static List<WordBox> sortReadingOrder(List<Rect> boxes) {
        boxes.sort((a, b) -> a.top - b.top);

        List<List<Rect>> lines = new ArrayList<>();
        List<int[]> lineBounds = new ArrayList<>(); // running [top, bottom] per line

        for (Rect box : boxes) {
            int bestLine = -1;
            for (int li = 0; li < lines.size(); li++) {
                int[] b = lineBounds.get(li);
                int overlap = Math.min(box.bottom, b[1]) - Math.max(box.top, b[0]);
                int minHeight = Math.min(box.height(), b[1] - b[0]);
                if (minHeight > 0 && overlap > minHeight * LINE_OVERLAP_RATIO) {
                    bestLine = li;
                    break;
                }
            }
            if (bestLine == -1) {
                lines.add(new ArrayList<>());
                lineBounds.add(new int[]{box.top, box.bottom});
                bestLine = lines.size() - 1;
            } else {
                int[] b = lineBounds.get(bestLine);
                b[0] = Math.min(b[0], box.top);   // line bounds expand as boxes are added —
                b[1] = Math.max(b[1], box.bottom); // tolerates gradual skew across the line
            }
            lines.get(bestLine).add(box);
        }

        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) order.add(i);
        order.sort((a, b) -> lineBounds.get(a)[0] - lineBounds.get(b)[0]);

        List<WordBox> ordered = new ArrayList<>();
        int lineIndex = 0;
        for (int li : order) {
            List<Rect> line = lines.get(li);
            line.sort((a, b) -> a.left - b.left);
            for (Rect r : line) ordered.add(new WordBox(r, lineIndex));
            lineIndex++;
        }
        return ordered;
    }

    // ── Merge residual split words ───────────────────────────────────────────

    private static List<WordBox> mergeCloseWords(List<WordBox> words, double medianGlyphHeight) {
        List<WordBox> result = new ArrayList<>();
        int i = 0;
        while (i < words.size()) {
            int line = words.get(i).lineIndex;
            List<WordBox> lineWords = new ArrayList<>();
            while (i < words.size() && words.get(i).lineIndex == line) {
                lineWords.add(words.get(i));
                i++;
            }
            result.addAll(mergeLineWords(lineWords, medianGlyphHeight, line));
        }
        return result;
    }

    private static List<WordBox> mergeLineWords(List<WordBox> lineWords, double medianGlyphHeight, int lineIndex) {
        if (lineWords.size() <= 1) return lineWords;

        double mergeThreshold = medianGlyphHeight * MERGE_GAP_FACTOR;
        List<Rect> merged = new ArrayList<>();
        Rect current = lineWords.get(0).rect;

        for (int i = 1; i < lineWords.size(); i++) {
            Rect next = lineWords.get(i).rect;
            int gap = next.left - current.right;
            if (gap >= 0 && gap < mergeThreshold) {
                current = new Rect(
                        Math.min(current.left, next.left), Math.min(current.top, next.top),
                        Math.max(current.right, next.right), Math.max(current.bottom, next.bottom));
            } else {
                merged.add(current);
                current = next;
            }
        }
        merged.add(current);

        List<WordBox> out = new ArrayList<>();
        for (Rect r : merged) out.add(new WordBox(r, lineIndex));
        return out;
    }

    // ── Oversized-region re-split ────────────────────────────────────────────

    /**
     * Scans every box; anything much wider than a typical word gets
     * re-processed on its own with a tighter kernel to split it back into
     * word-sized pieces. This is the actual fix for dense/justified text
     * where inter-word gaps are nearly as small as intra-word gaps — the
     * main dilation pass alone can't tell them apart, but a second pass
     * restricted to just the suspicious region, with a smaller kernel,
     * usually can.
     *
     * Recursive: a blob merging 3+ words might not fully separate in one
     * pass, so each resulting piece is re-checked and split again if it's
     * still oversized, up to MAX_RESPLIT_DEPTH.
     */
    private static List<WordBox> splitOversizedBoxes(Mat binary, List<WordBox> boxes, double medianGlyphHeight) {
        List<WordBox> result = new ArrayList<>();
        for (WordBox w : boxes) {
            result.addAll(splitOversizedBoxRecursive(binary, w, medianGlyphHeight, 0));
        }
        return result;
    }

    private static List<WordBox> splitOversizedBoxRecursive(Mat binary, WordBox w, double medianGlyphHeight, int depth) {
        double maxWidth = medianGlyphHeight * MAX_WORD_WIDTH_FACTOR;
        if (w.rect.width() <= maxWidth || depth >= MAX_RESPLIT_DEPTH) {
            List<WordBox> single = new ArrayList<>();
            single.add(w);
            return single;
        }

        // Shrink the kernel further on each recursive pass — if the first
        // attempt at this size didn't separate the blob, a slightly looser
        // kernel won't either.
        double kernelFactor = RESPLIT_KERNEL_FACTOR * Math.pow(0.6, depth);
        List<Rect> subBoxes = resplitRegion(binary, w.rect, medianGlyphHeight, kernelFactor);

        if (subBoxes.size() <= 1) {
            // No improvement possible — keep the original rather than losing content.
            List<WordBox> single = new ArrayList<>();
            single.add(w);
            return single;
        }

        List<WordBox> out = new ArrayList<>();
        for (Rect r : subBoxes) {
            out.addAll(splitOversizedBoxRecursive(binary, new WordBox(r, w.lineIndex), medianGlyphHeight, depth + 1));
        }
        return out;
    }

    /** Re-runs threshold→dilate→contours on just one region, with a kernel tighter than the main pass (kernelFactor < 1.0). */
    private static List<Rect> resplitRegion(Mat binary, Rect box, double medianGlyphHeight, double kernelFactor) {
        List<Rect> out = new ArrayList<>();

        int x = Math.max(0, box.left);
        int y = Math.max(0, box.top);
        int w = Math.min(box.width(), binary.cols() - x);
        int h = Math.min(box.height(), binary.rows() - y);
        if (w <= 0 || h <= 0) return out;

        Mat region = null;
        Mat kernel = null;
        Mat dilated = new Mat();
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();

        try {
            region = binary.submat(new org.opencv.core.Rect(x, y, w, h));

            int kernelW = clamp((int) Math.round(medianGlyphHeight * KERNEL_WIDTH_FACTOR * kernelFactor), MIN_KERNEL_W, MAX_KERNEL_W);
            int kernelH = clamp((int) Math.round(medianGlyphHeight * KERNEL_HEIGHT_FACTOR), MIN_KERNEL_H, MAX_KERNEL_H);
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kernelW, kernelH));
            Imgproc.dilate(region, dilated, kernel);

            Imgproc.findContours(dilated, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            double minW = medianGlyphHeight * MIN_WORD_WIDTH_FACTOR;
            double minH = medianGlyphHeight * MIN_WORD_HEIGHT_FACTOR;
            double minArea = medianGlyphHeight * medianGlyphHeight * MIN_WORD_AREA_FACTOR;

            for (MatOfPoint c : contours) {
                org.opencv.core.Rect r = Imgproc.boundingRect(c);
                if (r.width < minW || r.height < minH) continue;
                if ((double) r.width * r.height < minArea) continue;
                // Offset back into the full (detect-resolution) coordinate space.
                out.add(new Rect(x + r.x, y + r.y, x + r.x + r.width, y + r.y + r.height));
            }
            
            return out;
        } finally {
            if (region != null) region.release();
            if (kernel != null) kernel.release();
            dilated.release();
            hierarchy.release();
            for (MatOfPoint c : contours) c.release();
        }
    }

    // ── Coordinate mapping ────────────────────────────────────────────────────

    private static List<WordBox> scaleToOriginal(List<WordBox> boxes, double scale) {
        if (scale >= 1.0) return boxes;
        double inv = 1.0 / scale;
        List<WordBox> out = new ArrayList<>();
        for (WordBox w : boxes) {
            Rect r = w.rect;
            Rect mapped = new Rect(
                    (int) Math.round(r.left * inv), (int) Math.round(r.top * inv),
                    (int) Math.round(r.right * inv), (int) Math.round(r.bottom * inv));
            out.add(new WordBox(mapped, w.lineIndex));
        }
        return out;
    }

    private static int clamp(int val, int min, int max) {
        return Math.max(min, Math.min(max, val));
    }
}