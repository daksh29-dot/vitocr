package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * OcrTestRunner
 * =============
 * Runs the full pipeline (HybridSegmenter -> OCR) over a fixed set of test
 * images with known-correct transcriptions, and produces ONE combined
 * BenchmarkLogger CSV covering the whole run — this is what you compare
 * across model changes (classical vs DBNet, INT8 vs float, etc.) to see if
 * a change actually improved accuracy rather than eyeballing a few pages.
 *
 * Expected test-set layout (put this under app/src/main/assets/testset/,
 * or point atDir at any external folder you sync test images into):
 *
 *   testset/
 *     manifest.csv          <- image_filename,expected_text
 *     page_001.jpg
 *     page_002.jpg
 *     ...
 *
 * manifest.csv format (one line per full-page test case):
 *   page_001.jpg,"Full expected transcription of the page, in order"
 *
 * NOTE: ground truth here is matched to the whole recognised page text
 * (words joined in reading order), not per-box, since manually labelling
 * per-box ground truth is rarely worth it. Per-box CER still gets computed
 * once you have per-box references; leave groundTruth null per-record and
 * this runner will instead log ONE whole-page comparison row via
 * recordPageLevel() at the end of each image.
 */
public final class OcrTestRunner {

    public interface Ocr {
        String recognise(Bitmap wordCrop);
    }

    public static final class TestCase {
        public final String imageFileName;
        public final String expectedText;
        public TestCase(String imageFileName, String expectedText) {
            this.imageFileName = imageFileName;
            this.expectedText = expectedText;
        }
    }

    /** Parses manifest.csv into TestCases. Very small/forgiving CSV parser —
     *  expects at most one comma-containing quoted field (expected_text). */
    public static List<TestCase> loadManifest(File manifestCsv) throws IOException {
        List<TestCase> cases = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(manifestCsv), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                int firstComma = line.indexOf(',');
                if (firstComma < 0) continue;
                String fileName = line.substring(0, firstComma).trim();
                String expected = line.substring(firstComma + 1).trim();
                if (expected.startsWith("\"") && expected.endsWith("\"")) {
                    expected = expected.substring(1, expected.length() - 1).replace("\"\"", "\"");
                }
                cases.add(new TestCase(fileName, expected));
            }
        }
        return cases;
    }

    /**
     * Runs every test case through segmentation + OCR, logging per-word
     * entries (no ground truth, since we only have page-level truth) plus
     * one page-level accuracy comparison per image. Returns the exported CSV.
     */
    public static File run(Context context, File imagesDir, List<TestCase> testCases,
                            HybridSegmenter segmenter, Ocr ocr) throws IOException {
        BenchmarkLogger bench = new BenchmarkLogger("full_testset_run");

        for (TestCase tc : testCases) {
            File imgFile = new File(imagesDir, tc.imageFileName);
            Bitmap page = BitmapFactory.decodeFile(imgFile.getAbsolutePath());
            if (page == null) continue;

            HybridSegmenter.Result seg = segmenter.segment(page);
            StringBuilder pageText = new StringBuilder();
            int lastLine = -1;

            for (WordSegmenter.WordBox box : seg.boxes) {
                Bitmap crop = WordSegmenter.crop(page, box.rect);
                if (crop == null) continue;

                long t0 = System.nanoTime();
                String text = ocr.recognise(crop);
                long latencyMs = (System.nanoTime() - t0) / 1_000_000;

                bench.record(box, text, null, seg.detectorType, latencyMs);

                if (box.lineIndex != lastLine && pageText.length() > 0) pageText.append("\n");
                else if (pageText.length() > 0) pageText.append(" ");
                pageText.append(text);
                lastLine = box.lineIndex;
            }

            // Page-level ground-truth comparison, logged as a synthetic
            // whole-page "box" (rect covers the full page) so it shows up
            // in the same CSV with a real CER against tc.expectedText.
            WordSegmenter.WordBox pageBox = new WordSegmenter.WordBox(
                    new android.graphics.Rect(0, 0, page.getWidth(), page.getHeight()), -1, 1.0f);
            bench.record(pageBox, pageText.toString(), tc.expectedText, seg.detectorType + "_page", 0);
        }

        return bench.exportCsv(context);
    }
}
