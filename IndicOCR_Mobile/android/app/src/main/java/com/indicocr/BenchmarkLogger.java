package com.indicocr;

import android.content.Context;
import android.content.Intent;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import androidx.core.content.FileProvider;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.Font;
import org.apache.poi.ss.usermodel.IndexedColors;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.streaming.SXSSFSheet;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * BenchmarkLogger
 * ===============
 * Records what was recognised, how fast, and (when a reference/ground-truth
 * string is supplied) how accurate it was — then exports it as a real
 * .xlsx file via Apache POI, which this project already depends on
 * (poi-ooxml is in build.gradle for exactly this kind of export). Falls
 * back to a plain CSV export too, in case POI ever misbehaves on a
 * particular device — CSV also opens directly in Excel/Sheets.
 *
 * Typical usage in MainActivity:
 *
 *   BenchmarkLogger bench = new BenchmarkLogger("session");     // once
 *   HybridSegmenter.Result seg = segmenter.segment(pageBitmap);
 *   for (WordSegmenter.WordBox box : seg.boxes) {
 *       Bitmap crop = WordSegmenter.crop(pageBitmap, box.rect);
 *       long t0 = System.nanoTime();
 *       String text = ocrEngine.recognise(crop);
 *       long latencyMs = (System.nanoTime() - t0) / 1_000_000;
 *       bench.record(box, text, null, seg.detectorType, latencyMs);
 *   }
 *   File xlsx = bench.exportXlsx(context);
 *   bench.shareFile(context, xlsx, XLSX_MIME);
 */
public final class BenchmarkLogger {

    private static BenchmarkLogger instance;

    public static synchronized BenchmarkLogger getInstance() {
        if (instance == null) {
            instance = new BenchmarkLogger("session_" + System.currentTimeMillis());
        }
        return instance;
    }

    public static final String XLSX_MIME =
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
    public static final String CSV_MIME = "text/csv";

    /** One logged recognition event. */
    public static final class Entry {
        final int boxIndex;
        final int lineIndex;
        final Rect rect;
        final String detectorType;   // "dbnet" or "classical"
        final String recognisedText;
        final String groundTruthText; // nullable — leave null outside test mode
        final float detectorConfidence;
        final long recognitionLatencyMs;
        final long timestamp;

        Entry(int boxIndex, int lineIndex, Rect rect, String detectorType,
              String recognisedText, String groundTruthText,
              float detectorConfidence, long recognitionLatencyMs) {
            this.boxIndex = boxIndex;
            this.lineIndex = lineIndex;
            this.rect = rect;
            this.detectorType = detectorType;
            this.recognisedText = recognisedText == null ? "" : recognisedText;
            this.groundTruthText = groundTruthText;
            this.detectorConfidence = detectorConfidence;
            this.recognitionLatencyMs = recognitionLatencyMs;
            this.timestamp = System.currentTimeMillis();
        }
    }

    private final String sessionId;
    private final List<Entry> entries = new ArrayList<>();
    private int nextBoxIndex = 0;

    public BenchmarkLogger(String sessionId) {
        this.sessionId = sessionId;
    }

    /** Records one recognised word/line. Pass groundTruth=null outside test mode. */
    public void record(WordSegmenter.WordBox box, String recognisedText, String groundTruthText,
                        String detectorType, long recognitionLatencyMs) {
        entries.add(new Entry(nextBoxIndex++, box.lineIndex, box.rect, detectorType,
                recognisedText, groundTruthText, box.confidence, recognitionLatencyMs));
    }

    public void clear() {
        entries.clear();
        nextBoxIndex = 0;
    }

    public List<Entry> getEntries() {
        return entries;
    }

    public int size() {
        return entries.size();
    }

    // ── XLSX export (primary — matches this project's existing POI setup) ────

    private static final String[] HEADERS = {
            "session_id", "box_index", "line_index", "x", "y", "width", "height",
            "detector_type", "detector_confidence", "recognised_text", "ground_truth_text",
            "char_error_rate", "is_exact_match", "recognition_latency_ms", "timestamp"
    };

    public File exportXlsx(Context context) throws IOException {
        ExcelExporter.ensureTempStrategy(context);
        File dir = outputDir(context);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File out = new File(dir, sanitise(sessionId) + "_" + stamp + ".xlsx");

        try (SXSSFWorkbook wb = new SXSSFWorkbook(100)) {
            Sheet detail = wb.createSheet("Detail");
            SimpleDateFormat isoFmt = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);

            Row header = detail.createRow(0);
            for (int i = 0; i < HEADERS.length; i++) {
                Cell c = header.createCell(i);
                c.setCellValue(HEADERS[i]);
            }

            long totalLatency = 0;
            int totalWithGroundTruth = 0;
            int exactMatches = 0;
            double totalCer = 0;

            int rowIdx = 1;
            for (Entry e : entries) {
                Double cer = null;
                Boolean exact = null;
                if (e.groundTruthText != null) {
                    cer = (double) characterErrorRate(e.recognisedText, e.groundTruthText);
                    exact = e.recognisedText.trim().equals(e.groundTruthText.trim());
                    totalWithGroundTruth++;
                    totalCer += cer;
                    if (exact) exactMatches++;
                }
                totalLatency += e.recognitionLatencyMs;

                Row row = detail.createRow(rowIdx++);
                int col = 0;
                row.createCell(col++).setCellValue(sessionId);
                row.createCell(col++).setCellValue(e.boxIndex);
                row.createCell(col++).setCellValue(e.lineIndex);
                row.createCell(col++).setCellValue(e.rect.left);
                row.createCell(col++).setCellValue(e.rect.top);
                row.createCell(col++).setCellValue(e.rect.width());
                row.createCell(col++).setCellValue(e.rect.height());
                row.createCell(col++).setCellValue(e.detectorType);
                row.createCell(col++).setCellValue(e.detectorConfidence);
                row.createCell(col++).setCellValue(e.recognisedText);
                row.createCell(col++).setCellValue(e.groundTruthText == null ? "" : e.groundTruthText);
                row.createCell(col++).setCellValue(cer == null ? "" : String.format(Locale.US, "%.4f", cer));
                row.createCell(col++).setCellValue(exact == null ? "" : String.valueOf(exact));
                row.createCell(col++).setCellValue(e.recognitionLatencyMs);
                row.createCell(col).setCellValue(isoFmt.format(new Date(e.timestamp)));
            }
            // NOTE: fixed column widths below are used to avoid calling autoSizeColumn(),
            // which triggers SheetUtil and risks a crash on Android (NoClassDefFoundError).
            int[] fixedWidths = {4000, 2400, 2400, 2000, 2000, 2000, 2000,
                    3200, 3600, 9000, 9000, 3600, 3200, 4800, 5600};
            for (int i = 0; i < HEADERS.length && i < fixedWidths.length; i++) {
                detail.setColumnWidth(i, fixedWidths[i]);
            }

            // ── Summary sheet ──
            Sheet summary = wb.createSheet("Summary");
            int r = 0;
            writeKV(summary, r++, "total_words", String.valueOf(entries.size()));
            writeKV(summary, r++, "avg_recognition_latency_ms", entries.isEmpty() ? "0" :
                    String.format(Locale.US, "%.1f", (double) totalLatency / entries.size()));
            if (totalWithGroundTruth > 0) {
                writeKV(summary, r++, "words_with_ground_truth", String.valueOf(totalWithGroundTruth));
                writeKV(summary, r++, "exact_match_accuracy_pct",
                        String.format(Locale.US, "%.2f", 100.0 * exactMatches / totalWithGroundTruth));
                writeKV(summary, r++, "avg_character_error_rate",
                        String.format(Locale.US, "%.4f", totalCer / totalWithGroundTruth));
            } else {
                writeKV(summary, r++, "note",
                        "no ground_truth supplied - accuracy rows left blank; pass expected text into record()");
            }
            summary.setColumnWidth(0, 7000);
            summary.setColumnWidth(1, 9000);

            try (FileOutputStream fos = new FileOutputStream(out)) {
                wb.write(fos);
                fos.flush();
            } finally {
                wb.dispose();
            }
        } catch (Throwable t) {
            Log.e("BenchmarkLogger", "XLSX export failed - likely missing XML dependencies on this device", t);
            throw new IOException("Excel export failed: " + t.getMessage(), t);
        }

        return out;
    }

    private void writeKV(Sheet sheet, int rowIdx, String key, String value) {
        Row row = sheet.createRow(rowIdx);
        row.createCell(0).setCellValue(key);
        row.createCell(1).setCellValue(value);
    }

    // ── CSV export (lightweight fallback) ─────────────────────────────────────

    public File exportCsv(Context context) throws IOException {
        File dir = outputDir(context);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File out = new File(dir, sanitise(sessionId) + "_" + stamp + ".csv");

        try (FileOutputStream fos = new FileOutputStream(out);
             Writer writer = new OutputStreamWriter(fos, StandardCharsets.UTF_8)) {
            writer.write('\uFEFF'); // BOM so Excel renders Indic scripts correctly
            writer.write(String.join(",", HEADERS));
            writer.write("\n");

            SimpleDateFormat isoFmt = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.US);
            for (Entry e : entries) {
                Double cer = e.groundTruthText != null
                        ? characterErrorRate(e.recognisedText, e.groundTruthText) : null;
                Boolean exact = e.groundTruthText != null
                        ? e.recognisedText.trim().equals(e.groundTruthText.trim()) : null;

                writer.write(String.join(",",
                        csv(sessionId), String.valueOf(e.boxIndex), String.valueOf(e.lineIndex),
                        String.valueOf(e.rect.left), String.valueOf(e.rect.top),
                        String.valueOf(e.rect.width()), String.valueOf(e.rect.height()),
                        csv(e.detectorType), String.format(Locale.US, "%.3f", e.detectorConfidence),
                        csv(e.recognisedText), csv(e.groundTruthText == null ? "" : e.groundTruthText),
                        cer == null ? "" : String.format(Locale.US, "%.4f", cer),
                        exact == null ? "" : String.valueOf(exact),
                        String.valueOf(e.recognitionLatencyMs), isoFmt.format(new Date(e.timestamp))));
                writer.write("\n");
            }
        }
        return out;
    }

    // ── Sharing ──────────────────────────────────────────────────────────────

    /** Opens the system share sheet, reusing the FileProvider your Share
     *  feature already declares (authority "${applicationId}.fileprovider"). */
    public void shareFile(Context context, File file, String mimeType) {
        Uri uri = FileProvider.getUriForFile(context,
                context.getPackageName() + ".fileprovider", file);
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType(mimeType);
        intent.putExtra(Intent.EXTRA_STREAM, uri);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        context.startActivity(Intent.createChooser(intent, "Export OCR benchmark"));
    }

    // ── Accuracy math ────────────────────────────────────────────────────────

    /** Character Error Rate = Levenshtein edit distance / reference length. */
    public static double characterErrorRate(String hypothesis, String reference) {
        if (reference == null || reference.isEmpty()) return hypothesis.isEmpty() ? 0.0 : 1.0;
        int distance = levenshtein(hypothesis, reference);
        return (double) distance / reference.length();
    }

    private static int levenshtein(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) dp[i][0] = i;
        for (int j = 0; j <= b.length(); j++) dp[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + cost);
            }
        }
        return dp[a.length()][b.length()];
    }

    // ── Plumbing ─────────────────────────────────────────────────────────────

    private File outputDir(Context context) throws IOException {
        File dir = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "ocr_benchmarks");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Could not create benchmark output directory: " + dir);
        }
        return dir;
    }

    private static String csv(String field) {
        String f = field == null ? "" : field;
        boolean needsQuoting = f.contains(",") || f.contains("\"") || f.contains("\n");
        if (needsQuoting) f = "\"" + f.replace("\"", "\"\"") + "\"";
        return f;
    }

    private static String sanitise(String s) {
        return s.replaceAll("[^a-zA-Z0-9_-]", "_");
    }
}