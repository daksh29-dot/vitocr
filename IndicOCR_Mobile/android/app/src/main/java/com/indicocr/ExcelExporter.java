package com.indicocr;

import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;
import org.apache.poi.util.DefaultTempFileCreationStrategy;
import org.apache.poi.util.TempFile;

/**
 * ExcelExporter
 * =============
 * Writes evaluation results to an .xlsx file on device storage.
 * Optimized for Android with MediaStore and POI Streaming API.
 */
public class ExcelExporter {

    private static volatile boolean tempStrategySet = false;

    /**
     * Proper way to set up POI temp directory on Android.
     * System.setProperty("java.io.tmpdir") is often ignored or restricted.
     */
    public static synchronized void ensureTempStrategy(Context ctx) {
        if (tempStrategySet) return;
        File tempDir = new File(ctx.getCacheDir(), "poi-temp");
        if (!tempDir.exists()) tempDir.mkdirs();
        TempFile.setTempFileCreationStrategy(new DefaultTempFileCreationStrategy(tempDir));
        tempStrategySet = true;
    }

    public static File export(Context ctx,
                              EvalEngine.EvalSummary summary,
                              boolean hasGroundTruth,
                              EvalEngine.ProgressCallback cb) throws IOException {
        
        ensureTempStrategy(ctx);
        
        String ts   = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String name = hasGroundTruth
                      ? "eval_results_" + ts + ".xlsx"
                      : "predictions_" + ts + ".xlsx";

        // For modern Android (API 29+), MediaStore is preferred, but to keep the 
        // existing return type (File) and logic simple for the user's current 
        // setup, we'll continue using the public Downloads directory but with 
        // the fixed POI strategy.
        
        File outDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        if (!outDir.exists()) outDir.mkdirs();
        File outFile = new File(outDir, name);

        // Increase streaming window size for better write speed on some Android flash storages
        try (SXSSFWorkbook wb = buildWorkbook(summary, hasGroundTruth, cb)) {
            // OPTIMIZATION: Ensure the temp files are written to the right place and flushed
            try (FileOutputStream fos = new FileOutputStream(outFile)) {
                wb.write(fos);
                fos.flush();
            } finally {
                wb.dispose();
            }
        }

        return outFile;
    }

    /**
     * Builds a styled .xlsx from an already-complete CSV file (produced by
     * CsvResultWriter). This is the SAFE way to get an Excel file at scale:
     * by the time this runs, every row is already durably on disk as CSV,
     * so if this step hangs or throws, nothing is lost — it's purely a
     * cosmetic conversion, not the primary data path.
     *
     * Uses MediaStore.Downloads on API 29+ instead of a raw File write to
     * the public Downloads directory, since WRITE_EXTERNAL_STORAGE is
     * capped at maxSdkVersion=29 in the manifest and raw File writes to
     * public storage are blocked/unreliable above that without it.
     */
    public static File exportFromCsv(Context ctx, File csvFile, boolean hasGroundTruth) throws IOException {
        ensureTempStrategy(ctx);

        String ts = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String name = (hasGroundTruth ? "eval_results_" : "predictions_") + ts + ".xlsx";

        try (SXSSFWorkbook wb = new SXSSFWorkbook(100)) {
            Sheet sheet = wb.createSheet("Results");
            sheet.setColumnWidth(0, 12000); // image_name
            sheet.setColumnWidth(1, 15000); // image_path
            if (hasGroundTruth) {
                sheet.setColumnWidth(2, 10000); // ground_truth
                sheet.setColumnWidth(3, 10000); // prediction
                sheet.setColumnWidth(4, 3000);  // cer
                sheet.setColumnWidth(5, 3000);  // wer
                sheet.setColumnWidth(6, 5000);  // inference_ms
            } else {
                sheet.setColumnWidth(2, 10000); // prediction
                sheet.setColumnWidth(3, 5000);  // inference_ms
            }

            CellStyle headerStyle = wb.createCellStyle();
            Font hFont = wb.createFont();
            hFont.setBold(true);
            headerStyle.setFont(hFont);
            headerStyle.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
            headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

            try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(csvFile))) {
                String headerLine = reader.readLine(); // CSV header
                if (headerLine == null) throw new IOException("CSV is empty: " + csvFile);
                String[] headerCols = headerLine.split(",");

                Row header = sheet.createRow(0);
                for (int c = 0; c < headerCols.length; c++) {
                    Cell cell = header.createCell(c);
                    cell.setCellValue(headerCols[c]);
                    cell.setCellStyle(headerStyle);
                }

                String line;
                int rowNum = 1;
                while ((line = reader.readLine()) != null) {
                    String[] fields = parseCsvLine(line);
                    Row row = sheet.createRow(rowNum++);
                    for (int c = 0; c < fields.length; c++) {
                        row.createCell(c).setCellValue(fields[c]);
                    }
                }
            }

            // Write to app-external storage — same folder as the CSV, no permission
            // needed on any Android version, and guarantees a real file on disk
            // that FileProvider can share (unlike a MediaStore-only write, which
            // doesn't leave a local path we can hand to the share sheet).
            File outDir = ctx.getExternalFilesDir(null);
            if (outDir == null) outDir = ctx.getFilesDir();
            outDir.mkdirs();
            File outFile = new File(outDir, name);
            try (FileOutputStream fos = new FileOutputStream(outFile)) {
                wb.write(fos);
                fos.flush();
            } finally {
                wb.dispose();
            }
            return outFile;
        }
    }

    /** Minimal CSV line parser handling quoted fields (matches CsvResultWriter's escaping). */
    private static String[] parseCsvLine(String line) {
        List<String> fields = new java.util.ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (inQuotes) {
                if (ch == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur.append(ch);
                }
            } else {
                if (ch == '"') {
                    inQuotes = true;
                } else if (ch == ',') {
                    fields.add(cur.toString());
                    cur.setLength(0);
                } else {
                    cur.append(ch);
                }
            }
        }
        fields.add(cur.toString());
        return fields.toArray(new String[0]);
    }

    /**
     * Refactored workbook building logic for clarity and reuse.
     */
    private static SXSSFWorkbook buildWorkbook(EvalEngine.EvalSummary summary, 
                                               boolean hasGroundTruth, 
                                               EvalEngine.ProgressCallback cb) throws IOException {
        
        SXSSFWorkbook wb = new SXSSFWorkbook(100);
        Sheet sheet = wb.createSheet("Results");
        
        // Set fixed column widths
        sheet.setColumnWidth(0, 15000); 
        sheet.setColumnWidth(1, 10000); 
        sheet.setColumnWidth(2, 10000); 
        if (hasGroundTruth) {
            sheet.setColumnWidth(3, 3000); 
            sheet.setColumnWidth(4, 3000); 
            sheet.setColumnWidth(5, 5000); 
        }

        // Header style
        CellStyle headerStyle = wb.createCellStyle();
        Font hFont = wb.createFont();
        hFont.setBold(true);
        headerStyle.setFont(hFont);
        headerStyle.setFillForegroundColor(IndexedColors.GREY_25_PERCENT.getIndex());
        headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

        // Header row
        Row header = sheet.createRow(0);
        String[] cols = hasGroundTruth
            ? new String[]{"Image Path","Ground Truth","Prediction","CER","WER","Inference (ms)"}
            : new String[]{"Image Path","Prediction","Inference (ms)"};

        for (int c = 0; c < cols.length; c++) {
            Cell cell = header.createCell(c);
            cell.setCellValue(cols[c]);
            cell.setCellStyle(headerStyle);
        }

        // Data rows
        List<EvalEngine.ImageResult> results = summary.results;
        int totalResults = results.size();
        
        if (cb != null) cb.onProgress(0, totalResults);

        for (int r = 0; r < totalResults; r++) {
            if (cb != null && cb.isCancelled()) {
                throw new IOException("Export cancelled by user");
            }
            EvalEngine.ImageResult res = results.get(r);
            
            // THROTTLED UPDATES: Only update UI every 50 rows to avoid flooding the main thread
            // This prevents the app from "freezing" during large exports (29k+ images)
            if (cb != null && (r + 1) % 50 == 0) {
                String fileName = new File(res.imagePath).getName();
                cb.onProgressWithDetails(r + 1, totalResults, "Writing: " + fileName);
            }

            Row row = sheet.createRow(r + 1);
            if (hasGroundTruth) {
                row.createCell(0).setCellValue(res.imagePath);
                row.createCell(1).setCellValue(res.groundTruth != null ? res.groundTruth : "");
                row.createCell(2).setCellValue(res.prediction);
                row.createCell(3).setCellValue(res.cer >= 0 ? res.cer : 0);
                row.createCell(4).setCellValue(res.wer >= 0 ? res.wer : 0);
                row.createCell(5).setCellValue(res.inferenceMs);
            } else {
                row.createCell(0).setCellValue(res.imagePath);
                row.createCell(1).setCellValue(res.prediction);
                row.createCell(2).setCellValue(res.inferenceMs);
            }
            
            // INCREASED FREQUENCY: Update every 10 rows for 1000-item sets
            if (cb != null && (r + 1) % 10 == 0) {
                cb.onProgress(r + 1, totalResults);
            }
        }
        
        // Ensure final progress is called before summary sheet
        if (cb != null) cb.onProgress(totalResults, totalResults);

        // Summary sheet
        if (hasGroundTruth) {
            Sheet sumSheet = wb.createSheet("Summary");
            String[][] sumRows = {
                {"Total images",    String.valueOf(summary.totalImages)},
                {"Skipped",         String.valueOf(summary.skipped)},
                {"Avg CER",         String.format(Locale.US, "%.4f", summary.avgCER)},
                {"Avg WER",         String.format(Locale.US, "%.4f", summary.avgWER)},
                {"Avg CRR",         String.format(Locale.US, "%.4f", summary.avgCRR)},
                {"Avg WRR",         String.format(Locale.US, "%.4f", summary.avgWRR)},
                {"Total time (ms)", String.valueOf(summary.totalMs)},
                {"Avg time/img ms", String.format(Locale.US, "%.1f",
                                     summary.totalImages > 0
                                     ? (double) summary.totalMs / summary.totalImages
                                     : 0)},
            };
            for (int i = 0; i < sumRows.length; i++) {
                Row row = sumSheet.createRow(i);
                row.createCell(0).setCellValue(sumRows[i][0]);
                row.createCell(1).setCellValue(sumRows[i][1]);
            }
        }

        if (cb != null) cb.onProgress(totalResults, totalResults);
        return wb;
    }
}