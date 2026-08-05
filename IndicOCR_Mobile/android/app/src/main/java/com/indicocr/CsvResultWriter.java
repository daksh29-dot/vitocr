package com.indicocr;

import android.content.Context;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * CsvResultWriter
 * ================
 * Writes one CSV row PER IMAGE, immediately, as inference completes.
 * No batching, no in-memory accumulation of 30k rows, no POI temp-file
 * machinery, no single giant write-at-the-end step.
 *
 * Why this fixes the "stuck at 0/997" problem:
 *   - Old flow: infer all 997 -> THEN build+write entire .xlsx in one shot.
 *     Nothing touches disk (or the progress bar) until that final write
 *     starts, and if IT hangs (permissions / temp dir / MediaStore),
 *     you see 0/997 forever with zero visibility into why.
 *   - New flow: every single image writes its own line to disk the moment
 *     it's done. The progress counter IS the line count. If it stalls,
 *     you can literally open the CSV mid-run and see exactly which image
 *     was last written.
 *
 * Location: app-specific external storage (getExternalFilesDir) —
 * this requires NO storage permission on ANY Android version, unlike
 * the public Downloads folder which needs scoped-storage handling on
 * API 29+ and is missing WRITE_EXTERNAL_STORAGE above API 29 in your
 * current manifest.
 */
public class CsvResultWriter implements AutoCloseable {

    private final BufferedWriter writer;
    private final File file;
    private int rowsWritten = 0;

    public CsvResultWriter(Context ctx, boolean hasGroundTruth) throws IOException {
        String ts = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String name = (hasGroundTruth ? "eval_results_" : "predictions_") + ts + ".csv";

        // App-specific external storage: no runtime permission needed, ever.
        File outDir = ctx.getExternalFilesDir(null);
        if (outDir == null) outDir = ctx.getFilesDir(); // internal fallback if ext unavailable
        outDir.mkdirs();
        file = new File(outDir, name);

        writer = new BufferedWriter(new FileWriter(file), 8192);

        // Header
        if (hasGroundTruth) {
            writer.write("image_name,image_path,ground_truth,prediction,cer,wer,crr,wrr,inference_ms\n");
        } else {
            writer.write("image_name,image_path,prediction,inference_ms\n");
        }
        writer.flush(); // ensure header hits disk immediately — file is never "empty and unreadable"
    }

    public File getFile() {
        return file;
    }

    public int getRowsWritten() {
        return rowsWritten;
    }

    /** Call this immediately after each image finishes inference. */
    public void writeRow(EvalEngine.ImageResult res, boolean hasGroundTruth) throws IOException {
        String imageName = new File(res.imagePath).getName();
        StringBuilder sb = new StringBuilder();
        sb.append(csvEscape(imageName)).append(',');
        sb.append(csvEscape(res.imagePath)).append(',');
        if (hasGroundTruth) {
            sb.append(csvEscape(res.groundTruth != null ? res.groundTruth : "")).append(',');
            sb.append(csvEscape(res.prediction)).append(',');
            float cer = res.cer >= 0 ? res.cer : 0;
            float wer = res.wer >= 0 ? res.wer : 0;
            sb.append(cer).append(',');
            sb.append(wer).append(',');
            sb.append(1 - cer).append(','); // CRR = Character Recognition Rate
            sb.append(1 - wer).append(','); // WRR = Word Recognition Rate
            sb.append(res.inferenceMs).append('\n');
        } else {
            sb.append(csvEscape(res.prediction)).append(',');
            sb.append(res.inferenceMs).append('\n');
        }
        writer.write(sb.toString());
        writer.flush(); // flush EVERY row — at 30k rows of ~150 bytes this costs nothing,
                         // and guarantees a crash never loses more than the current row.
        rowsWritten++;
    }

    /** Standard CSV escaping: quote fields containing comma, quote, or newline. */
    private String csvEscape(String s) {
        if (s == null) return "";
        boolean needsQuote = s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");
        if (!needsQuote) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    @Override
    public void close() throws IOException {
        writer.close();
    }
}