package com.indicocr;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.DocumentsContract;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import ai.onnxruntime.OrtException;

/**
 * EvalActivity
 * ============
 * UI for running the image evaluation.
 *
 * Two modes selectable via radio buttons:
 *   1. Folder mode  — pick image folder → predictions only → Excel
 *   2. TSV mode     — pick TSV file     → predictions + CER/WER → Excel
 */
public class EvalActivity extends AppCompatActivity {

    // ── UI ────────────────────────────────────────────────────────────────────
    private EditText    folderInput;
    private Button      runBtn, terminateBtn, shareBtn;
    private ImageButton browseFolderBtn, btnBack;
    private ProgressBar progressBar;
    private TextView    progressText, logText, summaryText;
    private ScrollView  logScroll;

    private File lastCsvFile = null;
    private File lastXlsxFile = null;
    private Uri  lastSelectedFolderUri = null;

    // ── State ─────────────────────────────────────────────────────────────────
    private volatile EvalEngine engine;
    private ExecutorService  executor;
    private Future<?>        runningTask;
    private volatile boolean cancelled = false;
    private final Handler    uiHandler = new Handler(Looper.getMainLooper());

    private String formatTime(long ms) {
        if (ms <= 0) return "calculating...";
        long seconds = ms / 1000;
        long minutes = seconds / 60;
        seconds %= 60;
        if (minutes > 0) return minutes + "m " + seconds + "s";
        return seconds + "s";
    }

    // ── Permission & Folder launchers ──────────────────────────────────────────

    private final ActivityResultLauncher<Uri> folderPicker =
        registerForActivityResult(new ActivityResultContracts.OpenDocumentTree(), uri -> {
            if (uri != null) {
                lastSelectedFolderUri = uri;
                // Take persistable permission so we can access it after app restart or in background
                try {
                    getContentResolver().takePersistableUriPermission(uri,
                            Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                } catch (Exception e) {
                    android.util.Log.e("EvalActivity", "Failed to take persistable URI permission", e);
                }
                // Convert content URI to real path for display
                String path = getRealPath(uri);
                folderInput.setText(path != null ? path : uri.toString());
            }
        });

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_eval);
        executor = Executors.newSingleThreadExecutor();

        bindViews();
        setupListeners();
        requestStoragePermission();
        initEngine();
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateRunButtonState();
    }

    private void updateRunButtonState() {
        boolean ready = (engine != null);
        runBtn.setEnabled(ready);
        log("Checking button state: engine=" + (engine != null) + " -> enabled=" + ready);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (engine != null) engine.close();
        executor.shutdownNow();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Setup
    // ─────────────────────────────────────────────────────────────────────────

    private void bindViews() {
        folderInput  = findViewById(R.id.folderInput);
        browseFolderBtn = (ImageButton) findViewById(R.id.browseFolderBtn);
        runBtn       = findViewById(R.id.runBtn);
        terminateBtn = findViewById(R.id.cancelBtn);
        btnBack      = (ImageButton) findViewById(R.id.btnBack);
        progressBar  = findViewById(R.id.progressBar);
        progressText = findViewById(R.id.progressText);
        logText      = findViewById(R.id.logText);
        summaryText  = findViewById(R.id.summaryText);
        logScroll    = findViewById(R.id.logScroll);
        shareBtn     = findViewById(R.id.shareBtn);
    }

    private void setupListeners() {
        btnBack.setOnClickListener(v -> finish());
        browseFolderBtn.setOnClickListener(v -> folderPicker.launch(null));

        runBtn.setOnClickListener(v -> startEval());
        terminateBtn.setOnClickListener(v -> {
            cancelled = true;
            log("🛑 Terminating execution... Please wait.");
            terminateBtn.setEnabled(false);
            if (runningTask != null) {
                runningTask.cancel(true);
            }
        });
        shareBtn.setOnClickListener(v -> shareResults());
    }

    /**
     * Opens the system share sheet (WhatsApp, Gmail, Drive, Bluetooth,
     * "Save to Files", etc.) for the most recent result files. This is the
     * reliable way to get files off the device without needing to browse
     * into the restricted Android/data folder manually.
     */
    private void shareResults() {
        java.util.ArrayList<Uri> uris = new java.util.ArrayList<>();

        if (lastCsvFile != null && lastCsvFile.exists()) {
            uris.add(FileProvider.getUriForFile(this,
                    "com.indicocr.fileprovider", lastCsvFile));
        }
        if (lastXlsxFile != null && lastXlsxFile.exists()) {
            uris.add(FileProvider.getUriForFile(this,
                    "com.indicocr.fileprovider", lastXlsxFile));
        }

        if (uris.isEmpty()) {
            Toast.makeText(this, "No result file yet — run an evaluation first", Toast.LENGTH_SHORT).show();
            return;
        }

        Intent shareIntent;
        if (uris.size() == 1) {
            shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.putExtra(Intent.EXTRA_STREAM, uris.get(0));
            shareIntent.setType(uris.get(0).toString().endsWith(".csv") ? "text/csv"
                    : "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
        } else {
            shareIntent = new Intent(Intent.ACTION_SEND_MULTIPLE);
            shareIntent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris);
            shareIntent.setType("*/*");
        }
        shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(shareIntent, "Share results"));
    }

    private void initEngine() {
        log("Initialising engine... (Current thread: " + Thread.currentThread().getName() + ")");
        executor.submit(() -> {
            try {
                log("Background: Loading ONNX model and label map...");
                engine = new EvalEngine(getApplicationContext());
                log("Background: Engine instance created.");
                uiHandler.post(() -> {
                    log("UI Thread: Engine marked as ready.");
                    updateRunButtonState();
                    log("Engine loaded successfully. Ready for evaluation.");
                });
            } catch (Exception e) {
                android.util.Log.e("EvalActivity", "Engine init failed", e);
                // Print the full error and stack trace to the log window
                java.io.StringWriter sw = new java.io.StringWriter();
                java.io.PrintWriter pw = new java.io.PrintWriter(sw);
                e.printStackTrace(pw);
                String stackTrace = sw.toString();
                
                uiHandler.post(() -> {
                    log("ENGINE INIT FAILED!");
                    log("Error: " + e.getMessage());
                    log("Details:\n" + stackTrace);
                    Toast.makeText(this, "Failed to load engine", Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Evaluation
    // ─────────────────────────────────────────────────────────────────────────

    private void startEval() {
        String folderPath = folderInput.getText().toString().trim();
        if (folderPath.isEmpty()) {
            Toast.makeText(this, "Select a folder first", Toast.LENGTH_SHORT).show();
            return;
        }

        if (engine == null) {
            Toast.makeText(this, "Engine not ready yet", Toast.LENGTH_SHORT).show();
            return;
        }

        cancelled = false;
        setRunning(true);
        logText.setText("");
        summaryText.setText("");
        progressBar.setProgress(0);
        progressText.setText("0 / ?");

        log("Starting folder inference...");
        
        final String finalFolder;
        if (lastSelectedFolderUri != null && folderPath.equals(getRealPath(lastSelectedFolderUri))) {
            finalFolder = lastSelectedFolderUri.toString();
        } else if (folderPath.startsWith("content://")) {
            finalFolder = folderPath;
        } else {
            finalFolder = cleanPath(folderPath);
        }

        final long startTime = System.currentTimeMillis();
        EvalEngine.ProgressCallback cb = new EvalEngine.ProgressCallback() {
            @Override public void onProgress(int done, int total) {
                // Not used anymore, replaced by details
            }

            @Override
            public void onProgressWithDetails(int done, int total, String currentItem) {
                uiHandler.post(() -> {
                    progressBar.setMax(total);
                    progressBar.setProgress(done);

                    long elapsed = System.currentTimeMillis() - startTime;
                    String eta = "";
                    if (done > 5) {
                        long remainingMs = elapsed * (total - done) / done;
                        eta = " (ETA: " + formatTime(remainingMs) + ")";
                    }

                    progressText.setText(done + " / " + total + eta + "\n" + currentItem);
                    
                    if (done % 50 == 0 || done == total)
                        log("Processed " + done + " / " + total + " images");
                });
            }

            @Override public boolean isCancelled() { return cancelled; }
        };

        runningTask = executor.submit(() -> {
            CsvResultWriter csvWriter = null;
            try {
                csvWriter = new CsvResultWriter(getApplicationContext(), false);
                log("Live results: " + csvWriter.getFile().getAbsolutePath());
                final CsvResultWriter finalWriter = csvWriter;
                lastCsvFile = finalWriter.getFile();
                uiHandler.post(() -> shareBtn.setEnabled(true));

                EvalEngine.EvalSummary summary = engine.runFolder(finalFolder, false, cb, finalWriter);

                if (cancelled) {
                    uiHandler.post(() -> {
                        log("⚠️ Process terminated by user. "
                                + finalWriter.getRowsWritten() + " rows were already saved to: "
                                + finalWriter.getFile().getAbsolutePath());
                        setRunning(false);
                    });
                    return;
                }

                final String csvPath = finalWriter.getFile().getAbsolutePath();
                uiHandler.post(() -> {
                    showSummary(summary, false, csvPath);
                    log("Done. " + finalWriter.getRowsWritten() + " rows written to CSV.");
                    setRunning(false);
                });

                try {
                    log("Converting to Excel (optional)...");
                    File xlsx = ExcelExporter.exportFromCsv(getApplicationContext(), finalWriter.getFile(), false);
                    lastXlsxFile = xlsx;
                    uiHandler.post(() -> log("Excel copy ready — tap Share Results to send it: " + xlsx.getName()));
                } catch (Exception e) {
                    android.util.Log.e("EvalActivity", "Optional Excel conversion failed", e);
                    uiHandler.post(() -> log("⚠️ Excel conversion skipped (CSV is complete and safe): " + e.getMessage()));
                }

            } catch (Exception e) {
                android.util.Log.e("EvalActivity", "Process failed", e);
                java.io.StringWriter sw = new java.io.StringWriter();
                java.io.PrintWriter pw = new java.io.PrintWriter(sw);
                e.printStackTrace(pw);
                String stackTrace = sw.toString();

                final int rowsWritten = csvWriter != null ? csvWriter.getRowsWritten() : 0;
                uiHandler.post(() -> {
                    log("❌ ERROR: " + e.getMessage());
                    log(rowsWritten + " rows were already saved to disk before the error.");
                    log("Details:\n" + stackTrace);
                    setRunning(false);
                });
            } finally {
                if (csvWriter != null) {
                    try { csvWriter.close(); } catch (java.io.IOException ignored) {}
                }
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UI helpers
    // ─────────────────────────────────────────────────────────────────────────

    private void showSummary(EvalEngine.EvalSummary s, boolean hasGT, String xlsxPath) {
        StringBuilder sb = new StringBuilder();
        sb.append("═══════ Results ═══════\n");
        sb.append("Total images : ").append(s.totalImages).append("\n");
        sb.append("Skipped      : ").append(s.skipped).append("\n");
        sb.append("Total time   : ").append(s.totalMs / 1000).append(" s\n");
        sb.append("Avg / image  : ")
          .append(s.totalImages > 0 ? s.totalMs / s.totalImages : 0)
          .append(" ms\n");
        if (hasGT) {
            sb.append("─────────────────────\n");
            sb.append(String.format("CER  : %.4f\n", s.avgCER));
            sb.append(String.format("WER  : %.4f\n", s.avgWER));
            sb.append(String.format("CRR  : %.4f\n", s.avgCRR));
            sb.append(String.format("WRR  : %.4f\n", s.avgWRR));
        }
        sb.append("─────────────────────\n");
        sb.append("File Name    : ").append(new File(xlsxPath).getName()).append("\n");
        sb.append("Storage Path : ").append(xlsxPath).append("\n");
        summaryText.setText(sb.toString());
        log("Done. Excel saved to: " + xlsxPath);
    }

    private void setRunning(boolean running) {
        if (!running) {
            updateRunButtonState();
        } else {
            runBtn.setEnabled(false);
        }
        terminateBtn.setEnabled(running);
        terminateBtn.setVisibility(running ? View.VISIBLE : View.GONE);
        
        browseFolderBtn.setEnabled(!running);
        folderInput.setEnabled(!running);
    }

    private void log(String msg) {
        uiHandler.post(() -> {
            logText.append(msg + "\n");
            logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    private String cleanPath(String path) {
        if (path == null) return null;
        if (path.startsWith("Istorage")) path = "/" + path.substring(1);
        if (path.startsWith("1storage")) path = "/" + path.substring(1);
        path = path.replace("primary:", "/storage/emulated/0/");
        return path.replace("//", "/").trim();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Permissions + path utils
    // ─────────────────────────────────────────────────────────────────────────

    private void requestStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 1);
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                }, 1);
            }
        }
    }

    private String getRealPath(Uri uri) {
        if (uri == null) return null;
        log("🔍 Resolving URI: " + uri.toString());
        String path = null;

        try {
            if (DocumentsContract.isTreeUri(uri)) {
                String treeId = DocumentsContract.getTreeDocumentId(uri);
                if (treeId != null && treeId.contains(":")) {
                    String[] split = treeId.split(":");
                    if ("primary".equalsIgnoreCase(split[0])) {
                        path = Environment.getExternalStorageDirectory() + "/" + (split.length > 1 ? split[1] : "");
                    } else {
                        path = "/storage/" + split[0] + "/" + (split.length > 1 ? split[1] : "");
                    }
                }
            } else if (DocumentsContract.isDocumentUri(this, uri)) {
                String docId = DocumentsContract.getDocumentId(uri);
                if (docId != null && docId.contains(":")) {
                    String[] split = docId.split(":");
                    if ("primary".equalsIgnoreCase(split[0])) {
                        path = Environment.getExternalStorageDirectory() + "/" + (split.length > 1 ? split[1] : "");
                    } else {
                        path = "/storage/" + split[0] + "/" + (split.length > 1 ? split[1] : "");
                    }
                }
            }

            if (path == null) {
                path = uri.getPath();
                if (path != null) {
                    if (path.contains("/tree/primary:")) {
                        path = Environment.getExternalStorageDirectory() + "/" + path.split("/tree/primary:")[1];
                    } else if (path.contains("/document/primary:")) {
                        path = Environment.getExternalStorageDirectory() + "/" + path.split("/document/primary:")[1];
                    } else if (path.contains("/tree/")) {
                        // Handle secondary SD card tree URIs
                        String[] parts = path.split("/tree/")[1].split(":");
                        if (parts.length > 1) {
                            path = "/storage/" + parts[0] + "/" + parts[1];
                        }
                    }
                }
            }

            if (path != null) {
                File f = new File(path);
                if (f.exists()) {
                    log("✅ Resolved to: " + path);
                    return path;
                }
            }
        } catch (Exception e) {
            log("❌ Path error: " + e.getMessage());
        }

        log("⚠️ Path resolution failed. Dataset might show 0/0.");
        return uri.toString();
    }
}