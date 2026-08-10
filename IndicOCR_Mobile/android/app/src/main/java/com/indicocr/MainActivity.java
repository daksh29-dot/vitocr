package com.indicocr;

import android.Manifest;
import android.app.AlertDialog;
import android.content.pm.PackageManager;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatDelegate;

import com.indicocr.db.AppDatabase;
import com.indicocr.db.Document;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.common.util.concurrent.ListenableFuture;
import com.yalantis.ucrop.UCrop;
import com.yalantis.ucrop.UCropActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtException;

/**
 * MainActivity
 * ============
 * Demo activity showing two ways to use IndicOCREngine:
 * 1. Pick an image from the gallery
 * 2. Capture a photo with the camera (CameraX)
 * 3. Launch batch evaluation (40k images)
 *
 * The recognised Indic text appears in a TextView below.
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "IndicOCR";
    private static final String FILE_PROVIDER_AUTHORITY = "com.indicocr.fileprovider";

    // ── UI ────────────────────────────────────────────────────────────────────
    private PreviewView    cameraPreview;
    private View           scanGrid;
    private ImageView      imagePreview;
    private TextView       tvResult;
    private ImageButton    btnCapture;
    private ImageButton    btnGallery;
    private ImageButton    btnEval; // Added for Batch Evaluation
    private ImageButton    btnSearch; // Global search across saved scans
    private ImageButton    btnOnlineToggle;
    private ImageButton    btnSettings; // Doubles as the light/dark theme toggle for now
    private TextView       btnModeWord;
    private TextView       btnModePage;

    private View           cameraCard;
    private View           previewCard;

    /** false = single word, fast path (no segmentation). true = full page, runs WordSegmenter. */
    private boolean pageMode = false;

    // ── OCR ───────────────────────────────────────────────────────────────────
    private IndicOCREngine ocrEngine;
    private TextDetector textDetector; // TextBPN ONNX segmenter used by Page mode.
    private OnlineOcrClient onlineOcrClient;

    // ── Camera ────────────────────────────────────────────────────────────────
    private ImageCapture   imageCapture;
    private ExecutorService cameraExecutor;

    // ── Storage / folders / Recognition ──────────────────────────────────────
    private ExecutorService dbExecutor;
    private ExecutorService recognitionExecutor;
    private static final String[] FOLDERS = {"Receipts", "Notes", "Assignments", "IDs", "Others"};

    // ── Permission launcher ───────────────────────────────────────────────────
    private final ActivityResultLauncher<String> permissionLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
            if (granted) startCamera();
            else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
        });

    // ── Image Cropper (uCrop for Variable Angle) ──────────────────────────────
    private final ActivityResultLauncher<Intent> cropImage =
        registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Uri resultUri = UCrop.getOutput(result.getData());
                if (resultUri != null) {
                    validateAndRecognise(resultUri);
                }
            } else if (result.getResultCode() == UCrop.RESULT_ERROR && result.getData() != null) {
                Throwable cropError = UCrop.getError(result.getData());
                if (cropError != null) Log.e(TAG, "Crop error: " + cropError.getMessage());
                btnCapture.setEnabled(true);
            } else {
                btnCapture.setEnabled(true);
            }
        });

    // ── Gallery picker ────────────────────────────────────────────────────────
    private final ActivityResultLauncher<String> galleryLauncher =
        registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
            if (uri != null) launchCropper(uri);
        });

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        applyTheme(); // must run before super.onCreate() to take effect on first draw
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!org.opencv.android.OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV native library failed to load");
        }

        cameraPreview = findViewById(R.id.cameraPreview);
        scanGrid      = findViewById(R.id.scanGrid);
        imagePreview  = findViewById(R.id.imagePreview);
        tvResult      = findViewById(R.id.tvResult);
        btnCapture    = findViewById(R.id.btnCapture);
        btnGallery    = findViewById(R.id.btnGallery);
        btnEval       = findViewById(R.id.btnEval); // Initialized bottom sheet view binding
        btnSearch     = findViewById(R.id.btnSearch);
        btnSettings   = findViewById(R.id.btnSettings); // Doubles as the theme toggle
        btnOnlineToggle = findViewById(R.id.btnOnlineToggle);
        btnModeWord   = findViewById(R.id.btnModeWord);
        btnModePage   = findViewById(R.id.btnModePage);

        cameraCard    = findViewById(R.id.cameraCard);
        previewCard   = findViewById(R.id.previewCard);

        SelectableTextHelper.enable(this, tvResult); // Long-press result text: Copy / Search Web / Translate / Share

        onlineOcrClient = new OnlineOcrClient(getApplicationContext());

        cameraExecutor = Executors.newSingleThreadExecutor();
        dbExecutor     = Executors.newSingleThreadExecutor();
        recognitionExecutor = Executors.newSingleThreadExecutor();

        // ── Button listeners ──────────────────────────────────────────────────
        btnCapture.setOnClickListener(v -> captureAndRecognise());
        btnGallery.setOnClickListener(v -> galleryLauncher.launch("image/*"));
        
        // Setup listener for navigation to Batch Evaluation Activity
        btnEval.setOnClickListener(v -> 
            startActivity(new Intent(this, com.indicocr.EvalActivity.class)));

        // Setup listener for navigation to Global Search
        btnSearch.setOnClickListener(v ->
            startActivity(new Intent(this, GlobalSearchActivity.class)));

        btnOnlineToggle.setOnClickListener(v -> toggleOnlineMode());

        // Doubles as the light/dark toggle for now — split into a real
        // settings screen later if you add more preferences.
        btnSettings.setOnClickListener(v -> startActivity(new Intent(this, SettingsActivity.class)));

        btnModeWord.setOnClickListener(v -> setPageMode(false));
        btnModePage.setOnClickListener(v -> setPageMode(true));
        setPageMode(false); // default: fast single-word path

        // ── Camera permission ─────────────────────────────────────────────────
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateOnlineToggleIcon();
    }

    @Override
    protected void onDestroy() {
        cameraExecutor.shutdown();
        dbExecutor.shutdown();
        recognitionExecutor.shutdown();
        if (ocrEngine != null) ocrEngine.close();
        if (textDetector != null) textDetector.close();
        super.onDestroy();
    }

    private void toggleOnlineMode() {
        boolean current = AppSettings.isOnlineMode(this);
        if (!current && !AppSettings.hasApiUrl(this)) {
            Toast.makeText(this, "Set a Server URL in Settings to enable Cloud mode", Toast.LENGTH_LONG).show();
            return;
        }
        AppSettings.setOnlineMode(this, !current);
        updateOnlineToggleIcon();
        Toast.makeText(this, !current ? "Cloud Recognition Enabled" : "Offline Recognition (On-Device)", Toast.LENGTH_SHORT).show();
    }

    private void updateOnlineToggleIcon() {
        boolean online = AppSettings.isOnlineMode(this);
        btnOnlineToggle.setImageResource(online ? R.drawable.ic_cloud : R.drawable.ic_cloud_off);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OCR Engine init
    // ─────────────────────────────────────────────────────────────────────────

    private boolean ensureEngineReady() {
        if (ocrEngine != null) return true;
        
        Log.i(TAG, "Lazy-loading IndicOCR Engine...");
        try {
            ocrEngine = new IndicOCREngine(getApplicationContext());
            Log.i(TAG, "OCR engine ready");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to lazy-init OCR engine: " + e.getMessage());
            runOnUiThread(() -> tvResult.setText("⚠ Engine init failed: " + e.getMessage()));
            return false;
        }
    }

    private boolean ensureTextDetectorReady() {
        if (textDetector != null && textDetector.isReady()) {
            return true;
        }

        closeTextDetector();
        if (!TextDetector.isModelBundled(getApplicationContext())) {
            Log.e(TAG, "TextBPN model is not bundled; Page mode segmentation is unavailable.");
            return false;
        }

        Log.i(TAG, "Lazy-loading TextBPN detector for Page mode.");
        textDetector = new TextDetector(getApplicationContext());
        return textDetector.isReady();
    }

    private void closeTextDetector() {
        if (textDetector == null) return;
        textDetector.close();
        textDetector = null;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
            ProcessCameraProvider.getInstance(this);

        future.addListener(() -> {
            try {
                ProcessCameraProvider provider = future.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    // Cap resolution to 1080p to prevent memory spikes during rotation/cropping.
                    // This is plenty for OCR while being much lighter for the phone to process.
                    .setTargetResolution(new android.util.Size(1920, 1080))
                    .build();

                provider.unbindAll();
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                );
            } catch (Exception e) {
                Log.e(TAG, "Camera bind failed: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureAndRecognise() {
        if (imageCapture == null) return;
        btnCapture.setEnabled(false);
        tvResult.setText("Capturing…");

        // Create a temp file for the capture
        File photoFile = new File(getExternalCacheDir(), "scan_capture.jpg");
        ImageCapture.OutputFileOptions outputOptions =
            new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        imageCapture.takePicture(outputOptions, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                // Perform Auto-Crop based on the Scan Grid overlay before proceeding
                processAutoCrop(photoFile);
            }

            @Override
            public void onError(@NonNull ImageCaptureException exc) {
                Log.e(TAG, "Capture failed: " + exc.getMessage());
                runOnUiThread(() -> {
                    tvResult.setText("Capture failed");
                    btnCapture.setEnabled(true);
                });
            }
        });
    }

    private void processAutoCrop(File photoFile) {
        Bitmap fullBmp = null;
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 2; // Decode at half resolution for auto-crop logic (saves ~75% memory)
            fullBmp = BitmapFactory.decodeFile(photoFile.getAbsolutePath(), options);
            if (fullBmp == null) {
                launchCropper(Uri.fromFile(photoFile));
                return;
            }

            // 1. Calculate Grid Area relative to PreviewView
            float gridWidth  = scanGrid.getWidth();
            float gridHeight = scanGrid.getHeight();
            float gridLeft   = scanGrid.getLeft();
            float gridTop    = scanGrid.getTop();

            float previewWidth  = cameraPreview.getWidth();
            float previewHeight = cameraPreview.getHeight();

            // 2. Map to Bitmap scale with padding to avoid "tight" crops that cut off text
            float scaleX = (float) fullBmp.getWidth()  / previewWidth;
            float scaleY = (float) fullBmp.getHeight() / previewHeight;

            int cropX = Math.round(gridLeft * scaleX);
            int cropY = Math.round(gridTop  * scaleY);
            int cropW = Math.round(gridWidth * scaleX);
            int cropH = Math.round(gridHeight * scaleY);

            // Add 25% padding around the crop area so the user can fine-tune it in uCrop
            int paddingW = cropW / 4;
            int paddingH = cropH / 4;

            cropX -= paddingW;
            cropY -= paddingH;
            cropW += paddingW * 2;
            cropH += paddingH * 2;

            // Bounds check
            cropX = Math.max(0, cropX);
            cropY = Math.max(0, cropY);
            if (cropX + cropW > fullBmp.getWidth())  cropW = fullBmp.getWidth() - cropX;
            if (cropY + cropH > fullBmp.getHeight()) cropH = fullBmp.getHeight() - cropY;

            if (cropW > 10 && cropH > 10) {
                Bitmap croppedBmp = Bitmap.createBitmap(fullBmp, cropX, cropY, cropW, cropH);

                // Save cropped version to a new file to pass to uCrop (optional) or recogniser
                File croppedFile = new File(getExternalCacheDir(), "autocrop_" + System.currentTimeMillis() + ".jpg");
                try (FileOutputStream out = new FileOutputStream(croppedFile)) {
                    croppedBmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
                }

                // RECYCLE: Free memory immediately
                fullBmp.recycle();
                fullBmp = null;
                croppedBmp.recycle();

                // Pass the pre-cropped image to uCrop so the user just fine-tunes it
                launchCropper(Uri.fromFile(croppedFile));
            } else {
                fullBmp.recycle();
                fullBmp = null;
                launchCropper(Uri.fromFile(photoFile));
            }
        } catch (Exception e) {
            Log.e(TAG, "Auto-crop failed: " + e.getMessage());
            if (fullBmp != null) fullBmp.recycle();
            launchCropper(Uri.fromFile(photoFile));
        }
    }

    private void launchCropper(Uri uri) {
        // DEFUSE UI BLOCK: Move heavy normalization to a background thread
        recognitionExecutor.execute(() -> {
            try {
                runOnUiThread(() -> tvResult.setText("Processing image…"));
                
                Uri normalizedUri = normalizeForCrop(uri);
                File croppedFile = new File(getCacheDir(), "cropped_" + System.currentTimeMillis() + ".jpg");
                
                Uri destinationUri = FileProvider.getUriForFile(this, FILE_PROVIDER_AUTHORITY, croppedFile);

                UCrop.Options uOptions = new UCrop.Options();
                uOptions.setFreeStyleCropEnabled(true);
                uOptions.setShowCropGrid(false);
                uOptions.setAllowedGestures(UCropActivity.ALL, UCropActivity.ALL, UCropActivity.ALL);
                uOptions.setHideBottomControls(false);
                uOptions.setImageToCropBoundsAnimDuration(100);
                uOptions.setMaxScaleMultiplier(20f);
                uOptions.useSourceImageAspectRatio();

                UCrop uCrop = UCrop.of(normalizedUri, destinationUri)
                    .withOptions(uOptions)
                    .withMaxResultSize(2000, 2000);

                Intent cropIntent = uCrop.getIntent(this);
                cropIntent.setClass(this, CustomUCropActivity.class);
                cropIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
                
                runOnUiThread(() -> {
                    cropImage.launch(cropIntent);
                    tvResult.setText("Image ready.");
                });
            } catch (IOException e) {
                Log.e(TAG, "Normalization error: " + e.getMessage());
                runOnUiThread(() -> {
                    Toast.makeText(this, "Could not process image", Toast.LENGTH_SHORT).show();
                    btnCapture.setEnabled(true);
                    tvResult.setText("Processing failed.");
                });
            }
        });
    }

    private Uri normalizeForCrop(Uri sourceUri) throws IOException {
        int targetLongSide = 1280; // Faster cropper startup while preserving OCR detail

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(sourceUri)) {
            BitmapFactory.decodeStream(is, null, options);
        }
        if (options.outWidth <= 0 || options.outHeight <= 0) {
            throw new IOException("Could not read image bounds");
        }

        // Early Downsampling: If image is huge, shrink it while loading from disk
        options.inSampleSize = calculateInSampleSize(options, targetLongSide, targetLongSide);
        options.inJustDecodeBounds = false;
        options.inPreferredConfig = Bitmap.Config.RGB_565; // 50% RAM saving immediately

        try (InputStream is = getContentResolver().openInputStream(sourceUri)) {
            Bitmap original = BitmapFactory.decodeStream(is, null, options);
            if (original == null) throw new IOException("Failed to decode bitmap");

            int w = original.getWidth();
            int h = original.getHeight();
            int longSide = Math.max(w, h);
            float scale = (float) targetLongSide / longSide;

            Bitmap normalized;
            if (scale < 0.95f) { // Only scale if meaningfully larger
                int newW = Math.round(w * scale);
                int newH = Math.round(h * scale);
                normalized = Bitmap.createScaledBitmap(original, newW, newH, true);
                original.recycle();
            } else {
                normalized = original;
            }

            File tmp = new File(getCacheDir(), "ucrop_normalized_" + System.currentTimeMillis() + ".jpg");
            try (FileOutputStream out = new FileOutputStream(tmp)) {
                normalized.compress(Bitmap.CompressFormat.JPEG, 88, out);
            }
            
            normalized.recycle();

            return FileProvider.getUriForFile(this, FILE_PROVIDER_AUTHORITY, tmp);
        }
    }

    private void validateAndRecognise(Uri uri) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, options);
        } catch (IOException e) {
            Log.e(TAG, "Validation error: " + e.getMessage());
        }

        if (options.outWidth < 10 || options.outHeight < 10) {
            Toast.makeText(this, "Crop too small. Please try again.", Toast.LENGTH_LONG).show();
            btnCapture.setEnabled(true);
            return;
        }

        recogniseFromUri(uri);
    }

    private void recogniseFromUri(Uri uri) {
        tvResult.setText("Recognising…");
        recognitionExecutor.execute(() -> {
            try {
                Bitmap bmp = decodeSampledBitmapFromUri(uri, 1024, 1024);
                if (bmp != null) {
                    runRecognition(bmp);
                } else {
                    runOnUiThread(() -> tvResult.setText("Failed to decode image"));
                }
            } catch (Exception e) {
                Log.e(TAG, "Recognition error: " + e.getMessage());
                runOnUiThread(() -> tvResult.setText("Failed to load image"));
            }
        });
    }

    private Bitmap decodeSampledBitmapFromUri(Uri uri, int reqWidth, int reqHeight) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, options);
        }

        options.inSampleSize = calculateInSampleSize(options, reqWidth, reqHeight);
        options.inJustDecodeBounds = false;
        options.inPreferredConfig = Bitmap.Config.RGB_565; // 50% memory saving for preview
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(is, null, options);
        }
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    private void runRecognition(Bitmap bmp) {
        boolean onlineMode = AppSettings.isOnlineMode(getApplicationContext());

        if (!onlineMode && !ensureEngineReady()) {
            runOnUiThread(() -> {
                tvResult.setText("Engine failed to start.");
                btnCapture.setEnabled(true);
            });
            return;
        }

        Bitmap processedBmp = trimBlackBorders(bmp);

        runOnUiThread(() -> {
            imagePreview.setImageBitmap(processedBmp);
            previewCard.setVisibility(android.view.View.VISIBLE);
            cameraCard.setVisibility(android.view.View.GONE);
        });

        // RECYCLE: Free the original bitmap immediately to save memory
        if (processedBmp != bmp) {
            bmp.recycle();
        }

        try {
            long t0 = System.currentTimeMillis();
            String text;

            if (onlineMode) {
                // Online mode sends the whole crop to the hosted model in one
                // call — Word/Page mode toggle only applies to on-device
                // recognition, since the server side handles its own
                // segmentation strategy (or doesn't need one).
                text = onlineOcrClient.recognise(processedBmp);
            } else if (!pageMode) {
                // Fast path — exactly what this model was trained for: one
                // word crop, one inference call, no segmentation overhead.
                text = ocrEngine.recognise(processedBmp);
            } else {
                List<WordSegmenter.WordBox> words = new ArrayList<>();
                if (ensureTextDetectorReady()) {
                    words = textDetector.detect(processedBmp);
                } else {
                    Log.e(TAG, "TextBPN detector unavailable; Page mode cannot segment locally.");
                }

                if (words.isEmpty()) {
                    Log.w(TAG, "TextBPN returned no reliable word boxes; skipping page OCR.");
                }

                // Debug: save every detected box drawn on the page
                String debugPath = DebugVisualizer.drawAndSave(
                        getApplicationContext(), processedBmp, words, "final");
                if (debugPath != null) Log.i(TAG, "Debug overlay: " + debugPath);

                text = words.isEmpty() ? "" : recogniseWords(processedBmp, words);
            }

            long ms = System.currentTimeMillis() - t0;
            Log.i(TAG, "Recognised (" + (onlineMode ? "online" : (pageMode ? "page" : "word")) + " mode) in " + ms + " ms: " + text);
            String finalText = text;
            runOnUiThread(() -> {
                tvResult.setText(finalText.isEmpty() ? "(no text detected)" : finalText);
                btnCapture.setEnabled(true);
                if (!finalText.isEmpty() && !isFinishing()) {
                    promptSaveToFolder(finalText, processedBmp);
                }
            });
        } catch (Throwable t) {
            Log.e(TAG, "Recognition failed", t);
            runOnUiThread(() -> {
                tvResult.setText("Recognition failed: " + t.getMessage());
                btnCapture.setEnabled(true);
            });
        }
    }

    /** Recognises every segmented word and reassembles them into lines. */
    private String recogniseWords(Bitmap page, List<WordSegmenter.WordBox> words) throws OrtException {
        StringBuilder sb = new StringBuilder();
        int currentLine = -1;

        // Process in micro-batches to keep memory usage low. 
        // Page mode can detect 100+ words; cropping them all at once 
        // can easily spike RAM to 200MB+ on high-res photos.
        final int MICRO_BATCH_SIZE = 8;
        
        for (int i = 0; i < words.size(); i += MICRO_BATCH_SIZE) {
            int end = Math.min(i + MICRO_BATCH_SIZE, words.size());
            List<Bitmap> batchCrops = new ArrayList<>();
            
            for (int j = i; j < end; j++) {
                batchCrops.add(WordSegmenter.crop(page, words.get(j).rect));
            }

            List<String> recognisedBatch = ocrEngine.recogniseBatch(batchCrops);

            for (int j = 0; j < batchCrops.size(); j++) {
                int globalIdx = i + j;
                WordSegmenter.WordBox w = words.get(globalIdx);
                String text = recognisedBatch.get(j);
                
                if (w.lineIndex != currentLine) {
                    if (currentLine != -1) sb.append('\n');
                    currentLine = w.lineIndex;
                } else if (sb.length() > 0 && sb.charAt(sb.length() - 1) != '\n') {
                    sb.append(' ');
                }
                sb.append(text);

                // RECYCLE IMMEDIATELY: Don't wait for the whole page to finish
                Bitmap crop = batchCrops.get(j);
                if (crop != null) crop.recycle();
            }
        }
        
        return sb.toString();
    }

    private Bitmap trimBlackBorders(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int threshold = 30;

        int top = 0, bottom = height - 1, left = 0, right = width - 1;

        outer1:
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (!isBlack(bitmap.getPixel(x, y), threshold)) { top = y; break outer1; }
            }
        }
        outer2:
        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                if (!isBlack(bitmap.getPixel(x, y), threshold)) { bottom = y; break outer2; }
            }
        }
        outer3:
        for (int x = 0; x < width; x++) {
            for (int y = top; y <= bottom; y++) {
                if (!isBlack(bitmap.getPixel(x, y), threshold)) { left = x; break outer3; }
            }
        }
        outer4:
        for (int x = width - 1; x >= 0; x--) {
            for (int y = top; y <= bottom; y++) {
                if (!isBlack(bitmap.getPixel(x, y), threshold)) { right = x; break outer4; }
            }
        }

        if (right <= left || bottom <= top) return bitmap;

        return Bitmap.createBitmap(bitmap, left, top, (right - left + 1), (bottom - top + 1));
    }

    private void setPageMode(boolean isPage) {
        pageMode = isPage;
        if (!isPage) {
            closeTextDetector();
        }
        int textPrimary = androidx.core.content.ContextCompat.getColor(this, R.color.colorTextPrimary);

        btnModeWord.setBackgroundResource(isPage ? 0 : R.drawable.pill_segment_selected);
        btnModeWord.setTextColor(isPage ? textPrimary : 0xFFFFFFFF);

        btnModePage.setBackgroundResource(isPage ? R.drawable.pill_segment_selected : 0);
        btnModePage.setTextColor(isPage ? 0xFFFFFFFF : textPrimary);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Light / dark theme
    // ─────────────────────────────────────────────────────────────────────────

    /** Reads the saved theme choice and applies it. Must run before super.onCreate(). */
    private void applyTheme() {
        AppCompatDelegate.setDefaultNightMode(AppSettings.getNightMode(getApplicationContext()));
    }

    private boolean isBlack(int pixel, int threshold) {
        int r = Color.red(pixel), g = Color.green(pixel), b = Color.blue(pixel);
        return (r + g + b) / 3 < threshold;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Save to database (Global Search)
    // ─────────────────────────────────────────────────────────────────────────

    private void promptSaveToFolder(String text, Bitmap bmp) {
        new AlertDialog.Builder(this)
                .setTitle("Save to folder")
                .setItems(FOLDERS, (dialog, which) -> saveScan(text, bmp, FOLDERS[which]))
                .setNegativeButton("Skip", null)
                .show();
    }

    private void saveScan(String text, Bitmap bmp, String folder) {
        // Clone the bitmap for the background save to prevent "trying to use a recycled bitmap"
        // if the user starts a new scan while the previous one is still being saved.
        final Bitmap bmpToSave = bmp.copy(bmp.getConfig(), false);
        
        dbExecutor.execute(() -> {
            try {
                String imagePath = saveThumbnail(bmpToSave);

                Document doc = new Document();
                doc.title = text.length() > 30 ? text.substring(0, 30) + "…" : text;
                doc.folder = folder;
                doc.text = text;
                doc.imagePath = imagePath;
                doc.dateAdded = System.currentTimeMillis();

                AppDatabase.getInstance(getApplicationContext()).documentDao().insert(doc);
                runOnUiThread(() -> Toast.makeText(this, "Saved to " + folder, Toast.LENGTH_SHORT).show());
            } finally {
                bmpToSave.recycle();
            }
        });
    }

    /** Saves a copy of the recognised crop to internal storage so it can be shown in search results / detail view. */
    private String saveThumbnail(Bitmap bmp) {
        File dir = new File(getFilesDir(), "scans");
        if (!dir.exists()) dir.mkdirs();
        File file = new File(dir, "scan_" + System.currentTimeMillis() + ".jpg");
        try (FileOutputStream out = new FileOutputStream(file)) {
            bmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
        } catch (IOException e) {
            Log.e(TAG, "Failed saving thumbnail: " + e.getMessage());
            return null;
        }
        return file.getAbsolutePath();
    }
}
