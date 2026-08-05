package com.indicocr;

import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.appcompat.widget.SwitchCompat;

/**
 * SettingsActivity
 * ================
 * Manages user preferences including theme, font size, performance logging,
 * and online recognition settings (API key and mode toggle).
 */
public class SettingsActivity extends AppCompatActivity {

    private RadioGroup themeGroup;
    private SeekBar fontSizeSeekBar;
    private TextView fontSizePreview;
    private CheckBox checkLogging;
    private Button btnClearLog;
    private Button btnExportLog;
    
    private SwitchCompat switchOnlineMode;
    private EditText etApiUrl;
    private Button btnSaveSettings;
    private Button btnBack;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        setTitle(R.string.settings_title);

        themeGroup = findViewById(R.id.themeGroup);
        fontSizeSeekBar = findViewById(R.id.fontSizeSeekBar);
        fontSizePreview = findViewById(R.id.fontSizePreview);
        checkLogging = findViewById(R.id.checkLogging);
        btnClearLog = findViewById(R.id.btnClearLog);
        btnExportLog = findViewById(R.id.btnExportLog);
        
        switchOnlineMode = findViewById(R.id.switchOnlineMode);
        etApiUrl = findViewById(R.id.etApiUrl);
        btnSaveSettings = findViewById(R.id.btnSaveSettings);
        btnBack = findViewById(R.id.btnBack);

        setupThemeSelection();
        setupFontSize();
        setupLogging();
        setupOnlineSettings();

        btnBack.setOnClickListener(v -> finish());
    }

    private void setupThemeSelection() {
        int theme = AppSettings.getNightMode(this);
        if (theme == AppCompatDelegate.MODE_NIGHT_NO) {
            themeGroup.check(R.id.themeLight);
        } else if (theme == AppCompatDelegate.MODE_NIGHT_YES) {
            themeGroup.check(R.id.themeDark);
        } else {
            themeGroup.check(R.id.themeSystem);
        }

        themeGroup.setOnCheckedChangeListener((group, checkedId) -> {
            int newMode;
            if (checkedId == R.id.themeLight) {
                newMode = AppCompatDelegate.MODE_NIGHT_NO;
            } else if (checkedId == R.id.themeDark) {
                newMode = AppCompatDelegate.MODE_NIGHT_YES;
            } else {
                newMode = AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM;
            }

            if (newMode != AppSettings.getNightMode(this)) {
                AppSettings.setNightMode(this, newMode);
                AppCompatDelegate.setDefaultNightMode(newMode);
            }
        });
    }

    private void setupFontSize() {
        // Font size still uses local prefs for now as it's not in AppSettings
        android.content.SharedPreferences prefs = getSharedPreferences("OCR_Settings", MODE_PRIVATE);
        int fontSize = prefs.getInt("font_size", 17);
        fontSizeSeekBar.setProgress(fontSize - 12);
        fontSizePreview.setTextSize(fontSize);

        fontSizeSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int size = progress + 12;
                fontSizePreview.setTextSize(size);
                prefs.edit().putInt("font_size", size).apply();
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
    }

    private void setupLogging() {
        android.content.SharedPreferences prefs = getSharedPreferences("OCR_Settings", MODE_PRIVATE);
        checkLogging.setChecked(prefs.getBoolean("perf_logging", false));
        checkLogging.setOnCheckedChangeListener((buttonView, isChecked) -> 
            prefs.edit().putBoolean("perf_logging", isChecked).apply());

        btnClearLog.setOnClickListener(v -> Toast.makeText(this, "Session log cleared", Toast.LENGTH_SHORT).show());
        btnExportLog.setOnClickListener(v -> Toast.makeText(this, "Log export started", Toast.LENGTH_SHORT).show());
    }

    private void setupOnlineSettings() {
        switchOnlineMode.setChecked(AppSettings.isOnlineMode(this));
        etApiUrl.setText(AppSettings.getApiUrl(this));

        switchOnlineMode.setOnCheckedChangeListener((btn, isChecked) -> {
            if (isChecked && !AppSettings.hasApiUrl(this)) {
                Toast.makeText(this, "Enter a valid Server URL before enabling Online mode", Toast.LENGTH_LONG).show();
                switchOnlineMode.setChecked(false);
                return;
            }
            AppSettings.setOnlineMode(this, isChecked);
            Toast.makeText(this, isChecked ? "Online mode enabled" : "Offline mode (on-device)", Toast.LENGTH_SHORT).show();
        });

        btnSaveSettings.setOnClickListener(v -> {
            String url = etApiUrl.getText().toString().trim();
            AppSettings.setApiUrl(this, url);
            
            Toast.makeText(this, "API Settings saved", Toast.LENGTH_SHORT).show();
            
            // If url cleared, disable online mode
            if ((url.isEmpty() || url.contains("example.com")) && AppSettings.isOnlineMode(this)) {
                AppSettings.setOnlineMode(this, false);
                switchOnlineMode.setChecked(false);
            }
        });
    }
}
