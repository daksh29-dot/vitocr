package com.indicocr;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;

public class SettingsActivity extends AppCompatActivity {

    private RadioGroup themeGroup;
    private SeekBar fontSizeSeekBar;
    private TextView fontSizePreview;
    private CheckBox checkLogging;
    private Button btnClearLog;
    private Button btnExportLog;
    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        setTitle("Settings & Customization");

        prefs = getSharedPreferences("OCR_Settings", MODE_PRIVATE);

        themeGroup = findViewById(R.id.themeGroup);
        fontSizeSeekBar = findViewById(R.id.fontSizeSeekBar);
        fontSizePreview = findViewById(R.id.fontSizePreview);
        checkLogging = findViewById(R.id.checkLogging);
        btnClearLog = findViewById(R.id.btnClearLog);
        btnExportLog = findViewById(R.id.btnExportLog);

        // Load saved theme
        int theme = prefs.getInt("theme_mode", AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
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

            int currentMode = prefs.getInt("theme_mode", AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
            if (newMode != currentMode) {
                prefs.edit().putInt("theme_mode", newMode).apply();
                AppCompatDelegate.setDefaultNightMode(newMode);
            }
        });

        // Font size customization
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

        // Logging controls
        checkLogging.setChecked(prefs.getBoolean("perf_logging", false));
        checkLogging.setOnCheckedChangeListener((buttonView, isChecked) -> 
            prefs.edit().putBoolean("perf_logging", isChecked).apply());

        btnClearLog.setOnClickListener(v -> {
            // Logic to clear logs could go here (e.g., delete a file or database table)
            Toast.makeText(this, "Session log cleared", Toast.LENGTH_SHORT).show();
        });

        btnExportLog.setOnClickListener(v -> {
            // Logic to export logs as XLSX
            Toast.makeText(this, "Log export started", Toast.LENGTH_SHORT).show();
        });
    }
}