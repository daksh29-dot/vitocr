package com.indicocr;

import android.app.Application;
import android.content.SharedPreferences;
import androidx.appcompat.app.AppCompatDelegate;

public class OCRApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        // Apply saved theme as early as possible
        SharedPreferences prefs = getSharedPreferences("OCR_Settings", MODE_PRIVATE);
        int themeMode = prefs.getInt("theme_mode", AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
        AppCompatDelegate.setDefaultNightMode(themeMode);
    }
}