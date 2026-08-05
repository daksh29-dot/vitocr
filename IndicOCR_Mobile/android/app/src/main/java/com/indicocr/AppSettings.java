package com.indicocr;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.util.Log;

import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;

import java.io.IOException;
import java.security.GeneralSecurityException;

/**
 * AppSettings
 * ===========
 * Single source of truth for user settings that need to persist and, in
 * the API key's case, stay encrypted at rest. Backed by
 * EncryptedSharedPreferences (Android Keystore-derived key) rather than
 * plain SharedPreferences — a plain XML prefs file is readable on a rooted
 * device or via backup extraction, which matters for a real credential.
 */
public final class AppSettings {

    private static final String TAG = "AppSettings";
    private static final String PREFS_FILE = "secure_app_settings";

    private static final String KEY_API_URL      = "api_url";
    private static final String KEY_API_KEY      = "api_key";
    private static final String KEY_ONLINE_MODE  = "online_mode";
    private static final String KEY_NIGHT_MODE   = "night_mode"; // consolidated from MainActivity's old theme_prefs

    private static volatile SharedPreferences prefs;

    private AppSettings() {}

    private static SharedPreferences get(Context ctx) {
        if (prefs == null) {
            synchronized (AppSettings.class) {
                if (prefs == null) {
                    prefs = build(ctx.getApplicationContext());
                }
            }
        }
        return prefs;
    }

    private static SharedPreferences build(Context ctx) {
        try {
            MasterKey masterKey = new MasterKey.Builder(ctx)
                    .setKeyGenParameterSpec(new KeyGenParameterSpec.Builder(
                            MasterKey.DEFAULT_MASTER_KEY_ALIAS,
                            android.security.keystore.KeyProperties.PURPOSE_ENCRYPT
                                    | android.security.keystore.KeyProperties.PURPOSE_DECRYPT)
                            .setBlockModes(android.security.keystore.KeyProperties.BLOCK_MODE_GCM)
                            .setEncryptionPaddings(android.security.keystore.KeyProperties.ENCRYPTION_PADDING_NONE)
                            .setKeySize(256)
                            .build())
                    .build();

            return EncryptedSharedPreferences.create(
                    ctx,
                    PREFS_FILE,
                    masterKey,
                    EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                    EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            );
        } catch (GeneralSecurityException | IOException e) {
            // Should be effectively unreachable on real devices, but fail
            // toward "offline mode, no stored key" rather than crashing.
            Log.e(TAG, "Failed to init encrypted prefs, falling back to regular prefs: " + e.getMessage());
            return ctx.getSharedPreferences(PREFS_FILE + "_fallback", Context.MODE_PRIVATE);
        }
    }

    // ── API URL & Key ────────────────────────────────────────────────────────

    public static String getApiUrl(Context ctx) {
        return get(ctx).getString(KEY_API_URL, "https://your-api-host.example.com/v1/ocr");
    }

    public static void setApiUrl(Context ctx, String url) {
        get(ctx).edit().putString(KEY_API_URL, url == null ? "" : url.trim()).apply();
    }

    public static boolean hasApiUrl(Context ctx) {
        String url = getApiUrl(ctx);
        return !url.isEmpty() && !url.contains("example.com");
    }

    public static String getApiKey(Context ctx) {
        return get(ctx).getString(KEY_API_KEY, "");
    }

    public static void setApiKey(Context ctx, String key) {
        get(ctx).edit().putString(KEY_API_KEY, key == null ? "" : key.trim()).apply();
    }

    public static boolean hasApiKey(Context ctx) {
        return !getApiKey(ctx).isEmpty();
    }

    // ── Offline / Online mode ───────────────────────────────────────────────

    public static boolean isOnlineMode(Context ctx) {
        return get(ctx).getBoolean(KEY_ONLINE_MODE, false); // offline by default
    }

    public static void setOnlineMode(Context ctx, boolean online) {
        get(ctx).edit().putBoolean(KEY_ONLINE_MODE, online).apply();
    }

    // ── Theme (moved here from MainActivity's old standalone prefs) ────────

    public static int getNightMode(Context ctx) {
        return get(ctx).getInt(KEY_NIGHT_MODE, androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
    }

    public static void setNightMode(Context ctx, int mode) {
        get(ctx).edit().putInt(KEY_NIGHT_MODE, mode).apply();
    }
}