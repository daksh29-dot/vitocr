package com.indicocr;

import android.content.Context;
import android.graphics.Bitmap;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * OnlineOcrClient
 * ===============
 * Sends a word/page crop to the hosted model instead of running inference
 * on-device. Used when AppSettings.isOnlineMode() is true.
 *
 * ============================================================================
 * TODO — PLACEHOLDER PENDING YOUR ACTUAL API CONTRACT
 * ============================================================================
 * Everything below (ENDPOINT_URL, the request body shape, the auth header
 * name, and the response JSON field name) is a reasonable-guess placeholder,
 * NOT verified against your real hosted API. Update the four marked spots
 * once you share the actual docs/endpoint — this class is structured so
 * that's a small, localized change, not a rewrite.
 * ============================================================================
 */
public class OnlineOcrClient {

    private final OkHttpClient http;
    private final Context ctx;

    public OnlineOcrClient(Context context) {
        this.ctx = context.getApplicationContext();
        this.http = new OkHttpClient.Builder()
                .connectTimeout(15, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
    }

    /**
     * Sends one image crop to the hosted model and returns recognized text.
     * Throws IOException on network failure or a non-2xx response — caller
     * should catch this and fall back to a user-facing error message (and
     * optionally offer to retry offline).
     */
    public String recognise(Bitmap crop) throws IOException {
        String apiUrl = AppSettings.getApiUrl(ctx);

        if (apiUrl.isEmpty() || apiUrl.contains("example.com")) {
            throw new IOException("Server URL not set — open Settings to configure.");
        }

        byte[] jpegBytes = bitmapToJpeg(crop);

        // TODO #2: confirm request shape. This assumes multipart/form-data
        // with the image under field name "image" — swap for a base64 JSON
        // body here if your API expects that instead.
        RequestBody imageBody = RequestBody.create(jpegBytes, MediaType.parse("image/jpeg"));
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "crop.jpg", imageBody)
                .build();

        Request request = new Request.Builder()
                .url(apiUrl)
                .post(requestBody)
                .build();

        try (Response response = http.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String body = response.body() != null ? response.body().string() : "";
                throw new IOException("API error " + response.code() + ": " + body);
            }
            String responseBody = response.body() != null ? response.body().string() : "";
            return parseRecognizedText(responseBody);
        }
    }

    /** TODO #4: confirm the response JSON shape. This assumes {"text": "..."}. */
    private String parseRecognizedText(String responseBody) throws IOException {
        try {
            JSONObject json = new JSONObject(responseBody);
            return json.optString("text", "");
        } catch (JSONException e) {
            throw new IOException("Unexpected response format: " + responseBody, e);
        }
    }

    private byte[] bitmapToJpeg(Bitmap bmp) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
        return out.toByteArray();
    }
}