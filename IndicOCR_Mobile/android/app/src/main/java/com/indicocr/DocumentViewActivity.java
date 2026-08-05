package com.indicocr;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.format.DateFormat;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.indicocr.db.AppDatabase;
import com.indicocr.db.Document;

import java.util.concurrent.Executors;

public class DocumentViewActivity extends AppCompatActivity {

    public static final String EXTRA_DOC_ID = "extra_doc_id";

    private TextView tvText, tvMeta;
    private ImageView imgThumbnail;
    private int docId;
    private Document currentDoc;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_document_view);

        tvText = findViewById(R.id.tvDocText);
        tvMeta = findViewById(R.id.tvDocMeta);
        imgThumbnail = findViewById(R.id.imgDocThumbnail);
        findViewById(R.id.btnDelete).setOnClickListener(v -> confirmDelete());

        SelectableTextHelper.enable(this, tvText);

        docId = getIntent().getIntExtra(EXTRA_DOC_ID, -1);
        loadDocument();
    }

    private void loadDocument() {
        if (docId == -1) {
            finish();
            return;
        }
        Executors.newSingleThreadExecutor().execute(() -> {
            Document doc = AppDatabase.getInstance(getApplicationContext()).documentDao().getById(docId);
            runOnUiThread(() -> {
                if (doc == null) {
                    Toast.makeText(this, "Document not found", Toast.LENGTH_SHORT).show();
                    finish();
                    return;
                }
                currentDoc = doc;
                setTitle(doc.title.isEmpty() ? "Document" : doc.title);
                tvText.setText(doc.text);
                tvMeta.setText(doc.folder + " · " + DateFormat.format("d MMM yyyy, h:mm a", doc.dateAdded));

                if (doc.imagePath != null) {
                    Bitmap bmp = BitmapFactory.decodeFile(doc.imagePath);
                    if (bmp != null) {
                        imgThumbnail.setImageBitmap(bmp);
                        imgThumbnail.setVisibility(android.view.View.VISIBLE);
                    }
                }
            });
        });
    }

    private void confirmDelete() {
        if (currentDoc == null) return;
        new AlertDialog.Builder(this)
                .setTitle("Delete this scan?")
                .setMessage("This can't be undone.")
                .setPositiveButton("Delete", (d, w) -> {
                    Executors.newSingleThreadExecutor().execute(() -> {
                        AppDatabase.getInstance(getApplicationContext()).documentDao().delete(currentDoc);
                        runOnUiThread(this::finish);
                    });
                })
                .setNegativeButton("Cancel", null)
                .show();
    }
}
