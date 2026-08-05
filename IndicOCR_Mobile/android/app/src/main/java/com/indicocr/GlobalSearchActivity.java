package com.indicocr;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.widget.EditText;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.indicocr.db.AppDatabase;
import com.indicocr.db.Document;
import com.indicocr.db.SearchQueryBuilder;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Global search across every saved scan (documents + PDFs share this same
 * table — see step 9 in the pipeline: PDF export writes here too).
 */
public class GlobalSearchActivity extends AppCompatActivity {

    private EditText etSearch;
    private RecyclerView recyclerView;
    private TextView tvEmpty;
    private SearchResultAdapter adapter;
    private final ExecutorService dbExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_global_search);

        etSearch = findViewById(R.id.etSearch);
        recyclerView = findViewById(R.id.recyclerResults);
        tvEmpty = findViewById(R.id.tvEmpty);

        adapter = new SearchResultAdapter(this::openDocument);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);

        etSearch.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                runSearch(s.toString());
            }

            @Override
            public void afterTextChanged(Editable s) {}
        });

        // Show everything on first open.
        runSearch("");
    }

    private void openDocument(Document doc) {
        Intent intent = new Intent(this, DocumentViewActivity.class);
        intent.putExtra(DocumentViewActivity.EXTRA_DOC_ID, doc.rowid);
        startActivity(intent);
    }

    private void runSearch(String rawQuery) {
        dbExecutor.execute(() -> {
            List<Document> results;
            AppDatabase db = AppDatabase.getInstance(getApplicationContext());
            if (rawQuery.trim().isEmpty()) {
                results = db.documentDao().getAll();
            } else {
                String ftsQuery = SearchQueryBuilder.buildMatchQuery(rawQuery);
                results = ftsQuery.isEmpty() ? db.documentDao().getAll() : db.documentDao().search(ftsQuery);
            }
            List<Document> finalResults = results;
            runOnUiThread(() -> {
                adapter.submitList(finalResults);
                tvEmpty.setVisibility(finalResults.isEmpty() ? android.view.View.VISIBLE : android.view.View.GONE);
            });
        });
    }
}