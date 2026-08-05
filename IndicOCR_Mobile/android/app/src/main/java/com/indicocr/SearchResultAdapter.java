package com.indicocr;

import android.text.format.DateFormat;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.indicocr.db.Document;

import java.util.ArrayList;
import java.util.List;

public class SearchResultAdapter extends RecyclerView.Adapter<SearchResultAdapter.ViewHolder> {

    public interface OnDocumentClickListener {
        void onDocumentClick(Document doc);
    }

    private List<Document> items = new ArrayList<>();
    private final OnDocumentClickListener listener;

    public SearchResultAdapter(OnDocumentClickListener listener) {
        this.listener = listener;
    }

    public void submitList(List<Document> newItems) {
        items = newItems != null ? newItems : new ArrayList<>();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_document, parent, false);
        return new ViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Document doc = items.get(position);
        holder.title.setText(doc.title.isEmpty() ? "(untitled)" : doc.title);
        holder.meta.setText(doc.folder + " · " + DateFormat.format("d MMM yyyy, h:mm a", doc.dateAdded));
        holder.snippet.setText(doc.text);
        holder.itemView.setOnClickListener(v -> {
            if (listener != null) listener.onDocumentClick(doc);
        });
    }

    @Override
    public int getItemCount() {
        return items.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView title, meta, snippet;

        ViewHolder(View itemView) {
            super(itemView);
            title = itemView.findViewById(R.id.tvItemTitle);
            meta = itemView.findViewById(R.id.tvItemMeta);
            snippet = itemView.findViewById(R.id.tvItemSnippet);
        }
    }
}
