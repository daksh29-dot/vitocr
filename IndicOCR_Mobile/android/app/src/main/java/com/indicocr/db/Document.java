package com.indicocr.db;

import androidx.annotation.NonNull;
import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Fts4;
import androidx.room.PrimaryKey;

/**
 * A single saved OCR scan.
 *
 * This is an FTS4 virtual table (self-contained, not "external content") —
 * simplest possible setup for Room + full text search. Every column is
 * searchable via MATCH. Good for a personal document store; if you ever
 * scale to tens of thousands of scans, switch to an external-content FTS4
 * table backed by a plain table, but that needs manual sync triggers.
 *
 * IMPORTANT: FTS4 entities in Room require the primary key column to be
 * named exactly "rowid" — do not rename it.
 */
@Fts4
@Entity(tableName = "documents")
public class Document {

    @PrimaryKey(autoGenerate = true)
    public int rowid;

    /** Short label shown in lists — auto-derived from the first ~30 chars of text. */
    @NonNull
    public String title = "";

    /** One of: Receipts, Notes, Assignments, IDs, Others (or custom). */
    @NonNull
    public String folder = "Others";

    /** Full recognised text for this scan. */
    @NonNull
    public String text = "";

    /** Absolute path to the saved thumbnail/crop image on internal storage. Nullable. */
    @ColumnInfo(name = "imagePath")
    public String imagePath;

    /** Epoch millis when this scan was saved. */
    public long dateAdded;
}
