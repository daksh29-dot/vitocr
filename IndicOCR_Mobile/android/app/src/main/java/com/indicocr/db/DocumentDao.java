package com.indicocr.db;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.Query;

import java.util.List;

@Dao
public interface DocumentDao {

    @Insert
    long insert(Document doc);

    @Delete
    void delete(Document doc);

    @Query("SELECT rowid, * FROM documents ORDER BY dateAdded DESC")
    List<Document> getAll();

    @Query("SELECT rowid, * FROM documents WHERE folder = :folder ORDER BY dateAdded DESC")
    List<Document> getByFolder(String folder);

    @Query("SELECT rowid, * FROM documents WHERE rowid = :id LIMIT 1")
    Document getById(int id);

    /**
     * Full-text search. Pass a query already built by
     * SearchQueryBuilder.buildMatchQuery() — raw user input will not work
     * correctly with FTS4's MATCH syntax (no wildcards, unescaped quotes, etc).
     */
    @Query("SELECT rowid, * FROM documents WHERE documents MATCH :ftsQuery ORDER BY dateAdded DESC")
    List<Document> search(String ftsQuery);

    @Query("SELECT rowid, * FROM documents WHERE documents MATCH :ftsQuery AND folder = :folder ORDER BY dateAdded DESC")
    List<Document> searchInFolder(String ftsQuery, String folder);
}
