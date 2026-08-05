package com.indicocr.db;

/**
 * FTS4 MATCH does not accept arbitrary text — quotes, bare "*", and empty
 * tokens will throw. This turns "invoice tax" into "invoice* tax*" so
 * partial/prefix search works as the user types (FTS4 has no infix search
 * without an extra trigram index, prefix is what you get for free).
 */
public final class SearchQueryBuilder {

    private SearchQueryBuilder() {}

    public static String buildMatchQuery(String raw) {
        if (raw == null) return "";
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) return "";

        String[] tokens = trimmed.split("\\s+");
        StringBuilder sb = new StringBuilder();
        for (String token : tokens) {
            String cleaned = token.replaceAll("[\"*]", "");
            if (cleaned.isEmpty()) continue;
            if (sb.length() > 0) sb.append(' ');
            sb.append(cleaned).append('*');
        }
        return sb.toString();
    }
}
