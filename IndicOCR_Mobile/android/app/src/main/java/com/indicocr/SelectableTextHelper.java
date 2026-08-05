package com.indicocr;

import android.app.Activity;
import android.app.SearchManager;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.view.ActionMode;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

/**
 * Call SelectableTextHelper.enable(activity, textView) once, after the
 * TextView has its final text (or right after findViewById — it works
 * even before text is set, since it just configures selection behaviour).
 *
 * Android gives Copy + Select All for free once textIsSelectable(true) is
 * set. This adds three extra actions to that same floating toolbar.
 */
public final class SelectableTextHelper {

    private static final int ID_SEARCH_WEB = 10001;
    private static final int ID_TRANSLATE  = 10002;
    private static final int ID_SHARE      = 10003;

    private SelectableTextHelper() {}

    public static void enable(Activity activity, TextView textView) {
        textView.setTextIsSelectable(true);

        textView.setCustomSelectionActionModeCallback(new ActionMode.Callback() {
            @Override
            public boolean onCreateActionMode(ActionMode mode, Menu menu) {
                menu.add(0, ID_SEARCH_WEB, 0, "Search Web");
                menu.add(0, ID_TRANSLATE, 1, "Translate");
                menu.add(0, ID_SHARE, 2, "Share");
                return true;
            }

            @Override
            public boolean onPrepareActionMode(ActionMode mode, Menu menu) {
                return false;
            }

            @Override
            public boolean onActionItemClicked(ActionMode mode, MenuItem item) {
                String selected = getSelectedText(textView);
                if (selected.isEmpty()) return false;

                if (item.getItemId() == ID_SEARCH_WEB) {
                    searchWeb(activity, selected);
                    mode.finish();
                    return true;
                } else if (item.getItemId() == ID_TRANSLATE) {
                    translate(activity, selected);
                    mode.finish();
                    return true;
                } else if (item.getItemId() == ID_SHARE) {
                    share(activity, selected);
                    mode.finish();
                    return true;
                }
                return false;
            }

            @Override
            public void onDestroyActionMode(ActionMode mode) {}
        });
    }

    private static String getSelectedText(TextView tv) {
        int start = tv.getSelectionStart();
        int end = tv.getSelectionEnd();
        if (start < 0 || end < 0 || start == end || tv.getText() == null) return "";
        return tv.getText().toString().substring(Math.min(start, end), Math.max(start, end));
    }

    private static void searchWeb(Activity activity, String text) {
        try {
            Intent intent = new Intent(Intent.ACTION_WEB_SEARCH);
            intent.putExtra(SearchManager.QUERY, text);
            activity.startActivity(intent);
        } catch (ActivityNotFoundException e) {
            Intent fallback = new Intent(Intent.ACTION_VIEW,
                    Uri.parse("https://www.google.com/search?q=" + Uri.encode(text)));
            activity.startActivity(fallback);
        }
    }

    private static void translate(Activity activity, String text) {
        try {
            Intent intent = new Intent(Intent.ACTION_PROCESS_TEXT);
            intent.putExtra(Intent.EXTRA_PROCESS_TEXT, text);
            intent.setType("text/plain");
            intent.setPackage("com.google.android.apps.translate");
            activity.startActivity(intent);
        } catch (ActivityNotFoundException e) {
            try {
                Intent fallback = new Intent(Intent.ACTION_VIEW,
                        Uri.parse("https://translate.google.com/?text=" + Uri.encode(text)));
                activity.startActivity(fallback);
            } catch (ActivityNotFoundException e2) {
                Toast.makeText(activity, "No app available to translate", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private static void share(Activity activity, String text) {
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("text/plain");
        share.putExtra(Intent.EXTRA_TEXT, text);
        activity.startActivity(Intent.createChooser(share, "Share text"));
    }
}