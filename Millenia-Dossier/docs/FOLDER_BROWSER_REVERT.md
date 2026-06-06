# Folder Browser Revert Process

Streamlit has no native folder picker. The Browse UI in the Upload tab is a custom in-app directory browser built with `st.selectbox` navigation (drive root → subdirectory drill-down → confirm).

If this feature breaks or needs to be undone, use the backup below.

## How to Revert

The pre-browse state means: file uploader + text-input folder path (both working). The Browse UI replaces the text input.

```powershell
# From repo root (Millenia Steamlit Process merge)
copy /Y "Millenia-Dossier\_revert_backups\streamlit_app.pre-browse.bak" "Millenia-Dossier\app\streamlit_app.py"
```

After restoring, the Upload tab will have:
- `st.file_uploader` accepting all supported extensions
- `st.text_input` for typing a folder path
- Recursive folder scanning via `Path.rglob("*")`

## What the Backup Captures

The backup at `_revert_backups/streamlit_app.pre-browse.bak` is the Upload tab with two entrypoints:
- File picker for individual uploads
- Text input for folder path -> recursive scan

This state was verified working — streams cleanly at `http://localhost:8602`.

## Files Modified

| File | Change |
|------|--------|
| `app/streamlit_app.py` | Upload tab logic |
| `docs/FOLDER_BROWSER_REVERT.md` | This file |

No other files are affected by this feature.
