# Pipeline

## 1. Upload + Template

Upload one or more PDFs and select an extraction template.

## 2. Run Docling

Docling parses the PDFs and creates raw outputs, an outlined PDF, a layout item map, table manifests, and visual item manifests.

## 3. Item Review + Quick Skim

The app displays detected items in reading order.

Quick skim labels help identify visual crops that are likely useful, empty, decorative, or needing review.

## 4. LLM Table Correction

Tables detected by Docling are sent to a text LLM for structure cleanup.

The output is saved as structured rows and columns.

## 5. Image / Chart VLM Review

Visual crops can be reviewed with optional human notes before VLM analysis.

## 6. Build LLM Feed

The app rebuilds a clean Markdown feed from the reviewed item sequence.

## 7. Analysis + Extraction

The selected template is applied to the cleaned feed.

## 8. Excel Preview + Download

Structured outputs are previewed in the app and exported as an Excel workbook.
