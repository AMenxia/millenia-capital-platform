# Architecture

Document Intelligence Lab uses a DoclingDocument-centered pipeline.

The core design is to preserve the document's ordered structure instead of flattening the PDF into plain text too early.

```text
PDF
→ Docling parse
→ DoclingDocument item map
→ item-specific processing
→ cleaned document feed
→ structured extraction
```

## Document Item Map

After parsing, the app creates an ordered item map. Each item can include:

- item ID
- source document ID
- file name
- page number
- item type
- reading order index
- bounding box
- raw text or table content
- crop path
- review status
- model output status

This item map is saved as:

```text
layout_items.json
```

## Item Routing

Different item types are processed differently.

```text
text → copied into the LLM feed
section header → converted into a Markdown heading
list item → converted into a Markdown list entry
table → corrected through the LLM table workflow
picture/chart → reviewed through the VLM workflow
caption → attached near the relevant item when available
```

## Why this design works

This design keeps parsing, review, correction, and extraction separate.

That makes it easier to inspect failures, rerun specific steps, compare raw and corrected outputs, and export structured results.
