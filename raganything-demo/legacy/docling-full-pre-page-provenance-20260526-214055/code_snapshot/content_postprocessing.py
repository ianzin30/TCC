"""Local normalization for parser content before multimodal processing."""

from __future__ import annotations

from typing import Any


def _cell_text(cell: Any) -> str:
    if isinstance(cell, dict):
        cell = cell.get("text", "")
    text = str(cell or "")
    return " ".join(text.replace("|", r"\|").split())


def _markdown_table(grid: Any) -> str | None:
    if not isinstance(grid, list) or not grid:
        return None
    if not all(isinstance(row, list) for row in grid):
        return None

    rows = [[_cell_text(cell) for cell in row] for row in grid]
    column_count = max((len(row) for row in rows), default=0)
    if not column_count:
        return None

    padded_rows = [row + [""] * (column_count - len(row)) for row in rows]
    rendered_rows = ["| " + " | ".join(row) + " |" for row in padded_rows]
    separator = "| " + " | ".join(["---"] * column_count) + " |"
    rendered_rows.insert(1, separator)
    return "\n".join(rendered_rows)


def normalize_docling_tables(content_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Replace verbose Docling table metadata with text-only Markdown tables."""
    normalized_content: list[dict[str, Any]] = []
    reductions: list[dict[str, int]] = []

    for item in content_list:
        normalized_item = item
        table_body = item.get("table_body")
        if item.get("type") == "table" and isinstance(table_body, dict):
            compact_body = _markdown_table(table_body.get("grid"))
            if compact_body:
                normalized_item = dict(item)
                normalized_item["table_body"] = compact_body
                reductions.append(
                    {
                        "raw_chars": len(str(table_body)),
                        "compact_chars": len(compact_body),
                    }
                )
        normalized_content.append(normalized_item)

    return normalized_content, reductions


def print_table_normalization(reductions: list[dict[str, int]]) -> None:
    for index, reduction in enumerate(reductions, start=1):
        raw_chars = reduction["raw_chars"]
        compact_chars = reduction["compact_chars"]
        percent = (1 - compact_chars / raw_chars) * 100 if raw_chars else 0.0
        print(
            f"[normalize] table {index}: {raw_chars} -> {compact_chars} chars "
            f"({percent:.1f}% smaller)",
            flush=True,
        )
