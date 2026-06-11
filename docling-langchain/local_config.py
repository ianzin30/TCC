"""Configuration for the text-only Docling + LangChain retrieval baseline."""

from __future__ import annotations

import os
from pathlib import Path


ARCH_NAME = "docling_langchain_bge_m3_milvus"
OUTPUT_STEM = "docling-langchain-bge-m3-retrieval-pages"

PARSER_NAME = "docling_langchain_doc_chunks"
PAGE_PROVENANCE = "docling_dl_meta"
EXPORT_TYPE_NAME = "DOC_CHUNKS"

EMBED_MODEL_NAME = "BAAI/bge-m3"
VECTOR_STORE = "milvus_lite"
CHUNK_TOKEN_SIZE = 250
TEXT_RETRIEVAL_K = 100
K_VALUES = (1, 3, 5, 10, 20)

UNANSWERABLE_MARKER = "Not answerable"

EMBED_BATCH_SIZE = 8
EMBED_NORMALIZE = True
EMBED_DEVICE = os.getenv("TEXTUAL_EMBED_DEVICE", "auto")

MILVUS_INDEX_PARAMS = {"index_type": "FLAT", "metric_type": "IP"}
MILVUS_SEARCH_PARAMS = {"metric_type": "IP", "params": {}}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MMLONGBENCH_PARQUET = (
    PROJECT_ROOT / "MMLongBench-Doc" / "data" / "train-00000-of-00001.parquet"
)
MMLONGBENCH_PDFS_DIR = PROJECT_ROOT / "MMLongBench-Doc" / "documents"
OUTPUT_DIR = SCRIPT_DIR / "smoke_results"
CACHE_DIR = SCRIPT_DIR / "cache" / ARCH_NAME
INDEX_CACHE_DIR = CACHE_DIR / "indexes"


def _parse_pdf_index_range(value: str | None) -> tuple[int, int]:
    if not value:
        return (1, 30)
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(
            "TEXTUAL_PDF_INDEX_RANGE must use 'start,end', for example '1,1'."
        )
    start, end = (int(part) for part in parts)
    return (start, end)


PDF_INDEX_RANGE: tuple[int, int] = _parse_pdf_index_range(
    os.getenv("TEXTUAL_PDF_INDEX_RANGE")
)
