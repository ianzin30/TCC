"""Evaluate text-only Docling + LangChain retrieval on MMLongBench pages.

Usage:
    uv run python evaluate_docling_langchain_pages.py

For a small smoke run:
    TEXTUAL_PDF_INDEX_RANGE=1,1 uv run python evaluate_docling_langchain_pages.py

The evaluator builds a local Milvus Lite index from Docling text chunks and
compares retrieved chunk page provenance with MMLongBench evidence pages. It
does retrieval only; no LLM answer generation is performed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from transformers import AutoTokenizer

from local_config import (
    ARCH_NAME,
    CACHE_DIR,
    CHUNK_TOKEN_SIZE,
    EMBED_BATCH_SIZE,
    EMBED_DEVICE,
    EMBED_MODEL_NAME,
    EMBED_NORMALIZE,
    EXPORT_TYPE_NAME,
    INDEX_CACHE_DIR,
    K_VALUES,
    MILVUS_INDEX_PARAMS,
    MILVUS_SEARCH_PARAMS,
    MMLONGBENCH_PARQUET,
    MMLONGBENCH_PDFS_DIR,
    OUTPUT_DIR,
    OUTPUT_STEM,
    PAGE_PROVENANCE,
    PARSER_NAME,
    PDF_INDEX_RANGE,
    TEXT_RETRIEVAL_K,
    UNANSWERABLE_MARKER,
    VECTOR_STORE,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MANIFEST_VERSION = 1


@dataclass
class DocumentIndex:
    vectorstore: Milvus
    manifest: dict[str, Any]
    artifacts: dict[str, Any]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, list) else [parsed]


def _load_questions(pdf_index_range: tuple[int, int]) -> pd.DataFrame:
    """Load answerable QA pairs for a one-based inclusive PDF index range."""
    df = pd.read_parquet(MMLONGBENCH_PARQUET)
    document_ids = df["doc_id"].drop_duplicates().reset_index(drop=True)

    if len(pdf_index_range) != 2:
        raise ValueError("PDF_INDEX_RANGE must contain exactly (start_index, end_index).")
    start_index, end_index = pdf_index_range
    if (
        not isinstance(start_index, int)
        or not isinstance(end_index, int)
        or start_index < 1
        or end_index < start_index
        or end_index > len(document_ids)
    ):
        raise ValueError(
            "PDF_INDEX_RANGE must be a valid one-based inclusive range; "
            f"received {pdf_index_range} for {len(document_ids)} PDF(s)."
        )

    selected_documents = document_ids.iloc[start_index - 1 : end_index].tolist()
    print(
        f"[grid] PDF_INDEX_RANGE={pdf_index_range} -> "
        f"{len(selected_documents)} PDF(s)",
        flush=True,
    )
    for index, doc_id in enumerate(selected_documents, start=start_index):
        print(f"[grid]   {index}: {doc_id}", flush=True)

    mask = df["doc_id"].isin(selected_documents) & (df["answer"] != UNANSWERABLE_MARKER)
    filtered = df.loc[mask].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No answerable questions found for PDF_INDEX_RANGE={pdf_index_range}, "
            f"documents={selected_documents}."
        )
    return filtered


def _resolve_pdf_path(doc_id: str) -> Path:
    path = MMLONGBENCH_PDFS_DIR / doc_id
    if not path.exists():
        raise FileNotFoundError(f"PDF not present on disk: {path}")
    return path


def _safe_name(value: str, *, max_len: int = 180) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)
    safe = safe.strip("_") or "document"
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{safe[: max_len - 13]}_{digest}"


def _doc_cache_dir(doc_id: str) -> Path:
    return INDEX_CACHE_DIR / Path(doc_id).stem


def _manifest_path(doc_id: str) -> Path:
    return _doc_cache_dir(doc_id) / "index_manifest.json"


def _milvus_db_path(doc_id: str) -> Path:
    return _doc_cache_dir(doc_id) / "milvus_lite.db"


def _collection_name(doc_id: str) -> str:
    return f"lc_{_safe_name(Path(doc_id).stem, max_len=180)}"


def _ensure_milvus_orm_connection(db_path: Path) -> str:
    """Register the ORM alias expected internally by langchain-milvus.

    langchain-milvus 0.3.x uses MilvusClient, but still touches PyMilvus'
    ORM Collection API for field extraction. MilvusClient creates a stable
    alias from the URI without registering an ORM connection, so we register it
    explicitly before constructing the vector store.
    """
    from pymilvus import MilvusClient, connections

    uri = str(db_path)
    alias = MilvusClient(uri=uri)._using
    if not connections.has_connection(alias):
        connections.connect(alias=alias, uri=uri)
    return str(alias)


def _select_embed_device() -> str:
    if EMBED_DEVICE != "auto":
        return EMBED_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _build_embedding() -> tuple[HuggingFaceEmbeddings, str]:
    device = _select_embed_device()
    embedding = HuggingFaceEmbeddings(
        model=EMBED_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": EMBED_NORMALIZE,
            "batch_size": EMBED_BATCH_SIZE,
        },
    )
    return embedding, device


def _coerce_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _add_page_number(pages: set[int], value: Any) -> None:
    if isinstance(value, bool):
        return
    try:
        page_number = int(value)
    except (TypeError, ValueError):
        return
    if page_number >= 1:
        pages.add(page_number)


def _page_numbers_from_metadata(metadata: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    dl_meta = _coerce_json_object(metadata.get("dl_meta"))

    for item in dl_meta.get("doc_items", []):
        if not isinstance(item, dict):
            continue
        for provenance in item.get("prov", []):
            if isinstance(provenance, dict):
                _add_page_number(pages, provenance.get("page_no"))

    for key in ("page_no", "page_number", "page"):
        _add_page_number(pages, metadata.get(key))

    return sorted(pages)


def _headings_from_metadata(metadata: dict[str, Any]) -> list[str]:
    dl_meta = _coerce_json_object(metadata.get("dl_meta"))
    headings = dl_meta.get("headings", [])
    if not isinstance(headings, list):
        return []
    return [str(heading) for heading in headings]


def _prepare_documents(doc_id: str, raw_docs: list[Document]) -> list[Document]:
    prepared: list[Document] = []
    for index, doc in enumerate(raw_docs, start=1):
        content = doc.page_content.strip()
        if not content:
            continue

        page_numbers = _page_numbers_from_metadata(doc.metadata)
        if not page_numbers:
            raise RuntimeError(
                f"Docling chunk {index} for {doc_id} has no dl_meta prov.page_no."
            )

        chunk_id = f"{Path(doc_id).stem}-chunk-{index:05d}"
        headings = _headings_from_metadata(doc.metadata)
        prepared.append(
            Document(
                page_content=content,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": str(doc.metadata.get("source") or doc_id),
                    "page_numbers_json": json.dumps(page_numbers, ensure_ascii=False),
                    "page_numbers_csv": ",".join(str(page) for page in page_numbers),
                    "headings_json": json.dumps(headings, ensure_ascii=False),
                    "content_type": "text",
                },
            )
        )

    if not prepared:
        raise RuntimeError(f"Docling produced no non-empty text chunks for {doc_id}.")
    return prepared


def _load_docling_chunks(pdf_path: Path) -> tuple[list[Document], float]:
    started = time.perf_counter()
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_NAME),
        max_tokens=CHUNK_TOKEN_SIZE,
    )
    loader = DoclingLoader(
        file_path=[str(pdf_path)],
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=tokenizer),
    )
    docs = loader.load()
    return docs, round(time.perf_counter() - started, 2)


def _base_manifest(doc_id: str, pdf_path: Path) -> dict[str, Any]:
    stat = pdf_path.stat()
    return {
        "manifest_version": MANIFEST_VERSION,
        "arch": ARCH_NAME,
        "doc_id": doc_id,
        "pdf_filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "pdf_size_bytes": stat.st_size,
        "pdf_mtime_ns": stat.st_mtime_ns,
        "parser": PARSER_NAME,
        "export_type": EXPORT_TYPE_NAME,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "page_provenance": PAGE_PROVENANCE,
        "embedding_model_name": EMBED_MODEL_NAME,
        "embedding_normalize": EMBED_NORMALIZE,
        "embedding_batch_size": EMBED_BATCH_SIZE,
        "vector_store": VECTOR_STORE,
        "milvus_index_params": MILVUS_INDEX_PARAMS,
        "milvus_search_params": MILVUS_SEARCH_PARAMS,
        "text_retrieval_k": TEXT_RETRIEVAL_K,
        "collection_name": _collection_name(doc_id),
        "milvus_db_path": str(_milvus_db_path(doc_id)),
    }


def _manifest_matches(manifest: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "manifest_version",
        "arch",
        "doc_id",
        "pdf_filename",
        "pdf_size_bytes",
        "parser",
        "export_type",
        "chunk_token_size",
        "page_provenance",
        "embedding_model_name",
        "embedding_normalize",
        "embedding_batch_size",
        "vector_store",
        "milvus_index_params",
        "milvus_search_params",
        "text_retrieval_k",
        "collection_name",
    )
    return all(manifest.get(key) == expected.get(key) for key in keys)


def _load_cached_index(
    doc_id: str,
    pdf_path: Path,
    embedding: HuggingFaceEmbeddings,
    expected_manifest: dict[str, Any],
) -> DocumentIndex | None:
    manifest_path = _manifest_path(doc_id)
    db_path = _milvus_db_path(doc_id)
    if not manifest_path.exists() or not db_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not _manifest_matches(manifest, expected_manifest):
        return None

    started = time.perf_counter()
    _ensure_milvus_orm_connection(db_path)
    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name=manifest["collection_name"],
        connection_args={"uri": str(db_path)},
        index_params=MILVUS_INDEX_PARAMS,
        search_params=MILVUS_SEARCH_PARAMS,
        consistency_level="Strong",
        drop_old=False,
    )
    cache_load_duration_s = round(time.perf_counter() - started, 2)
    artifacts = _document_artifacts(
        pdf_path,
        manifest,
        index_cache_hit=True,
        parse_duration_s=0.0,
        indexing_duration_s=0.0,
        cache_load_duration_s=cache_load_duration_s,
    )
    return DocumentIndex(vectorstore=vectorstore, manifest=manifest, artifacts=artifacts)


def _document_artifacts(
    pdf_path: Path,
    manifest: dict[str, Any],
    *,
    index_cache_hit: bool,
    parse_duration_s: float,
    indexing_duration_s: float,
    cache_load_duration_s: float,
) -> dict[str, Any]:
    return {
        "pdf_path": str(pdf_path),
        "chunk_count": manifest.get("chunk_count"),
        "text_characters": manifest.get("text_characters"),
        "page_numbered_chunks": manifest.get("page_numbered_chunks"),
        "unique_pages": manifest.get("unique_pages"),
        "index_cache_hit": index_cache_hit,
        "parse_duration_s": parse_duration_s,
        "indexing_duration_s": indexing_duration_s,
        "cache_load_duration_s": cache_load_duration_s,
        "cached_parse_duration_s": manifest.get("parse_duration_s"),
        "cached_indexing_duration_s": manifest.get("indexing_duration_s"),
        "milvus_db_path": manifest.get("milvus_db_path"),
        "collection_name": manifest.get("collection_name"),
        "built_at_utc": manifest.get("built_at_utc"),
    }


def _build_index(
    doc_id: str, group: pd.DataFrame, embedding: HuggingFaceEmbeddings
) -> DocumentIndex:
    pdf_path = _resolve_pdf_path(doc_id)
    expected_manifest = _base_manifest(doc_id, pdf_path)
    cached = _load_cached_index(doc_id, pdf_path, embedding, expected_manifest)
    if cached is not None:
        print(f"[index] {doc_id}: loaded cached Milvus Lite index", flush=True)
        return cached

    doc_cache_dir = _doc_cache_dir(doc_id)
    if doc_cache_dir.exists():
        shutil.rmtree(doc_cache_dir)
    doc_cache_dir.mkdir(parents=True, exist_ok=True)

    raw_docs, parse_duration_s = _load_docling_chunks(pdf_path)
    docs = _prepare_documents(doc_id, raw_docs)
    page_numbers = {
        page
        for doc in docs
        for page in json.loads(doc.metadata["page_numbers_json"])
    }

    started = time.perf_counter()
    _ensure_milvus_orm_connection(_milvus_db_path(doc_id))
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=expected_manifest["collection_name"],
        connection_args={"uri": str(_milvus_db_path(doc_id))},
        index_params=MILVUS_INDEX_PARAMS,
        search_params=MILVUS_SEARCH_PARAMS,
        consistency_level="Strong",
        drop_old=True,
    )
    indexing_duration_s = round(time.perf_counter() - started, 2)

    manifest = {
        **expected_manifest,
        "questions": len(group),
        "chunk_count": len(docs),
        "page_numbered_chunks": len(docs),
        "unique_pages": sorted(page_numbers),
        "text_characters": sum(len(doc.page_content) for doc in docs),
        "parse_duration_s": parse_duration_s,
        "indexing_duration_s": indexing_duration_s,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _manifest_path(doc_id).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifacts = _document_artifacts(
        pdf_path,
        manifest,
        index_cache_hit=False,
        parse_duration_s=parse_duration_s,
        indexing_duration_s=indexing_duration_s,
        cache_load_duration_s=0.0,
    )
    print(
        f"[index] {doc_id}: built {len(docs)} text chunk(s); "
        f"parse_duration_s={parse_duration_s}; "
        f"indexing_duration_s={indexing_duration_s}",
        flush=True,
    )
    return DocumentIndex(vectorstore=vectorstore, manifest=manifest, artifacts=artifacts)


def _pages_from_retrieved_metadata(metadata: dict[str, Any]) -> list[int]:
    raw = metadata.get("page_numbers_json")
    if isinstance(raw, str):
        pages = json.loads(raw)
    else:
        pages = raw
    if not isinstance(pages, list):
        raise RuntimeError(f"Retrieved chunk has invalid page_numbers_json={raw!r}")
    page_numbers = sorted({int(page) for page in pages if int(page) >= 1})
    if not page_numbers:
        raise RuntimeError("Retrieved chunk has empty page_numbers_json.")
    return page_numbers


def _ranked_chunks(
    retrieved: list[tuple[Document, float]], evidence_pages: list[int]
) -> list[dict[str, Any]]:
    evidence = set(evidence_pages)
    ranked: list[dict[str, Any]] = []
    for rank, (doc, score) in enumerate(retrieved, start=1):
        page_numbers = _pages_from_retrieved_metadata(doc.metadata)
        ranked.append(
            {
                "rank": rank,
                "chunk_id": str(doc.metadata.get("chunk_id") or f"chunk-{rank}"),
                "score": float(score),
                "page_numbers": page_numbers,
                "content_type": str(doc.metadata.get("content_type") or "text"),
                "content_preview": doc.page_content[:180],
                "matches_evidence": bool(evidence.intersection(page_numbers)),
            }
        )
    return ranked


def _ranked_pages(
    ranked_chunks: list[dict[str, Any]], evidence_pages: list[int]
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    evidence = set(evidence_pages)
    for chunk in ranked_chunks:
        for page_number in chunk["page_numbers"]:
            if page_number in seen:
                continue
            seen.add(page_number)
            pages.append(
                {
                    "page_number": page_number,
                    "first_chunk_rank": chunk["rank"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                    "matches_evidence": page_number in evidence,
                }
            )
    return pages


def _metrics_for_question(
    ranked_chunks: list[dict[str, Any]], evidence_pages: list[int]
) -> dict[str, Any]:
    evidence = set(evidence_pages)
    first_hit_rank: int | None = None
    for chunk in ranked_chunks:
        if evidence.intersection(chunk["page_numbers"]):
            first_hit_rank = chunk["rank"]
            break

    metrics: dict[str, Any] = {
        "first_hit_rank": first_hit_rank,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "retrieved_chunks": len(ranked_chunks),
        "retrieved_unique_pages": len(
            {
                page
                for chunk in ranked_chunks
                for page in chunk.get("page_numbers", [])
            }
        ),
        "evidence_pages_count": len(evidence),
    }
    for k in K_VALUES:
        pages_at_k = {
            page
            for chunk in ranked_chunks[:k]
            for page in chunk.get("page_numbers", [])
        }
        matched = evidence.intersection(pages_at_k)
        metrics[f"hit_at_{k}"] = bool(matched)
        metrics[f"evidence_page_recall_at_{k}"] = (
            len(matched) / len(evidence) if evidence else 0.0
        )
    return metrics


def _failed_result(doc_id: str, row: Any, error: str) -> dict[str, Any]:
    evidence_pages = [int(page) for page in _parse_list(row["evidence_pages"])]
    return {
        "arch": ARCH_NAME,
        "model_name": EMBED_MODEL_NAME,
        "embedding_model_name": EMBED_MODEL_NAME,
        "doc_id": doc_id,
        "question": row["question"],
        "evidence_pages": evidence_pages,
        "evidence_sources": _parse_list(row.get("evidence_sources")),
        "parser": PARSER_NAME,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "page_provenance": PAGE_PROVENANCE,
        "text_llm_model": None,
        "vision_llm_model": None,
        "document_artifacts": {},
        "ranked_chunks": [],
        "ranked_pages": [],
        "retrieval_metadata": {},
        "metrics": _metrics_for_question([], evidence_pages),
        "duration_s": 0.0,
        "error": error,
    }


def _evaluate_document(
    doc_id: str, group: pd.DataFrame, embedding: HuggingFaceEmbeddings
) -> list[dict[str, Any]]:
    try:
        document_index = _build_index(doc_id, group, embedding)
    except Exception as exc:
        error = repr(exc)
        return [_failed_result(doc_id, row, error) for _, row in group.iterrows()]

    results: list[dict[str, Any]] = []
    for question_index, (_, row) in enumerate(group.iterrows(), start=1):
        evidence_pages = [int(page) for page in _parse_list(row["evidence_pages"])]
        evidence_sources = _parse_list(row.get("evidence_sources"))
        print(
            f"[retrieval {question_index}/{len(group)}] {doc_id}: "
            f"{row['question']}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            retrieved = document_index.vectorstore.similarity_search_with_score(
                row["question"], k=TEXT_RETRIEVAL_K
            )
            if not retrieved:
                raise RuntimeError("Milvus retrieval returned no chunks.")
            ranked_chunks = _ranked_chunks(retrieved, evidence_pages)
            ranked_pages = _ranked_pages(ranked_chunks, evidence_pages)
            metrics = _metrics_for_question(ranked_chunks, evidence_pages)
            error = None
        except Exception as exc:
            ranked_chunks = []
            ranked_pages = []
            metrics = _metrics_for_question([], evidence_pages)
            error = repr(exc)

        results.append(
            {
                "arch": ARCH_NAME,
                "model_name": EMBED_MODEL_NAME,
                "embedding_model_name": EMBED_MODEL_NAME,
                "doc_id": doc_id,
                "question": row["question"],
                "evidence_pages": evidence_pages,
                "evidence_sources": evidence_sources,
                "parser": document_index.manifest["parser"],
                "chunk_token_size": document_index.manifest["chunk_token_size"],
                "page_provenance": document_index.manifest["page_provenance"],
                "text_llm_model": None,
                "vision_llm_model": None,
                "document_artifacts": document_index.artifacts,
                "ranked_chunks": ranked_chunks,
                "ranked_pages": ranked_pages,
                "retrieval_metadata": {
                    "retrieval_unit": "text_chunk",
                    "text_retrieval_k": TEXT_RETRIEVAL_K,
                    "vector_store": VECTOR_STORE,
                    "milvus_index_params": MILVUS_INDEX_PARAMS,
                    "milvus_search_params": MILVUS_SEARCH_PARAMS,
                    "embedding_normalize": EMBED_NORMALIZE,
                },
                "metrics": metrics,
                "duration_s": round(time.perf_counter() - started, 2),
                "error": error,
            }
        )
    return results


def _summarize_by_document(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_doc.setdefault(result["doc_id"], []).append(result)

    summaries: list[dict[str, Any]] = []
    for doc_id, doc_results in by_doc.items():
        durations = [float(result["duration_s"]) for result in doc_results]
        first_hit_ranks = [
            result["metrics"]["first_hit_rank"]
            for result in doc_results
            if result["metrics"]["first_hit_rank"] is not None
        ]
        summary: dict[str, Any] = {
            "doc_id": doc_id,
            "questions": len(doc_results),
            "failed_queries": sum(1 for result in doc_results if result["error"]),
            "query_duration_total_s": round(sum(durations), 2),
            "query_duration_mean_s": round(_mean(durations), 2),
            "query_duration_median_s": round(_median(durations), 2),
            "first_hit_rank_mean": (
                round(_mean([float(rank) for rank in first_hit_ranks]), 4)
                if first_hit_ranks
                else None
            ),
            "mrr": _mean([result["metrics"]["mrr"] for result in doc_results]),
            "document_artifacts": doc_results[0].get("document_artifacts", {}),
        }
        for k in K_VALUES:
            summary[f"hit_at_{k}"] = _mean(
                [
                    1.0 if result["metrics"][f"hit_at_{k}"] else 0.0
                    for result in doc_results
                ]
            )
            summary[f"evidence_page_recall_at_{k}"] = _mean(
                [
                    result["metrics"][f"evidence_page_recall_at_{k}"]
                    for result in doc_results
                ]
            )
        summaries.append(summary)
    return summaries


def _summarize(results: list[dict[str, Any]], embed_device: str) -> dict[str, Any]:
    question_count = len(results)
    document_summaries = _summarize_by_document(results)
    query_durations = [float(result["duration_s"]) for result in results]
    first_hit_ranks = [
        result["metrics"]["first_hit_rank"]
        for result in results
        if result["metrics"]["first_hit_rank"] is not None
    ]
    artifact_values = [
        summary.get("document_artifacts", {}) for summary in document_summaries
    ]
    summary: dict[str, Any] = {
        "arch": ARCH_NAME,
        "model_name": EMBED_MODEL_NAME,
        "embedding_model_name": EMBED_MODEL_NAME,
        "device": embed_device,
        "dtype": None,
        "pdf_index_range": list(PDF_INDEX_RANGE),
        "documents": len(document_summaries),
        "questions": question_count,
        "failed_queries": sum(1 for result in results if result["error"]),
        "parser": PARSER_NAME,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "page_provenance": PAGE_PROVENANCE,
        "text_llm_model": None,
        "vision_llm_model": None,
        "text_llm_options": {
            "embedding_model_name": EMBED_MODEL_NAME,
            "embedding_batch_size": EMBED_BATCH_SIZE,
            "embedding_normalize": EMBED_NORMALIZE,
            "vector_store": VECTOR_STORE,
            "text_retrieval_k": TEXT_RETRIEVAL_K,
        },
        "vision_llm_options": None,
        "text_llm_think": None,
        "vision_llm_think": None,
        "gpu_offload_policy": f"sentence_transformers_device_{embed_device}",
        "query_duration_total_s": round(sum(query_durations), 2),
        "query_duration_mean_s": round(_mean(query_durations), 2),
        "query_duration_median_s": round(_median(query_durations), 2),
        "first_hit_rank_mean": (
            round(_mean([float(rank) for rank in first_hit_ranks]), 4)
            if first_hit_ranks
            else None
        ),
        "mrr": (
            sum(result["metrics"]["mrr"] for result in results) / question_count
            if question_count
            else 0.0
        ),
        "processing_duration_available": True,
        "processing_duration_note": (
            "Textual indexing duration includes Docling DOC_CHUNKS parsing plus "
            "BGE-M3 embedding insertion into Milvus Lite. Query duration measures "
            "query embedding plus vector search."
        ),
        "artifact_totals": {
            "text_characters": sum(
                int(value.get("text_characters") or 0) for value in artifact_values
            ),
            "text_chunks": sum(
                int(value.get("chunk_count") or 0) for value in artifact_values
            ),
            "page_numbered_chunks": sum(
                int(value.get("page_numbered_chunks") or 0)
                for value in artifact_values
            ),
            "parse_duration_total_s": round(
                sum(float(value.get("parse_duration_s") or 0.0) for value in artifact_values),
                2,
            ),
            "indexing_duration_total_s": round(
                sum(
                    float(value.get("indexing_duration_s") or 0.0)
                    for value in artifact_values
                ),
                2,
            ),
            "cache_load_duration_total_s": round(
                sum(
                    float(value.get("cache_load_duration_s") or 0.0)
                    for value in artifact_values
                ),
                2,
            ),
            "cached_parse_duration_total_s": round(
                sum(
                    float(value.get("cached_parse_duration_s") or 0.0)
                    for value in artifact_values
                ),
                2,
            ),
            "cached_indexing_duration_total_s": round(
                sum(
                    float(value.get("cached_indexing_duration_s") or 0.0)
                    for value in artifact_values
                ),
                2,
            ),
            "multimodal_items": None,
            "multimodal_items_attempted": None,
            "multimodal_chunks_added": None,
            "total_chunks": None,
            "docling_parse_issue_count": None,
            "text_extraction_invalid_outputs": None,
            "text_extraction_complete_zero_relation_chunks": None,
            "text_extraction_complete_zero_entity_chunks": None,
        },
        "document_summaries": document_summaries,
    }
    for k in K_VALUES:
        summary[f"hit_at_{k}"] = (
            sum(bool(result["metrics"][f"hit_at_{k}"]) for result in results)
            / question_count
            if question_count
            else 0.0
        )
        summary[f"evidence_page_recall_at_{k}"] = (
            sum(
                result["metrics"][f"evidence_page_recall_at_{k}"]
                for result in results
            )
            / question_count
            if question_count
            else 0.0
        )
    return summary


def main() -> None:
    load_dotenv()
    questions = _load_questions(PDF_INDEX_RANGE)
    embedding, embed_device = _build_embedding()
    results: list[dict[str, Any]] = []
    for doc_id, group in questions.groupby("doc_id", sort=False):
        results.extend(_evaluate_document(doc_id, group, embedding))

    OUTPUT_DIR.mkdir(exist_ok=True)
    details_path = OUTPUT_DIR / f"{OUTPUT_STEM}.jsonl"
    readable_details_path = OUTPUT_DIR / f"{OUTPUT_STEM}-readable.json"
    summary_path = OUTPUT_DIR / f"{OUTPUT_STEM}-summary.json"

    with details_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    readable_details_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = _summarize(results, embed_device)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[done] retrieval details -> {details_path}", flush=True)
    print(f"[done] readable details -> {readable_details_path}", flush=True)
    print(f"[done] retrieval summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
