"""Evaluate hybrid retrieval against MMLongBench evidence pages.

Usage:
    uv run python evaluate_retrieval_pages.py

This script requires a page-aware multimodal attempt produced by:
    uv run python smoke_text_checkpoint.py
    uv run python smoke_multimodal_from_checkpoint.py

It uses LightRAG's structured retrieval API and never asks the VLM to
generate a final answer from the recovered images.
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from lightrag.base import QueryParam

from hello_raganything import _load_questions
from local_config import PDF_INDEX_RANGE, RETRIEVAL_EVAL
from retrieval_provenance import PAGE_PROVENANCE
from smoke_multimodal_from_checkpoint import (
    ATTEMPT_ARCH_NAME,
    _attempt_working_dir,
    _build_rag,
)


K_VALUES = (1, 3, 5, 10, 20)


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, list) else [parsed]


def _validate_attempt(doc_id: str) -> tuple[Path, dict[str, Any]]:
    attempt_dir = _attempt_working_dir(doc_id)
    manifest_path = attempt_dir / "multimodal_attempt_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing page-aware multimodal attempt for {doc_id}. Run "
            "`uv run python smoke_text_checkpoint.py` followed by "
            "`uv run python smoke_multimodal_from_checkpoint.py` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("parser") != RETRIEVAL_EVAL.parser_name
        or manifest.get("chunk_token_size") != RETRIEVAL_EVAL.chunk_token_size
        or manifest.get("page_provenance") != PAGE_PROVENANCE
    ):
        raise RuntimeError(
            f"Attempt for {doc_id} does not use deterministic page provenance."
        )
    extraction_quality = manifest.get("text_extraction_quality", {})
    if (
        extraction_quality.get("invalid_outputs") != 0
        or extraction_quality.get("validated_outputs") != manifest.get("text_chunks")
    ):
        raise RuntimeError(
            f"Attempt for {doc_id} was based on an unvalidated textual "
            "checkpoint. Rebuild the text checkpoint and multimodal attempt "
            "before final retrieval evaluation."
        )
    return attempt_dir, manifest


async def _ranked_chunks_with_pages(rag, retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    result_chunks = retrieval.get("data", {}).get("chunks", [])
    ranked_chunks: list[dict[str, Any]] = []
    for rank, chunk in enumerate(result_chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        stored = await rag.lightrag.text_chunks.get_by_id(chunk_id)
        pages = stored.get("page_numbers") if stored else None
        if not pages:
            raise RuntimeError(f"Retrieved chunk {chunk_id} has no page_numbers")
        ranked_chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "page_numbers": list(pages),
                "content_type": stored.get("content_type", "unknown"),
                "content_preview": chunk.get("content", "")[:180],
            }
        )
    return ranked_chunks


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
                    "content_type": chunk["content_type"],
                    "chunk_id": chunk["chunk_id"],
                    "matches_evidence": page_number in evidence,
                }
            )
    return pages


async def _evaluate_document(doc_id: str, group) -> list[dict[str, Any]]:
    attempt_dir, manifest = _validate_attempt(doc_id)
    rag = _build_rag(str(attempt_dir))
    results: list[dict[str, Any]] = []
    try:
        initialized = await rag._ensure_lightrag_initialized()
        if not initialized or not initialized.get("success"):
            raise RuntimeError(f"LightRAG initialization failed: {initialized}")

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
                retrieval = await rag.lightrag.aquery_data(
                    row["question"],
                    QueryParam(mode="hybrid", enable_rerank=False),
                )
                if retrieval.get("status") != "success":
                    raise RuntimeError(f"Retrieval did not succeed: {retrieval}")
                ranked_chunks = await _ranked_chunks_with_pages(rag, retrieval)
                evidence_set = set(evidence_pages)
                for chunk in ranked_chunks:
                    chunk["matches_evidence"] = bool(
                        evidence_set.intersection(chunk["page_numbers"])
                    )
                ranked_pages = _ranked_pages(ranked_chunks, evidence_pages)
                metrics = _metrics_for_question(ranked_chunks, evidence_pages)
                retrieval_metadata = retrieval.get("metadata", {})
                error = None
            except Exception as exc:
                ranked_chunks = []
                ranked_pages = []
                metrics = _metrics_for_question([], evidence_pages)
                retrieval_metadata = {}
                error = repr(exc)
            results.append(
                {
                    "arch": ATTEMPT_ARCH_NAME,
                    "doc_id": doc_id,
                    "question": row["question"],
                    "evidence_pages": evidence_pages,
                    "evidence_sources": evidence_sources,
                    "parser": manifest["parser"],
                    "chunk_token_size": manifest["chunk_token_size"],
                    "ranked_chunks": ranked_chunks,
                    "ranked_pages": ranked_pages,
                    "retrieval_metadata": retrieval_metadata,
                    "metrics": metrics,
                    "duration_s": round(time.perf_counter() - started, 2),
                    "error": error,
                }
            )
    finally:
        await rag.finalize_storages()
    return results


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    question_count = len(results)
    summary: dict[str, Any] = {
        "arch": ATTEMPT_ARCH_NAME,
        "questions": question_count,
        "failed_queries": sum(1 for result in results if result["error"]),
        "parser": RETRIEVAL_EVAL.parser_name,
        "chunk_token_size": RETRIEVAL_EVAL.chunk_token_size,
        "page_provenance": PAGE_PROVENANCE,
        "mrr": (
            sum(result["metrics"]["mrr"] for result in results) / question_count
            if question_count
            else 0.0
        ),
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


async def main() -> None:
    questions = _load_questions(PDF_INDEX_RANGE)
    results: list[dict[str, Any]] = []
    for doc_id, group in questions.groupby("doc_id", sort=False):
        results.extend(await _evaluate_document(doc_id, group))

    output_dir = Path("./smoke_results")
    output_dir.mkdir(exist_ok=True)
    details_path = output_dir / f"{ATTEMPT_ARCH_NAME}-retrieval-pages.jsonl"
    readable_details_path = (
        output_dir / f"{ATTEMPT_ARCH_NAME}-retrieval-pages-readable.json"
    )
    summary_path = output_dir / f"{ATTEMPT_ARCH_NAME}-retrieval-pages-summary.json"
    with details_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    readable_details_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = _summarize(results)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[done] retrieval details -> {details_path}", flush=True)
    print(f"[done] readable details -> {readable_details_path}", flush=True)
    print(f"[done] retrieval summary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
