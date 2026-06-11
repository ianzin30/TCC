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
import statistics
import time
from pathlib import Path
from typing import Any

from lightrag.base import QueryParam

from hello_raganything import _load_questions
from local_config import (
    PDF_INDEX_RANGE,
    RERANK_EVAL,
    RETRIEVAL_EVAL,
    build_lightrag_workspace,
    build_parser_kwargs,
    ensure_reranker_available,
    model_manifest_fields,
    rerank_manifest_fields,
    rerank_model_func,
    require_current_model_manifest,
)
from retrieval_provenance import PAGE_PROVENANCE
from smoke_multimodal_from_checkpoint import (
    ATTEMPT_ARCH_NAME,
    _attempt_working_dir,
    _build_rag,
    validate_lightrag_checkpoint_storage,
)


K_VALUES = (1, 3, 5, 10, 20)
OUTPUT_STEM = (
    f"{ATTEMPT_ARCH_NAME}-{RERANK_EVAL.output_suffix}-retrieval-pages"
    if RERANK_EVAL.enabled
    else f"{ATTEMPT_ARCH_NAME}-retrieval-pages"
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
    require_current_model_manifest(manifest, "Multimodal attempt")
    if (
        manifest.get("parser") != RETRIEVAL_EVAL.parser_name
        or manifest.get("parser_options") != build_parser_kwargs()
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
    if manifest.get("lightrag_workspace") != build_lightrag_workspace(str(attempt_dir)):
        raise RuntimeError(
            f"Attempt for {doc_id} does not declare the expected isolated "
            "LightRAG workspace. Rebuild text and multimodal checkpoints."
        )
    if int(manifest.get("total_chunks") or 0) <= 0:
        raise RuntimeError(
            f"Attempt for {doc_id} contains no total_chunks. Rebuild the "
            "text checkpoint and multimodal attempt."
        )
    if int(manifest.get("multimodal_chunks_added") or 0) < 0:
        raise RuntimeError(
            f"Attempt for {doc_id} has negative multimodal_chunks_added. "
            "This indicates contaminated storage; rebuild the checkpoint."
        )
    validate_lightrag_checkpoint_storage(
        attempt_dir,
        manifest,
        expected_chunks=int(manifest["total_chunks"]),
        expected_status="processed",
        require_multimodal_processed=True,
        artifact_name=f"Multimodal attempt for {doc_id}",
    )
    return attempt_dir, manifest


def _document_artifacts_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    text_quality = manifest.get("text_extraction_quality", {})
    docling_issues = manifest.get("docling_parse_issues", {})
    text_duration = _optional_float(manifest.get("text_processing_duration_s"))
    multimodal_duration = _optional_float(
        manifest.get("multimodal_processing_duration_s")
    )
    total_duration = _optional_float(manifest.get("total_processing_duration_s"))
    processing_note = None
    if total_duration is None:
        processing_note = (
            "Processing durations were not recorded in this manifest; rebuild "
            "the text checkpoint and multimodal attempt to capture them."
        )
    return {
        "doc_id": manifest.get("doc_id"),
        "text_characters": manifest.get("text_characters"),
        "text_chunks": manifest.get("text_chunks"),
        "multimodal_items": manifest.get("multimodal_items"),
        "multimodal_items_attempted": manifest.get("multimodal_items_attempted"),
        "multimodal_chunks_added": manifest.get("multimodal_chunks_added"),
        "total_chunks": manifest.get("total_chunks"),
        "lightrag_workspace": manifest.get("lightrag_workspace"),
        "docling_parse_issue_count": docling_issues.get("issue_count", 0),
        "docling_parse_issue_patterns": docling_issues.get("matched_patterns", []),
        "text_extraction_validated_outputs": text_quality.get("validated_outputs"),
        "text_extraction_invalid_outputs": text_quality.get("invalid_outputs"),
        "text_extraction_complete_zero_relation_chunks": len(
            text_quality.get("complete_zero_relation_chunks", [])
        ),
        "text_extraction_complete_zero_entity_chunks": len(
            text_quality.get("complete_zero_entity_chunks", [])
        ),
        "text_processing_started_at_utc": manifest.get(
            "text_processing_started_at_utc"
        ),
        "text_processing_finished_at_utc": manifest.get(
            "text_processing_finished_at_utc"
        ),
        "text_processing_duration_s": text_duration,
        "multimodal_processing_started_at_utc": manifest.get(
            "multimodal_processing_started_at_utc"
        ),
        "multimodal_processing_finished_at_utc": manifest.get(
            "multimodal_processing_finished_at_utc"
        ),
        "multimodal_processing_duration_s": multimodal_duration,
        "total_processing_started_at_utc": manifest.get(
            "total_processing_started_at_utc"
        ),
        "total_processing_finished_at_utc": manifest.get(
            "total_processing_finished_at_utc"
        ),
        "total_processing_duration_s": total_duration,
        "processing_duration_note": processing_note,
    }


async def _ranked_chunks_with_pages(rag, retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    result_chunks = retrieval.get("data", {}).get("chunks", [])
    ranked_chunks: list[dict[str, Any]] = []
    for rank, chunk in enumerate(result_chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        stored = await rag.lightrag.text_chunks.get_by_id(chunk_id)
        pages = stored.get("page_numbers") if stored else None
        if not pages:
            raise RuntimeError(f"Retrieved chunk {chunk_id} has no page_numbers")
        ranked_chunk = {
            "rank": rank,
            "chunk_id": chunk_id,
            "page_numbers": list(pages),
            "content_type": stored.get("content_type", "unknown"),
            "content_preview": chunk.get("content", "")[:180],
        }
        if "rerank_score" in chunk:
            ranked_chunk["rerank_score"] = float(chunk["rerank_score"])
        if "rerank_original_rank" in chunk:
            ranked_chunk["rerank_original_rank"] = int(chunk["rerank_original_rank"])
        ranked_chunks.append(ranked_chunk)
    return ranked_chunks


async def _rerank_retrieved_chunks(
    query: str, result_chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rerank_metadata = {
        **rerank_manifest_fields(),
        "lightrag_internal_rerank_enabled": False,
        "rerank_input_chunks": len(result_chunks),
        "rerank_output_chunks": len(result_chunks),
    }
    if not RERANK_EVAL.enabled:
        return result_chunks, rerank_metadata

    documents = [chunk.get("content", "") for chunk in result_chunks]
    rerank_results = await rerank_model_func(
        query=query,
        documents=documents,
        top_n=RERANK_EVAL.top_n,
    )
    if not rerank_results:
        raise RuntimeError("Reranker returned no results")

    reranked_chunks: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for result in rerank_results:
        index = int(result["index"])
        if index < 0 or index >= len(result_chunks):
            raise RuntimeError(f"Reranker returned out-of-range index {index}")
        if index in seen_indexes:
            raise RuntimeError(f"Reranker returned duplicate index {index}")
        seen_indexes.add(index)

        chunk = result_chunks[index].copy()
        chunk["rerank_score"] = float(result["relevance_score"])
        chunk["rerank_original_rank"] = index + 1
        reranked_chunks.append(chunk)

    rerank_metadata["rerank_output_chunks"] = len(reranked_chunks)
    return reranked_chunks, rerank_metadata


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
                    **(
                        {"first_chunk_rerank_score": chunk["rerank_score"]}
                        if "rerank_score" in chunk
                        else {}
                    ),
                }
            )
    return pages


async def _evaluate_document(doc_id: str, group) -> list[dict[str, Any]]:
    attempt_dir, manifest = _validate_attempt(doc_id)
    rag = _build_rag(str(attempt_dir), enable_reranker=RERANK_EVAL.enabled)
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
                    QueryParam(
                        mode="hybrid",
                        chunk_top_k=RERANK_EVAL.top_n,
                        enable_rerank=False,
                    ),
                )
                if retrieval.get("status") != "success":
                    raise RuntimeError(f"Retrieval did not succeed: {retrieval}")
                result_chunks, rerank_metadata = await _rerank_retrieved_chunks(
                    row["question"], retrieval.get("data", {}).get("chunks", [])
                )
                retrieval = {
                    **retrieval,
                    "data": {**retrieval.get("data", {}), "chunks": result_chunks},
                }
                ranked_chunks = await _ranked_chunks_with_pages(rag, retrieval)
                if not ranked_chunks:
                    raise RuntimeError(
                        "Retrieval succeeded but returned no chunks; storage or "
                        "retrieval metadata is not usable for page evaluation."
                    )
                evidence_set = set(evidence_pages)
                for chunk in ranked_chunks:
                    chunk["matches_evidence"] = bool(
                        evidence_set.intersection(chunk["page_numbers"])
                    )
                ranked_pages = _ranked_pages(ranked_chunks, evidence_pages)
                metrics = _metrics_for_question(ranked_chunks, evidence_pages)
                retrieval_metadata = {
                    **retrieval.get("metadata", {}),
                    **rerank_metadata,
                }
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
                    "text_llm_model": manifest["text_llm_model"],
                    "vision_llm_model": manifest["vision_llm_model"],
                    **rerank_manifest_fields(),
                    "document_artifacts": _document_artifacts_from_manifest(manifest),
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


def _summarize_by_document(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_doc.setdefault(result["doc_id"], []).append(result)

    document_summaries: list[dict[str, Any]] = []
    for doc_id, doc_results in by_doc.items():
        question_count = len(doc_results)
        durations = [float(result["duration_s"]) for result in doc_results]
        document_artifacts = doc_results[0].get("document_artifacts", {})
        first_hit_ranks = [
            result["metrics"]["first_hit_rank"]
            for result in doc_results
            if result["metrics"]["first_hit_rank"] is not None
        ]
        summary: dict[str, Any] = {
            "doc_id": doc_id,
            "questions": question_count,
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
            "text_processing_duration_s": document_artifacts.get(
                "text_processing_duration_s"
            ),
            "multimodal_processing_duration_s": document_artifacts.get(
                "multimodal_processing_duration_s"
            ),
            "total_processing_duration_s": document_artifacts.get(
                "total_processing_duration_s"
            ),
            "document_artifacts": document_artifacts,
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
        document_summaries.append(summary)
    return document_summaries


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    question_count = len(results)
    document_summaries = _summarize_by_document(results)
    query_durations = [float(result["duration_s"]) for result in results]
    first_hit_ranks = [
        result["metrics"]["first_hit_rank"]
        for result in results
        if result["metrics"]["first_hit_rank"] is not None
    ]
    artifact_values = [
        summary["document_artifacts"] for summary in document_summaries
    ]
    text_processing_durations = [
        value["text_processing_duration_s"]
        for value in artifact_values
        if value.get("text_processing_duration_s") is not None
    ]
    multimodal_processing_durations = [
        value["multimodal_processing_duration_s"]
        for value in artifact_values
        if value.get("multimodal_processing_duration_s") is not None
    ]
    total_processing_durations = [
        value["total_processing_duration_s"]
        for value in artifact_values
        if value.get("total_processing_duration_s") is not None
    ]
    processing_duration_available = len(total_processing_durations) == len(
        artifact_values
    )
    summary: dict[str, Any] = {
        **model_manifest_fields(),
        **rerank_manifest_fields(),
        "arch": ATTEMPT_ARCH_NAME,
        "pdf_index_range": list(PDF_INDEX_RANGE),
        "documents": len(document_summaries),
        "questions": question_count,
        "failed_queries": sum(1 for result in results if result["error"]),
        "parser": RETRIEVAL_EVAL.parser_name,
        "chunk_token_size": RETRIEVAL_EVAL.chunk_token_size,
        "page_provenance": PAGE_PROVENANCE,
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
        "processing_duration_available": processing_duration_available,
        "processing_duration_note": (
            None
            if processing_duration_available
            else "Some manifests do not contain processing durations. Rebuild "
            "text checkpoints and multimodal attempts to capture complete timing."
        ),
        "text_processing_duration_total_s": round(
            sum(text_processing_durations), 2
        ),
        "text_processing_duration_mean_s": round(
            _mean(text_processing_durations), 2
        ),
        "multimodal_processing_duration_total_s": round(
            sum(multimodal_processing_durations), 2
        ),
        "multimodal_processing_duration_mean_s": round(
            _mean(multimodal_processing_durations), 2
        ),
        "total_processing_duration_total_s": round(
            sum(total_processing_durations), 2
        ),
        "total_processing_duration_mean_s": round(
            _mean(total_processing_durations), 2
        ),
        "artifact_totals": {
            "text_characters": sum(
                int(value.get("text_characters") or 0) for value in artifact_values
            ),
            "text_chunks": sum(
                int(value.get("text_chunks") or 0) for value in artifact_values
            ),
            "multimodal_items": sum(
                int(value.get("multimodal_items") or 0) for value in artifact_values
            ),
            "multimodal_items_attempted": sum(
                int(value.get("multimodal_items_attempted") or 0)
                for value in artifact_values
            ),
            "multimodal_chunks_added": sum(
                int(value.get("multimodal_chunks_added") or 0)
                for value in artifact_values
            ),
            "total_chunks": sum(
                int(value.get("total_chunks") or 0) for value in artifact_values
            ),
            "docling_parse_issue_count": sum(
                int(value.get("docling_parse_issue_count") or 0)
                for value in artifact_values
            ),
            "text_extraction_invalid_outputs": sum(
                int(value.get("text_extraction_invalid_outputs") or 0)
                for value in artifact_values
            ),
            "text_extraction_complete_zero_relation_chunks": sum(
                int(value.get("text_extraction_complete_zero_relation_chunks") or 0)
                for value in artifact_values
            ),
            "text_extraction_complete_zero_entity_chunks": sum(
                int(value.get("text_extraction_complete_zero_entity_chunks") or 0)
                for value in artifact_values
            ),
            "text_processing_duration_total_s": round(
                sum(text_processing_durations), 2
            ),
            "multimodal_processing_duration_total_s": round(
                sum(multimodal_processing_durations), 2
            ),
            "total_processing_duration_total_s": round(
                sum(total_processing_durations), 2
            ),
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


async def main() -> None:
    ensure_reranker_available()
    questions = _load_questions(PDF_INDEX_RANGE)
    results: list[dict[str, Any]] = []
    for doc_id, group in questions.groupby("doc_id", sort=False):
        results.extend(await _evaluate_document(doc_id, group))

    output_dir = Path("./smoke_results")
    output_dir.mkdir(exist_ok=True)
    details_path = output_dir / f"{OUTPUT_STEM}.jsonl"
    readable_details_path = output_dir / f"{OUTPUT_STEM}-readable.json"
    summary_path = output_dir / f"{OUTPUT_STEM}-summary.json"
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
