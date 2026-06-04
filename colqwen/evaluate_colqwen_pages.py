"""Evaluate ColQwen3.5 page retrieval against MMLongBench evidence pages.

Usage:
    uv run python evaluate_colqwen_pages.py

The evaluator renders each PDF page, embeds pages with ColQwen3.5, ranks
pages for each question, and compares the ranking with MMLongBench
``evidence_pages``. It evaluates retrieval only; it does not generate final
answers.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import torch
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor


MODEL_NAME = "athrael-soju/colqwen3.5-4.5B-v3"
ARCH_NAME = "colqwen3_5-4.5B-v3"
OUTPUT_STEM = "colqwen3_5-retrieval-pages"
PARSER_NAME = "pdf_page_rendering"
PAGE_PROVENANCE = "rendered_pdf_page"
MODEL_LOAD_STRATEGY = "cpu_staged_to_selected_device"

PDF_INDEX_RANGE: tuple[int, int] = (1, 5)
UNANSWERABLE_MARKER = "Not answerable"
K_VALUES = (1, 3, 5, 10, 20)

RENDER_DPI = 144
PAGE_IMAGE_BATCH_SIZE = 1
QUERY_BATCH_SIZE = 1

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MMLONGBENCH_PARQUET = (
    PROJECT_ROOT / "MMLongBench-Doc" / "data" / "train-00000-of-00001.parquet"
)
MMLONGBENCH_PDFS_DIR = PROJECT_ROOT / "MMLongBench-Doc" / "documents"
OUTPUT_DIR = SCRIPT_DIR / "smoke_results"
CACHE_DIR = SCRIPT_DIR / "cache" / "colqwen3_5"
RENDER_CACHE_DIR = CACHE_DIR / "rendered_pages"
EMBEDDING_CACHE_DIR = CACHE_DIR / "page_embeddings"


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


def _safe_model_name() -> str:
    return MODEL_NAME.replace("/", "__").replace(":", "_")


def _render_cache_dir(doc_id: str) -> Path:
    return RENDER_CACHE_DIR / Path(doc_id).stem


def _embedding_cache_path(doc_id: str) -> Path:
    filename = f"{_safe_model_name()}_dpi{RENDER_DPI}.pt"
    return EMBEDDING_CACHE_DIR / Path(doc_id).stem / filename


def _render_pdf_pages(pdf_path: Path, doc_id: str) -> tuple[list[Path], dict[str, Any]]:
    started = time.perf_counter()
    output_dir = _render_cache_dir(doc_id)
    manifest_path = output_dir / "render_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        page_paths = [output_dir / page["filename"] for page in manifest["pages"]]
        if (
            manifest.get("pdf_path") == str(pdf_path)
            and manifest.get("render_dpi") == RENDER_DPI
            and all(path.exists() for path in page_paths)
        ):
            manifest["cache_hit"] = True
            manifest["duration_s"] = round(time.perf_counter() - started, 2)
            return page_paths, manifest

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_paths: list[Path] = []
    page_entries: list[dict[str, Any]] = []
    zoom = RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            page_path = output_dir / f"page_{page_index:04d}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(str(page_path))
            page_paths.append(page_path)
            page_entries.append(
                {
                    "page_number": page_index,
                    "filename": page_path.name,
                    "width": pixmap.width,
                    "height": pixmap.height,
                }
            )

    manifest = {
        "pdf_path": str(pdf_path),
        "doc_id": doc_id,
        "render_dpi": RENDER_DPI,
        "page_count": len(page_paths),
        "pages": page_entries,
        "cache_hit": False,
        "duration_s": round(time.perf_counter() - started, 2),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return page_paths, manifest


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    print("Warning: CPU will probably be very slow.", flush=True)
    return "cpu", torch.float32


def _load_model() -> tuple[ColQwen3_5, ColQwen3_5Processor, str, torch.dtype]:
    device, dtype = _device_and_dtype()
    # Avoid a Windows access violation in Transformers async weight materialization.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    load_device = "cpu" if device != "cpu" else device
    model = ColQwen3_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=load_device,
        attn_implementation="sdpa",
    ).eval()
    if load_device != device:
        model = model.to(device)
    processor = ColQwen3_5Processor.from_pretrained(MODEL_NAME)
    return model, processor, device, dtype


def _load_or_encode_page_embeddings(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    doc_id: str,
    page_paths: list[Path],
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    started = time.perf_counter()
    cache_path = _embedding_cache_path(doc_id)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            cached.get("model_name") == MODEL_NAME
            and cached.get("render_dpi") == RENDER_DPI
            and cached.get("page_count") == len(page_paths)
        ):
            metadata = dict(cached.get("metadata", {}))
            metadata.update(
                {
                    "cache_hit": True,
                    "duration_s": round(time.perf_counter() - started, 2),
                    "cache_path": str(cache_path),
                }
            )
            cached_embeddings = cached["embeddings"]
            if isinstance(cached_embeddings, torch.Tensor):
                cached_embeddings = list(torch.unbind(cached_embeddings))
            return cached_embeddings, metadata

    embeddings: list[torch.Tensor] = []

    for start in range(0, len(page_paths), PAGE_IMAGE_BATCH_SIZE):
        batch_paths = page_paths[start : start + PAGE_IMAGE_BATCH_SIZE]
        images = [_load_image(path) for path in batch_paths]
        batch = processor.process_images(images).to(model.device)

        with torch.inference_mode():
            batch_embeddings = model(**batch)

        embeddings.extend(list(torch.unbind(batch_embeddings.detach().cpu())))
        print(
            f"[index] {doc_id}: embedded pages "
            f"{start + 1}-{start + len(batch_paths)}/{len(page_paths)}",
            flush=True,
        )

    metadata = {
        "model_name": MODEL_NAME,
        "render_dpi": RENDER_DPI,
        "page_count": len(page_paths),
        "page_image_batch_size": PAGE_IMAGE_BATCH_SIZE,
        "embedding_storage": "list[torch.Tensor]",
        "cache_hit": False,
        "duration_s": round(time.perf_counter() - started, 2),
        "cache_path": str(cache_path),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "render_dpi": RENDER_DPI,
            "page_count": len(page_paths),
            "embedding_storage": "list[torch.Tensor]",
            "embeddings": embeddings,
            "metadata": metadata,
        },
        cache_path,
    )

    return embeddings, metadata


def _encode_queries(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    queries: list[str],
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    for start in range(0, len(queries), QUERY_BATCH_SIZE):
        batch_queries = queries[start : start + QUERY_BATCH_SIZE]
        batch = processor.process_queries(batch_queries).to(model.device)
        with torch.inference_mode():
            model.rope_deltas = None
            batch_embeddings = model(**batch)
        embeddings.append(batch_embeddings.detach())
    return torch.cat(embeddings, dim=0)


def _score_pages(
    processor: ColQwen3_5Processor,
    query_embeddings: torch.Tensor,
    page_embeddings: list[torch.Tensor],
) -> tuple[torch.Tensor, str]:
    if hasattr(processor, "score"):
        return processor.score(query_embeddings, page_embeddings)[0], "processor.score"
    return (
        processor.score_multi_vector(query_embeddings, page_embeddings)[0],
        "processor.score_multi_vector",
    )


def _ranked_pages_from_scores(
    scores: torch.Tensor, evidence_pages: list[int]
) -> list[dict[str, Any]]:
    evidence = set(evidence_pages)
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    ranked_pages: list[dict[str, Any]] = []
    for rank, page_index in enumerate(ranked_indices, start=1):
        page_number = int(page_index) + 1
        ranked_pages.append(
            {
                "rank": rank,
                "page_number": page_number,
                "score": float(scores[page_index].item()),
                "matches_evidence": page_number in evidence,
            }
        )
    return ranked_pages


def _metrics_for_question(
    ranked_pages: list[dict[str, Any]], evidence_pages: list[int]
) -> dict[str, Any]:
    evidence = set(evidence_pages)
    first_hit_rank: int | None = None
    for page in ranked_pages:
        if page["page_number"] in evidence:
            first_hit_rank = page["rank"]
            break

    metrics: dict[str, Any] = {
        "first_hit_rank": first_hit_rank,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "retrieved_pages": len(ranked_pages),
        "evidence_pages_count": len(evidence),
    }
    for k in K_VALUES:
        pages_at_k = {page["page_number"] for page in ranked_pages[:k]}
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
        "model_name": MODEL_NAME,
        "doc_id": doc_id,
        "question": row["question"],
        "evidence_pages": evidence_pages,
        "evidence_sources": _parse_list(row.get("evidence_sources")),
        "parser": PARSER_NAME,
        "chunk_token_size": None,
        "text_llm_model": None,
        "vision_llm_model": MODEL_NAME,
        "document_artifacts": {},
        "ranked_chunks": [],
        "ranked_pages": [],
        "retrieval_metadata": {},
        "metrics": _metrics_for_question([], evidence_pages),
        "duration_s": 0.0,
        "error": error,
    }


def _evaluate_document(
    model: ColQwen3_5,
    processor: ColQwen3_5Processor,
    doc_id: str,
    group: pd.DataFrame,
) -> list[dict[str, Any]]:
    pdf_path = _resolve_pdf_path(doc_id)
    document_started = time.perf_counter()
    try:
        page_paths, render_metadata = _render_pdf_pages(pdf_path, doc_id)
        page_embeddings, embedding_metadata = _load_or_encode_page_embeddings(
            model, processor, doc_id, page_paths
        )
    except Exception as exc:
        error = repr(exc)
        return [_failed_result(doc_id, row, error) for _, row in group.iterrows()]

    page_embeddings = [embedding.to(model.device) for embedding in page_embeddings]
    indexing_duration_s = round(time.perf_counter() - document_started, 2)
    document_artifacts = {
        "pdf_path": str(pdf_path),
        "page_count": len(page_paths),
        "render_dpi": RENDER_DPI,
        "page_image_batch_size": PAGE_IMAGE_BATCH_SIZE,
        "render_cache_hit": render_metadata.get("cache_hit", False),
        "render_duration_s": render_metadata.get("duration_s"),
        "embedding_cache_hit": embedding_metadata.get("cache_hit", False),
        "embedding_duration_s": embedding_metadata.get("duration_s"),
        "indexing_duration_s": indexing_duration_s,
        "embedding_cache_path": embedding_metadata.get("cache_path"),
    }

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
            query_embeddings = _encode_queries(model, processor, [row["question"]])
            scores, score_function = _score_pages(
                processor, query_embeddings, page_embeddings
            )
            ranked_pages = _ranked_pages_from_scores(scores.detach().cpu(), evidence_pages)
            metrics = _metrics_for_question(ranked_pages, evidence_pages)
            error = None
        except Exception as exc:
            ranked_pages = []
            metrics = _metrics_for_question([], evidence_pages)
            score_function = None
            error = repr(exc)

        results.append(
            {
                "arch": ARCH_NAME,
                "model_name": MODEL_NAME,
                "doc_id": doc_id,
                "question": row["question"],
                "evidence_pages": evidence_pages,
                "evidence_sources": evidence_sources,
                "parser": PARSER_NAME,
                "chunk_token_size": None,
                "text_llm_model": None,
                "vision_llm_model": MODEL_NAME,
                "ranked_pages": ranked_pages,
                "ranked_chunks": [],
                "retrieval_metadata": {
                    "retrieval_unit": "page",
                    "page_count": len(page_paths),
                    "render_dpi": RENDER_DPI,
                    "score_function": score_function,
                },
                "metrics": metrics,
                "duration_s": round(time.perf_counter() - started, 2),
                "document_artifacts": document_artifacts,
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


def _summarize(
    results: list[dict[str, Any]], device: str, dtype: torch.dtype
) -> dict[str, Any]:
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
        "model_name": MODEL_NAME,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "pdf_index_range": list(PDF_INDEX_RANGE),
        "render_dpi": RENDER_DPI,
        "page_image_batch_size": PAGE_IMAGE_BATCH_SIZE,
        "query_batch_size": QUERY_BATCH_SIZE,
        "documents": len(document_summaries),
        "questions": question_count,
        "failed_queries": sum(1 for result in results if result["error"]),
        "parser": PARSER_NAME,
        "chunk_token_size": None,
        "page_provenance": PAGE_PROVENANCE,
        "text_llm_model": None,
        "vision_llm_model": MODEL_NAME,
        "text_llm_options": None,
        "vision_llm_options": {
            "render_dpi": RENDER_DPI,
            "page_image_batch_size": PAGE_IMAGE_BATCH_SIZE,
            "query_batch_size": QUERY_BATCH_SIZE,
        },
        "text_llm_think": None,
        "vision_llm_think": None,
        "gpu_offload_policy": MODEL_LOAD_STRATEGY,
        "page_count_total": sum(
            int(value.get("page_count") or 0) for value in artifact_values
        ),
        "render_duration_total_s": round(
            sum(float(value.get("render_duration_s") or 0.0) for value in artifact_values),
            2,
        ),
        "embedding_duration_total_s": round(
            sum(
                float(value.get("embedding_duration_s") or 0.0)
                for value in artifact_values
            ),
            2,
        ),
        "indexing_duration_total_s": round(
            sum(
                float(value.get("indexing_duration_s") or 0.0)
                for value in artifact_values
            ),
            2,
        ),
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
            "ColQwen indexing duration includes PDF page rendering plus page "
            "embedding, and query duration measures query embedding plus page scoring."
        ),
        "artifact_totals": {
            "page_count": sum(
                int(value.get("page_count") or 0) for value in artifact_values
            ),
            "text_characters": None,
            "text_chunks": None,
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
    questions = _load_questions(PDF_INDEX_RANGE)
    model, processor, device, dtype = _load_model()
    results: list[dict[str, Any]] = []
    for doc_id, group in questions.groupby("doc_id", sort=False):
        results.extend(_evaluate_document(model, processor, doc_id, group))

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
    summary = _summarize(results, device, dtype)
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
