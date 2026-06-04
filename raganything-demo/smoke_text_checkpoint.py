"""Build a reusable text-only checkpoint for the MMLongBench smoke document.

Usage:
    uv run python smoke_text_checkpoint.py

This intentionally stops before multimodal processing. A later run of
``smoke_multimodal_from_checkpoint.py`` copies this checkpoint and performs
the expensive/fragile image and table stage without repeating text indexing.
"""

import asyncio
import logging
import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from raganything import RAGAnything
from raganything.base import DocStatus
from raganything.utils import insert_text_content, separate_content
from lightrag.utils import sanitize_text_for_encoding

from hello_raganything import ARCH_NAME, _load_questions, _resolve_pdf_path
from local_config import (
    EXTRACTION_QUALITY,
    ExtractionQualityStats,
    PARSER_OUTPUT_DIR,
    PDF_INDEX_RANGE,
    RETRIEVAL_EVAL,
    TEXT_LLM,
    VISION_LLM,
    analyze_extraction_output,
    build_embedding_func,
    build_lightrag_kwargs,
    build_lightrag_workspace,
    build_llm_func,
    build_parser_kwargs,
    build_rag_config,
    build_vision_func,
    model_manifest_fields,
)
from retrieval_provenance import (
    PAGE_PROVENANCE,
    build_page_aware_chunking_func,
    build_text_content_with_page_spans,
    register_docling_provenance_parser,
    validate_content_pages,
)


CHECKPOINT_ARCH_NAME = f"{ARCH_NAME}-text-checkpoint"
DOCLING_PARSE_ISSUE_LOG = Path(
    f"./rag_storage/{CHECKPOINT_ARCH_NAME}/docling_parse_issues.jsonl"
)
DOCLING_PARSE_ISSUE_PATTERNS = (
    "std::bad_alloc",
    "bad allocation",
    "memoryerror",
    "unable to allocate",
    "onnxruntimeerror",
)


def _checkpoint_working_dir(doc_id: str) -> str:
    return f"./rag_storage/{CHECKPOINT_ARCH_NAME}/{Path(doc_id).stem}"


class DoclingParseIssueCapture(logging.Handler):
    """Collect Docling/RapidOCR memory allocation warnings during one parse."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.events: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.exc_info:
            message = (
                message
                + "\n"
                + "".join(traceback.format_exception(*record.exc_info))
            )
        lower_message = message.lower()
        matches = [
            pattern
            for pattern in DOCLING_PARSE_ISSUE_PATTERNS
            if pattern in lower_message
        ]
        if not matches:
            return
        self.events.append(
            {
                "level": record.levelname,
                "logger": record.name,
                "matches": matches,
                "message": message,
            }
        )


def _summarize_docling_parse_issues(
    capture: DoclingParseIssueCapture, *, status: str, error: str | None = None
) -> dict:
    return {
        "status": status,
        "issue_count": len(capture.events),
        "matched_patterns": sorted(
            {match for event in capture.events for match in event["matches"]}
        ),
        "events": capture.events,
        "error": error,
    }


def _record_docling_parse_issues(doc_id: str, summary: dict) -> None:
    if summary["issue_count"] == 0:
        return
    DOCLING_PARSE_ISSUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "doc_id": doc_id,
        **summary,
    }
    with DOCLING_PARSE_ISSUE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(
        f"[docling issues] {doc_id}: captured {summary['issue_count']} "
        f"memory/allocation log event(s); see {DOCLING_PARSE_ISSUE_LOG}",
        flush=True,
    )


def _build_rag(working_dir: str) -> tuple[RAGAnything, ExtractionQualityStats]:
    register_docling_provenance_parser()
    quality_stats = ExtractionQualityStats()
    workspace = build_lightrag_workspace(working_dir)
    rag = RAGAnything(
        config=build_rag_config(
            working_dir=working_dir, parser_name=RETRIEVAL_EVAL.parser_name
        ),
        llm_model_func=build_llm_func(
            enforce_extraction_quality=True, quality_stats=quality_stats
        ),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(
            chunk_token_size=RETRIEVAL_EVAL.chunk_token_size,
            workspace=workspace,
        ),
    )
    return rag, quality_stats


async def _text_extraction_quality_summary(
    rag: RAGAnything, chunk_ids: list[str], quality_stats: ExtractionQualityStats
) -> dict:
    """Verify that cached text extractions were complete before checkpointing."""
    stored_chunks = await rag.lightrag.text_chunks.get_by_ids(chunk_ids)
    details: list[dict] = []
    for chunk_id, chunk in zip(chunk_ids, stored_chunks):
        cache_ids = (chunk or {}).get("llm_cache_list", [])
        cache_entries = await rag.lightrag.llm_response_cache.get_by_ids(cache_ids)
        extraction_entries = [
            entry
            for entry in cache_entries
            if entry and entry.get("cache_type") == "extract"
        ]
        if not extraction_entries:
            raise RuntimeError(f"Text chunk {chunk_id} has no extraction cache result")
        analysis = analyze_extraction_output(extraction_entries[0]["return"])
        details.append({"chunk_id": chunk_id, **analysis})

    empty_entities = [
        detail
        for detail in details
        if detail["issues"] == ["no valid entity records"]
        and not detail["malformed_records"]
    ]
    allowed_empty_entities = empty_entities[
        : EXTRACTION_QUALITY.max_empty_entity_chunks
    ]
    allowed_empty_chunk_ids = {
        detail["chunk_id"] for detail in allowed_empty_entities
    }
    invalid = [
        detail
        for detail in details
        if not detail["valid"] and detail["chunk_id"] not in allowed_empty_chunk_ids
    ]
    if invalid:
        raise RuntimeError(
            "Text extraction quality validation failed for chunks: "
            + ", ".join(detail["chunk_id"] for detail in invalid)
        )
    zero_relations = [
        detail["chunk_id"] for detail in details if detail["relation_records"] == 0
    ]
    print(
        f"[quality] {len(details)}/{len(details)} text extraction output(s) "
        "complete and structurally valid/allowed; complete outputs with "
        f"0 relations: {len(zero_relations)}; complete outputs with "
        f"0 entities: {len(empty_entities)}",
        flush=True,
    )
    return {
        "validated_outputs": len(details),
        "invalid_outputs": 0,
        "complete_zero_relation_chunks": zero_relations,
        "complete_zero_entity_chunks": [
            detail["chunk_id"] for detail in empty_entities
        ],
        "base_resources": {
            "num_ctx": EXTRACTION_QUALITY.base_num_ctx,
            "num_predict": EXTRACTION_QUALITY.base_num_predict,
        },
        "attempt_timeout_s": EXTRACTION_QUALITY.attempt_timeout_s,
        "retry_strategy": "high_budget_retry_after_confirmed_length",
        "elevated_retry_resources": (
            {
                "num_ctx": EXTRACTION_QUALITY.elevated_num_ctx,
                "num_predict": EXTRACTION_QUALITY.elevated_num_predict,
            }
            if EXTRACTION_QUALITY.max_elevated_retries
            else None
        ),
        "max_format_retries": EXTRACTION_QUALITY.max_format_retries,
        "max_dropped_malformed_records": (
            EXTRACTION_QUALITY.max_dropped_malformed_records
        ),
        "max_elevated_retries": EXTRACTION_QUALITY.max_elevated_retries,
        "max_empty_entity_chunks": EXTRACTION_QUALITY.max_empty_entity_chunks,
        **quality_stats.snapshot(),
    }


async def _build_document_checkpoint(doc_id: str, question_count: int) -> None:
    started_at_utc = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    pdf_path = _resolve_pdf_path(doc_id)
    working_dir = _checkpoint_working_dir(doc_id)
    checkpoint_path = Path(working_dir)

    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
        print(f"[clean] removed previous text checkpoint {working_dir}", flush=True)

    rag, quality_stats = _build_rag(working_dir)
    try:
        print(
            f"[models] text={TEXT_LLM.name}; vision={VISION_LLM.name}; "
            f"GPU placement={model_manifest_fields()['gpu_offload_policy']}",
            flush=True,
        )
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(f"LightRAG initialization failed: {init_result}")

        print(f"[text checkpoint] {doc_id} ({question_count} questions)", flush=True)
        docling_issue_capture = DoclingParseIssueCapture()
        root_logger = logging.getLogger()
        root_logger.addHandler(docling_issue_capture)
        try:
            content_list, content_doc_id = await rag.parse_document(
                str(pdf_path),
                PARSER_OUTPUT_DIR,
                "auto",
                True,
                **build_parser_kwargs(),
            )
        except Exception as e:
            docling_parse_issues = _summarize_docling_parse_issues(
                docling_issue_capture, status="parse_failed", error=str(e)
            )
            _record_docling_parse_issues(doc_id, docling_parse_issues)
            raise
        finally:
            root_logger.removeHandler(docling_issue_capture)
        docling_parse_issues = _summarize_docling_parse_issues(
            docling_issue_capture, status="parsed"
        )
        _record_docling_parse_issues(doc_id, docling_parse_issues)
        validate_content_pages(content_list)
        text_content, multimodal_items = separate_content(content_list)
        sanitized_text_content = sanitize_text_for_encoding(text_content)
        page_aware_text, page_spans = build_text_content_with_page_spans(
            content_list, sanitize_for_lightrag=True
        )
        if page_aware_text != sanitized_text_content:
            raise RuntimeError(
                "Page-aware text construction differs from LightRAG-sanitized text"
            )
        rag.lightrag.chunking_func = build_page_aware_chunking_func(
            page_aware_text, page_spans
        )
        file_ref = rag._get_file_reference(str(pdf_path))

        await insert_text_content(
            rag.lightrag,
            input=text_content,
            file_paths=file_ref,
            ids=content_doc_id,
        )
        status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        if status.get("status") == "failed":
            raise RuntimeError(
                "Text extraction failed before checkpoint creation: "
                f"{status.get('error_msg', 'no failure details recorded')}"
            )
        chunk_ids = status.get("chunks_list", [])
        stored_chunks = await rag.lightrag.text_chunks.get_by_ids(chunk_ids)
        missing_pages = [
            chunk_id
            for chunk_id, chunk in zip(chunk_ids, stored_chunks)
            if not chunk or not chunk.get("page_numbers")
        ]
        if missing_pages:
            raise RuntimeError(
                f"Text chunks missing deterministic page provenance: {missing_pages}"
            )
        extraction_quality = await _text_extraction_quality_summary(
            rag, chunk_ids, quality_stats
        )
        await rag._upsert_doc_status(
            content_doc_id,
            file_ref,
            status=DocStatus.HANDLING,
            error_msg="",
        )

        finished_at_utc = datetime.now(timezone.utc)
        text_processing_duration_s = round(time.perf_counter() - started_perf, 2)
        manifest = {
            **model_manifest_fields(),
            "doc_id": doc_id,
            "content_doc_id": content_doc_id,
            "pdf_path": str(pdf_path),
            "text_characters": len(text_content),
            "text_chunks": len(chunk_ids),
            "multimodal_items": len(multimodal_items),
            "working_dir": working_dir,
            "lightrag_workspace": build_lightrag_workspace(working_dir),
            "parser": RETRIEVAL_EVAL.parser_name,
            "parser_options": build_parser_kwargs(),
            "docling_parse_issues": docling_parse_issues,
            "chunk_token_size": RETRIEVAL_EVAL.chunk_token_size,
            "page_provenance": PAGE_PROVENANCE,
            "text_extraction_quality": extraction_quality,
            "text_processing_started_at_utc": started_at_utc.isoformat(),
            "text_processing_finished_at_utc": finished_at_utc.isoformat(),
            "text_processing_duration_s": text_processing_duration_s,
        }
        manifest_path = checkpoint_path / "text_checkpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[checkpoint ready] text graph saved at {working_dir}; "
            f"{len(multimodal_items)} multimodal items deferred; "
            f"text_processing_duration_s={text_processing_duration_s}",
            flush=True,
        )
    finally:
        await rag.finalize_storages()


async def main() -> None:
    questions = _load_questions(PDF_INDEX_RANGE)
    print(
        f"[grid] {questions['doc_id'].nunique()} PDF(s), "
        f"{len(questions)} answerable questions",
        flush=True,
    )
    for doc_id, group in questions.groupby("doc_id", sort=False):
        await _build_document_checkpoint(doc_id, len(group))


if __name__ == "__main__":
    asyncio.run(main())
