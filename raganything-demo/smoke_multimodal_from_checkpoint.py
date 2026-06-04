"""Process multimodal content from a saved text-only smoke-test checkpoint.

Usage:
    uv run python smoke_multimodal_from_checkpoint.py

Each invocation creates a fresh multimodal attempt copied from the text
checkpoint. It invokes individual processing directly, avoiding the known
batch-then-fallback repetition when an Ollama multimodal call crashes. Query
evaluation is intentionally performed by ``evaluate_retrieval_pages.py``.
"""

import asyncio
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from raganything import RAGAnything
from raganything.utils import separate_content

from content_postprocessing import normalize_docling_tables, print_table_normalization
from hello_raganything import (
    ARCH_NAME,
    _load_questions,
    _resolve_pdf_path,
)
from local_config import (
    PARSER_OUTPUT_DIR,
    PDF_INDEX_RANGE,
    RETRIEVAL_EVAL,
    TEXT_LLM,
    VISION_LLM,
    build_embedding_func,
    build_lightrag_kwargs,
    build_lightrag_workspace,
    build_llm_func,
    build_parser_kwargs,
    build_rag_config,
    build_vision_func,
    require_current_model_manifest,
)
from retrieval_provenance import (
    PAGE_PROVENANCE,
    process_multimodal_content_individual_with_pages,
    register_docling_provenance_parser,
    validate_content_pages,
)


CHECKPOINT_ARCH_NAME = f"{ARCH_NAME}-text-checkpoint"
ATTEMPT_ARCH_NAME = f"{ARCH_NAME}-multimodal-attempt"


def _text_checkpoint_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{CHECKPOINT_ARCH_NAME}/{Path(doc_id).stem}")


def _attempt_working_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{ATTEMPT_ARCH_NAME}/{Path(doc_id).stem}")


def _build_rag(working_dir: str) -> RAGAnything:
    register_docling_provenance_parser()
    workspace = build_lightrag_workspace(working_dir)
    return RAGAnything(
        config=build_rag_config(
            working_dir=working_dir, parser_name=RETRIEVAL_EVAL.parser_name
        ),
        llm_model_func=build_llm_func(),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(
            chunk_token_size=RETRIEVAL_EVAL.chunk_token_size,
            workspace=workspace,
        ),
    )


def _lightrag_storage_dir(root_dir: Path, manifest: dict) -> Path:
    workspace = manifest.get("lightrag_workspace") or build_lightrag_workspace(
        str(root_dir)
    )
    workspace_dir = root_dir / workspace
    return workspace_dir if workspace_dir.exists() else root_dir


def _read_storage_json(root_dir: Path, manifest: dict, filename: str) -> dict:
    storage_dir = _lightrag_storage_dir(root_dir, manifest)
    path = storage_dir / filename
    if not path.exists():
        raise RuntimeError(f"Missing LightRAG storage file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lightrag_checkpoint_storage(
    root_dir: Path,
    manifest: dict,
    *,
    expected_chunks: int,
    expected_status: str,
    require_multimodal_processed: bool | None,
    artifact_name: str,
) -> dict:
    """Check that a checkpoint/attempt contains one coherent document only."""

    doc_status = _read_storage_json(root_dir, manifest, "kv_store_doc_status.json")
    text_chunks = _read_storage_json(root_dir, manifest, "kv_store_text_chunks.json")
    content_doc_id = manifest.get("content_doc_id")

    if len(doc_status) != 1:
        raise RuntimeError(
            f"{artifact_name} storage is contaminated: expected exactly one "
            f"doc_status record, found {len(doc_status)}."
        )
    if content_doc_id not in doc_status:
        raise RuntimeError(
            f"{artifact_name} storage does not contain manifest content_doc_id "
            f"{content_doc_id!r}. Found: {list(doc_status)}"
        )

    status_record = doc_status[content_doc_id]
    if status_record.get("status") != expected_status:
        raise RuntimeError(
            f"{artifact_name} has unexpected doc status "
            f"{status_record.get('status')!r}; expected {expected_status!r}."
        )
    if require_multimodal_processed is not None and bool(
        status_record.get("multimodal_processed")
    ) != require_multimodal_processed:
        raise RuntimeError(
            f"{artifact_name} multimodal_processed is "
            f"{status_record.get('multimodal_processed')!r}; expected "
            f"{require_multimodal_processed!r}."
        )

    chunk_ids = status_record.get("chunks_list", [])
    if len(chunk_ids) != expected_chunks:
        raise RuntimeError(
            f"{artifact_name} chunk count mismatch: manifest expects "
            f"{expected_chunks}, doc_status lists {len(chunk_ids)}."
        )
    if len(text_chunks) != expected_chunks:
        raise RuntimeError(
            f"{artifact_name} storage is contaminated: manifest expects "
            f"{expected_chunks} text_chunks, storage has {len(text_chunks)}."
        )

    chunk_id_set = set(chunk_ids)
    stored_chunk_ids = set(text_chunks)
    if chunk_id_set != stored_chunk_ids:
        missing = sorted(chunk_id_set - stored_chunk_ids)[:5]
        extra = sorted(stored_chunk_ids - chunk_id_set)[:5]
        raise RuntimeError(
            f"{artifact_name} chunk id mismatch. Missing={missing}; extra={extra}."
        )

    wrong_doc_chunks = [
        chunk_id
        for chunk_id, chunk in text_chunks.items()
        if chunk.get("full_doc_id") != content_doc_id
    ][:5]
    if wrong_doc_chunks:
        raise RuntimeError(
            f"{artifact_name} has chunks from another document: {wrong_doc_chunks}."
        )

    missing_pages = [
        chunk_id
        for chunk_id, chunk in text_chunks.items()
        if not chunk.get("page_numbers")
    ][:5]
    if missing_pages:
        raise RuntimeError(
            f"{artifact_name} has chunks without page_numbers: {missing_pages}."
        )

    return {
        "storage_dir": str(_lightrag_storage_dir(root_dir, manifest)),
        "doc_status_records": len(doc_status),
        "text_chunks_records": len(text_chunks),
        "chunks_list_records": len(chunk_ids),
        "status": status_record.get("status"),
        "multimodal_processed": bool(status_record.get("multimodal_processed")),
    }


def _prepare_attempt(doc_id: str) -> tuple[Path, dict]:
    checkpoint_dir = _text_checkpoint_dir(doc_id)
    manifest_path = checkpoint_dir / "text_checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing text checkpoint for {doc_id}. "
            "Run `uv run python smoke_text_checkpoint.py` first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_current_model_manifest(manifest, "Text checkpoint")
    if (
        manifest.get("parser") != RETRIEVAL_EVAL.parser_name
        or manifest.get("parser_options") != build_parser_kwargs()
        or manifest.get("chunk_token_size") != RETRIEVAL_EVAL.chunk_token_size
        or manifest.get("page_provenance") != PAGE_PROVENANCE
    ):
        raise RuntimeError(
            "Text checkpoint predates page-aware retrieval evaluation. "
            "Run `uv run python smoke_text_checkpoint.py` first."
        )
    extraction_quality = manifest.get("text_extraction_quality", {})
    if (
        extraction_quality.get("invalid_outputs") != 0
        or extraction_quality.get("validated_outputs") != manifest.get("text_chunks")
    ):
        raise RuntimeError(
            "Text checkpoint predates validated textual extraction or contains "
            "invalid extraction outputs. Run `uv run python "
            "smoke_text_checkpoint.py` before multimodal processing."
        )
    if manifest.get("lightrag_workspace") != build_lightrag_workspace(
        str(checkpoint_dir)
    ):
        raise RuntimeError(
            "Text checkpoint does not declare the expected LightRAG workspace. "
            "Run `uv run python smoke_text_checkpoint.py` first."
        )
    validate_lightrag_checkpoint_storage(
        checkpoint_dir,
        manifest,
        expected_chunks=int(manifest["text_chunks"]),
        expected_status="handling",
        require_multimodal_processed=False,
        artifact_name=f"Text checkpoint for {doc_id}",
    )
    attempt_dir = _attempt_working_dir(doc_id)
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, attempt_dir)
    return attempt_dir, manifest


async def _process_multimodal_attempt(
    doc_id: str,
) -> None:
    started_at_utc = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    pdf_path = _resolve_pdf_path(doc_id)
    attempt_dir, manifest = _prepare_attempt(doc_id)
    print(
        f"[attempt] restored text checkpoint into {attempt_dir}; "
        "starting individual multimodal processing",
        flush=True,
    )
    print(
        f"[models] text={TEXT_LLM.name}; vision={VISION_LLM.name}; "
        f"GPU placement={manifest['gpu_offload_policy']}",
        flush=True,
    )

    rag = _build_rag(str(attempt_dir))
    try:
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(f"LightRAG initialization failed: {init_result}")

        content_list, content_doc_id = await rag.parse_document(
            str(pdf_path),
            PARSER_OUTPUT_DIR,
            "auto",
            True,
            **build_parser_kwargs(),
        )
        if content_doc_id != manifest["content_doc_id"]:
            raise RuntimeError(
                "Parsed document id differs from text checkpoint; rebuild the checkpoint."
            )

        validate_content_pages(content_list)
        content_list, table_reductions = normalize_docling_tables(content_list)
        print_table_normalization(table_reductions)
        _, multimodal_items = separate_content(content_list)
        if hasattr(rag, "set_content_source_for_context") and multimodal_items:
            rag.set_content_source_for_context(content_list, rag.config.content_format)

        print(
            f"[multimodal] processing {len(multimodal_items)} item(s) individually; "
            "a failed item will be logged and skipped by RAG-Anything",
            flush=True,
        )
        await process_multimodal_content_individual_with_pages(
            rag, multimodal_items, str(pdf_path), content_doc_id
        )

        status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        chunk_ids = status.get("chunks_list", [])
        stored_chunks = await rag.lightrag.text_chunks.get_by_ids(chunk_ids)
        missing_pages = [
            chunk_id
            for chunk_id, chunk in zip(chunk_ids, stored_chunks)
            if not chunk or not chunk.get("page_numbers")
        ]
        if missing_pages:
            raise RuntimeError(
                f"Integrated chunks missing deterministic pages: {missing_pages}"
            )
        added_chunks = len(chunk_ids) - int(manifest["text_chunks"])
        if added_chunks < 0:
            raise RuntimeError(
                "Multimodal attempt lost text chunks before integration. This "
                "usually indicates contaminated LightRAG storage; rebuild the "
                "text checkpoint with the isolated workspace fix."
            )
        finished_at_utc = datetime.now(timezone.utc)
        multimodal_processing_duration_s = round(
            time.perf_counter() - started_perf, 2
        )
        text_processing_duration_s = manifest.get("text_processing_duration_s")
        total_processing_duration_s = None
        if text_processing_duration_s is not None:
            total_processing_duration_s = round(
                float(text_processing_duration_s) + multimodal_processing_duration_s,
                2,
            )
        attempt_manifest = {
            **manifest,
            "working_dir": str(attempt_dir),
            "lightrag_workspace": build_lightrag_workspace(str(attempt_dir)),
            "multimodal_items_attempted": len(multimodal_items),
            "multimodal_chunks_added": added_chunks,
            "total_chunks": len(chunk_ids),
            "multimodal_processing_started_at_utc": started_at_utc.isoformat(),
            "multimodal_processing_finished_at_utc": finished_at_utc.isoformat(),
            "multimodal_processing_duration_s": multimodal_processing_duration_s,
            "total_processing_started_at_utc": manifest.get(
                "text_processing_started_at_utc"
            ),
            "total_processing_finished_at_utc": finished_at_utc.isoformat(),
            "total_processing_duration_s": total_processing_duration_s,
        }
        manifest_path = attempt_dir / "multimodal_attempt_manifest.json"
        manifest_path.write_text(
            json.dumps(attempt_manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[attempt ready] {added_chunks}/{len(multimodal_items)} multimodal "
            "chunk(s) integrated with page metadata; "
            f"multimodal_processing_duration_s={multimodal_processing_duration_s}; "
            f"total_processing_duration_s={total_processing_duration_s}; "
            f"result={attempt_dir}",
            flush=True,
        )
    finally:
        await rag.finalize_storages()


async def main() -> None:
    questions = _load_questions(PDF_INDEX_RANGE)

    for doc_id in questions["doc_id"].drop_duplicates():
        await _process_multimodal_attempt(doc_id)


if __name__ == "__main__":
    asyncio.run(main())
