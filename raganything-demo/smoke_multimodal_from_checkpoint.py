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
    build_embedding_func,
    build_lightrag_kwargs,
    build_llm_func,
    build_parser_kwargs,
    build_rag_config,
    build_vision_func,
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
    return RAGAnything(
        config=build_rag_config(
            working_dir=working_dir, parser_name=RETRIEVAL_EVAL.parser_name
        ),
        llm_model_func=build_llm_func(),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(
            chunk_token_size=RETRIEVAL_EVAL.chunk_token_size
        ),
    )


def _prepare_attempt(doc_id: str) -> tuple[Path, dict]:
    checkpoint_dir = _text_checkpoint_dir(doc_id)
    manifest_path = checkpoint_dir / "text_checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing text checkpoint for {doc_id}. "
            "Run `uv run python smoke_text_checkpoint.py` first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("parser") != RETRIEVAL_EVAL.parser_name
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
    attempt_dir = _attempt_working_dir(doc_id)
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, attempt_dir)
    return attempt_dir, manifest


async def _process_multimodal_attempt(
    doc_id: str,
) -> None:
    pdf_path = _resolve_pdf_path(doc_id)
    attempt_dir, manifest = _prepare_attempt(doc_id)
    print(
        f"[attempt] restored text checkpoint into {attempt_dir}; "
        "starting individual multimodal processing",
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
        attempt_manifest = {
            **manifest,
            "working_dir": str(attempt_dir),
            "multimodal_items_attempted": len(multimodal_items),
            "multimodal_chunks_added": added_chunks,
            "total_chunks": len(chunk_ids),
        }
        manifest_path = attempt_dir / "multimodal_attempt_manifest.json"
        manifest_path.write_text(
            json.dumps(attempt_manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[attempt ready] {added_chunks}/{len(multimodal_items)} multimodal "
            f"chunk(s) integrated with page metadata; result={attempt_dir}",
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
