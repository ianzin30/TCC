"""Build a reusable text-only checkpoint for the MMLongBench smoke document.

Usage:
    uv run python smoke_text_checkpoint.py

This intentionally stops before multimodal processing. A later run of
``smoke_multimodal_from_checkpoint.py`` copies this checkpoint and performs
the expensive/fragile image and table stage without repeating text indexing.
"""

import asyncio
import json
import shutil
from pathlib import Path

from raganything import RAGAnything
from raganything.base import DocStatus
from raganything.utils import insert_text_content, separate_content

from hello_raganything import ARCH_NAME, PDF_GRID, _load_questions, _resolve_pdf_path
from local_config import (
    PARSER_OUTPUT_DIR,
    build_embedding_func,
    build_lightrag_kwargs,
    build_llm_func,
    build_parser_kwargs,
    build_rag_config,
    build_vision_func,
)


CHECKPOINT_ARCH_NAME = f"{ARCH_NAME}-text-checkpoint"


def _checkpoint_working_dir(doc_id: str) -> str:
    return f"./rag_storage/{CHECKPOINT_ARCH_NAME}/{Path(doc_id).stem}"


def _build_rag(working_dir: str) -> RAGAnything:
    return RAGAnything(
        config=build_rag_config(working_dir=working_dir),
        llm_model_func=build_llm_func(),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(),
    )


async def _build_document_checkpoint(doc_id: str, question_count: int) -> None:
    pdf_path = _resolve_pdf_path(doc_id)
    working_dir = _checkpoint_working_dir(doc_id)
    checkpoint_path = Path(working_dir)

    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
        print(f"[clean] removed previous text checkpoint {working_dir}", flush=True)

    rag = _build_rag(working_dir)
    try:
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(f"LightRAG initialization failed: {init_result}")

        print(f"[text checkpoint] {doc_id} ({question_count} questions)", flush=True)
        content_list, content_doc_id = await rag.parse_document(
            str(pdf_path),
            PARSER_OUTPUT_DIR,
            "auto",
            True,
            **build_parser_kwargs(),
        )
        text_content, multimodal_items = separate_content(content_list)
        file_ref = rag._get_file_reference(str(pdf_path))

        await insert_text_content(
            rag.lightrag,
            input=text_content,
            file_paths=file_ref,
            ids=content_doc_id,
        )
        await rag._upsert_doc_status(
            content_doc_id,
            file_ref,
            status=DocStatus.HANDLING,
            error_msg="",
        )

        manifest = {
            "doc_id": doc_id,
            "content_doc_id": content_doc_id,
            "pdf_path": str(pdf_path),
            "text_characters": len(text_content),
            "multimodal_items": len(multimodal_items),
            "working_dir": working_dir,
        }
        manifest_path = checkpoint_path / "text_checkpoint_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[checkpoint ready] text graph saved at {working_dir}; "
            f"{len(multimodal_items)} multimodal items deferred",
            flush=True,
        )
    finally:
        await rag.finalize_storages()


async def main() -> None:
    questions = _load_questions(PDF_GRID)
    print(
        f"[grid] {questions['doc_id'].nunique()} PDF(s), "
        f"{len(questions)} answerable questions",
        flush=True,
    )
    for doc_id, group in questions.groupby("doc_id", sort=False):
        await _build_document_checkpoint(doc_id, len(group))


if __name__ == "__main__":
    asyncio.run(main())
