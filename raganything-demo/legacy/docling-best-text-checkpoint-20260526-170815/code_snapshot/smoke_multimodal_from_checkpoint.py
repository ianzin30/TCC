"""Process multimodal content from a saved text-only smoke-test checkpoint.

Usage:
    uv run python smoke_multimodal_from_checkpoint.py

Each invocation creates a fresh multimodal attempt copied from the text
checkpoint. It invokes individual processing directly, avoiding the known
batch-then-fallback repetition when an Ollama multimodal call crashes.
"""

import asyncio
import json
import shutil
import time
from pathlib import Path

from raganything import RAGAnything
from raganything.utils import separate_content

from hello_raganything import (
    ARCH_NAME,
    PDF_GRID,
    _aquery_with_progress,
    _load_questions,
    _resolve_pdf_path,
)
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
ATTEMPT_ARCH_NAME = f"{ARCH_NAME}-multimodal-attempt"


def _text_checkpoint_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{CHECKPOINT_ARCH_NAME}/{Path(doc_id).stem}")


def _attempt_working_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{ATTEMPT_ARCH_NAME}/{Path(doc_id).stem}")


def _build_rag(working_dir: str) -> RAGAnything:
    return RAGAnything(
        config=build_rag_config(working_dir=working_dir),
        llm_model_func=build_llm_func(),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(),
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
    attempt_dir = _attempt_working_dir(doc_id)
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, attempt_dir)
    return attempt_dir, manifest


async def _process_multimodal_attempt(
    doc_id: str, group, results: list[dict], question_offset: int, question_total: int
) -> int:
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

        _, multimodal_items = separate_content(content_list)
        file_ref = rag._get_file_reference(str(pdf_path))
        if hasattr(rag, "set_content_source_for_context") and multimodal_items:
            rag.set_content_source_for_context(content_list, rag.config.content_format)

        print(
            f"[multimodal] processing {len(multimodal_items)} item(s) individually; "
            "a failed item will be logged and skipped by RAG-Anything",
            flush=True,
        )
        await rag._process_multimodal_content_individual(
            multimodal_items, file_ref, content_doc_id
        )

        question_number = question_offset
        for _, row in group.iterrows():
            question_number += 1
            started = time.perf_counter()
            try:
                model_answer = await _aquery_with_progress(
                    rag,
                    row["question"],
                    doc_id,
                    question_number,
                    question_total,
                )
                error = None
            except Exception as exc:
                model_answer, error = None, repr(exc)

            results.append(
                {
                    "arch": ATTEMPT_ARCH_NAME,
                    "doc_id": doc_id,
                    "question": row["question"],
                    "ground_truth": row["answer"],
                    "model_answer": model_answer,
                    "duration_s": round(time.perf_counter() - started, 2),
                    "error": error,
                }
            )
        return question_number
    finally:
        await rag.finalize_storages()


async def main() -> None:
    questions = _load_questions(PDF_GRID)
    results: list[dict] = []
    question_number = 0

    for doc_id, group in questions.groupby("doc_id", sort=False):
        question_number = await _process_multimodal_attempt(
            doc_id, group, results, question_number, len(questions)
        )

    out_dir = Path("./smoke_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ATTEMPT_ARCH_NAME}.jsonl"
    with out_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, default=str, ensure_ascii=False) + "\n")
    print(f"[done] {len(results)} answers -> {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
