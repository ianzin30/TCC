"""Validate RAG-Anything text checkpoints and multimodal attempts.

Usage:
    uv run python validate_rag_checkpoints.py

This is a read-only sanity check for the current PDF_INDEX_RANGE. It verifies
that each artifact uses the isolated LightRAG workspace, contains exactly one
document status record, has the expected chunk count, and all chunks carry
page_numbers for retrieval evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from hello_raganything import _load_questions
from local_config import (
    PDF_INDEX_RANGE,
    RETRIEVAL_EVAL,
    build_lightrag_workspace,
    build_parser_kwargs,
    require_current_model_manifest,
)
from retrieval_provenance import PAGE_PROVENANCE
from smoke_multimodal_from_checkpoint import (
    ATTEMPT_ARCH_NAME,
    CHECKPOINT_ARCH_NAME,
    validate_lightrag_checkpoint_storage,
)


def _text_checkpoint_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{CHECKPOINT_ARCH_NAME}/{Path(doc_id).stem}")


def _attempt_dir(doc_id: str) -> Path:
    return Path(f"./rag_storage/{ATTEMPT_ARCH_NAME}/{Path(doc_id).stem}")


def _check_common_manifest(manifest: dict[str, Any], root: Path, kind: str) -> None:
    require_current_model_manifest(manifest, kind)
    if manifest.get("parser") != RETRIEVAL_EVAL.parser_name:
        raise RuntimeError("parser mismatch")
    if manifest.get("parser_options") != build_parser_kwargs():
        raise RuntimeError("parser_options mismatch")
    if manifest.get("chunk_token_size") != RETRIEVAL_EVAL.chunk_token_size:
        raise RuntimeError("chunk_token_size mismatch")
    if manifest.get("page_provenance") != PAGE_PROVENANCE:
        raise RuntimeError("page_provenance mismatch")

    expected_workspace = build_lightrag_workspace(str(root))
    if manifest.get("lightrag_workspace") != expected_workspace:
        raise RuntimeError(
            "workspace mismatch: "
            f"{manifest.get('lightrag_workspace')!r} != {expected_workspace!r}"
        )


def _check_text_checkpoint(doc_id: str) -> dict[str, Any] | None:
    root = _text_checkpoint_dir(doc_id)
    manifest_path = root / "text_checkpoint_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_common_manifest(manifest, root, "Text checkpoint")
    quality = manifest.get("text_extraction_quality", {})
    if (
        quality.get("invalid_outputs") != 0
        or quality.get("validated_outputs") != manifest.get("text_chunks")
    ):
        raise RuntimeError("text extraction quality mismatch")

    storage = validate_lightrag_checkpoint_storage(
        root,
        manifest,
        expected_chunks=int(manifest["text_chunks"]),
        expected_status="handling",
        require_multimodal_processed=False,
        artifact_name=f"Text checkpoint for {doc_id}",
    )
    return {
        "chunks": manifest.get("text_chunks"),
        "multimodal_items": manifest.get("multimodal_items"),
        "duration_s": manifest.get("text_processing_duration_s"),
        "storage_dir": storage["storage_dir"],
    }


def _check_multimodal_attempt(doc_id: str) -> dict[str, Any] | None:
    root = _attempt_dir(doc_id)
    manifest_path = root / "multimodal_attempt_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_common_manifest(manifest, root, "Multimodal attempt")
    if int(manifest.get("total_chunks") or 0) <= 0:
        raise RuntimeError("total_chunks <= 0")
    if int(manifest.get("multimodal_chunks_added") or 0) < 0:
        raise RuntimeError("negative multimodal_chunks_added")

    storage = validate_lightrag_checkpoint_storage(
        root,
        manifest,
        expected_chunks=int(manifest["total_chunks"]),
        expected_status="processed",
        require_multimodal_processed=True,
        artifact_name=f"Multimodal attempt for {doc_id}",
    )
    return {
        "total_chunks": manifest.get("total_chunks"),
        "multimodal_chunks_added": manifest.get("multimodal_chunks_added"),
        "text_duration_s": manifest.get("text_processing_duration_s"),
        "multimodal_duration_s": manifest.get("multimodal_processing_duration_s"),
        "total_duration_s": manifest.get("total_processing_duration_s"),
        "storage_dir": storage["storage_dir"],
    }


def _format_indexes(items: list[tuple[int, str, Any]]) -> str:
    return ", ".join(str(index) for index, _, _ in items) or "-"


def _print_bad(title: str, items: list[tuple[int, str, str]]) -> None:
    if not items:
        return
    print(f"[{title}] bad details:")
    for index, doc_id, error in items:
        print(f"  {index}: {doc_id}: {error}")


def main() -> int:
    questions = _load_questions(PDF_INDEX_RANGE)
    docs = list(questions["doc_id"].drop_duplicates())

    text_ok: list[tuple[int, str, Any]] = []
    text_missing: list[tuple[int, str, Any]] = []
    text_bad: list[tuple[int, str, str]] = []
    multimodal_ok: list[tuple[int, str, Any]] = []
    multimodal_missing: list[tuple[int, str, Any]] = []
    multimodal_bad: list[tuple[int, str, str]] = []

    first_index = PDF_INDEX_RANGE[0]
    for offset, doc_id in enumerate(docs):
        index = first_index + offset
        try:
            info = _check_text_checkpoint(doc_id)
            target = text_missing if info is None else text_ok
            target.append((index, doc_id, info))
        except Exception as exc:
            text_bad.append((index, doc_id, str(exc)))

        try:
            info = _check_multimodal_attempt(doc_id)
            target = multimodal_missing if info is None else multimodal_ok
            target.append((index, doc_id, info))
        except Exception as exc:
            multimodal_bad.append((index, doc_id, str(exc)))

    print(f"[validate] PDF_INDEX_RANGE={PDF_INDEX_RANGE}; docs={len(docs)}")
    print(
        f"[text] ok={len(text_ok)} missing={len(text_missing)} "
        f"bad={len(text_bad)}"
    )
    print(f"[text] ok indexes: {_format_indexes(text_ok)}")
    print(f"[text] missing indexes: {_format_indexes(text_missing)}")
    _print_bad("text", text_bad)

    print(
        f"[multimodal] ok={len(multimodal_ok)} missing={len(multimodal_missing)} "
        f"bad={len(multimodal_bad)}"
    )
    print(f"[multimodal] ok indexes: {_format_indexes(multimodal_ok)}")
    print(f"[multimodal] missing indexes: {_format_indexes(multimodal_missing)}")
    _print_bad("multimodal", multimodal_bad)

    is_ready = not (
        text_missing or text_bad or multimodal_missing or multimodal_bad
    )
    print(f"[ready_for_eval] {is_ready}")
    return 0 if is_ready else 1


if __name__ == "__main__":
    sys.exit(main())
