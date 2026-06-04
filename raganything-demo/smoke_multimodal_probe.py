"""Probe one multimodal item type from the saved text checkpoint.

Usage:
    uv run python smoke_multimodal_probe.py IMAGE
    uv run python smoke_multimodal_probe.py TABLE --index 1
    uv run python smoke_multimodal_probe.py EQUATION --list

``--index`` is one-based within the selected modality. Each probe uses an
isolated copy of the text checkpoint and does not generate query answers.
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from raganything.utils import separate_content

from content_postprocessing import normalize_docling_tables, print_table_normalization
from hello_raganything import _load_questions, _resolve_pdf_path
from local_config import (
    PARSER_OUTPUT_DIR,
    PDF_INDEX_RANGE,
    RETRIEVAL_EVAL,
    build_parser_kwargs,
    require_current_model_manifest,
)
from retrieval_provenance import (
    PAGE_PROVENANCE,
    process_multimodal_content_individual_with_pages,
    validate_content_pages,
)
from smoke_multimodal_from_checkpoint import _build_rag, _text_checkpoint_dir


PROBE_ARCH_NAME = "ollama-qwen-bge-multimodal-probe"
SUPPORTED_TYPES = ("image", "table", "equation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process exactly one selected multimodal item from the text checkpoint."
    )
    parser.add_argument(
        "content_type",
        type=str.lower,
        choices=SUPPORTED_TYPES,
        metavar="TYPE",
        help="Multimodal type to probe: IMAGE, TABLE, or EQUATION.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=1,
        help="One-based occurrence within TYPE to process (default: 1).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching items and stop without invoking the model.",
    )
    args = parser.parse_args()
    if args.index < 1:
        parser.error("--index must be at least 1")
    return args


def _probe_working_dir(doc_id: str, content_type: str, index: int) -> Path:
    suffix = f"{Path(doc_id).stem}-{content_type}-{index}"
    return Path(f"./rag_storage/{PROBE_ARCH_NAME}/{suffix}")


def _prepare_probe_dir(doc_id: str, content_type: str, index: int) -> tuple[Path, dict]:
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
            "smoke_text_checkpoint.py` before probing multimodal content."
        )
    probe_dir = _probe_working_dir(doc_id, content_type, index)
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, probe_dir)
    return probe_dir, manifest


def _item_label(item: dict[str, Any], type_index: int) -> str:
    page = item.get("page_number", "unknown")
    content_index = item.get("_content_list_index", "unknown")
    path = item.get("img_path", "")
    path_name = Path(path).name if path else "-"
    return (
        f"{type_index}: page_number={page}, content_index={content_index}, "
        f"asset={path_name}"
    )


async def _probe_document(doc_id: str, content_type: str, index: int, list_only: bool) -> None:
    pdf_path = _resolve_pdf_path(doc_id)
    probe_dir, manifest = _prepare_probe_dir(doc_id, content_type, index)
    rag = _build_rag(str(probe_dir))
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
        _, all_multimodal_items = separate_content(content_list)
        matching_items = [
            item for item in all_multimodal_items if item.get("type") == content_type
        ]
        counts = {
            kind: sum(1 for item in all_multimodal_items if item.get("type") == kind)
            for kind in SUPPORTED_TYPES
        }
        print(
            "[available] "
            + ", ".join(f"{kind}={count}" for kind, count in counts.items()),
            flush=True,
        )

        if not matching_items:
            print(f"[skip] no {content_type} items exist in this document", flush=True)
            return

        if list_only:
            for position, item in enumerate(matching_items, start=1):
                print(f"[{content_type}] {_item_label(item, position)}", flush=True)
            return

        if index > len(matching_items):
            raise IndexError(
                f"Requested {content_type} --index {index}, "
                f"but only {len(matching_items)} item(s) exist."
            )

        selected_item = matching_items[index - 1]
        rag.set_content_source_for_context(content_list, rag.config.content_format)
        initial_status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        chunks_before = int(initial_status.get("chunks_count", 0))

        print(
            f"[probe] processing {_item_label(selected_item, index)} "
            f"from isolated copy {probe_dir}",
            flush=True,
        )
        await process_multimodal_content_individual_with_pages(
            rag, [selected_item], str(pdf_path), content_doc_id
        )

        final_status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        chunks_after = int(final_status.get("chunks_count", 0))
        added_chunks = chunks_after - chunks_before
        result = {
            "document": doc_id,
            "text_llm_model": manifest["text_llm_model"],
            "vision_llm_model": manifest["vision_llm_model"],
            "content_type": content_type,
            "index": index,
            "page_number": selected_item.get("page_number"),
            "page_idx": selected_item.get("page_idx"),
            "chunks_before": chunks_before,
            "chunks_after": chunks_after,
            "added_chunks": added_chunks,
            "success": added_chunks > 0,
        }
        result_path = probe_dir / "probe_result.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        verdict = "SUCCESS" if result["success"] else "FAILED"
        print(
            f"[probe {verdict}] {content_type} #{index}: "
            f"added_chunks={added_chunks}; result={result_path}",
            flush=True,
        )
    finally:
        await rag.finalize_storages()


async def main(args: argparse.Namespace) -> None:
    questions = _load_questions(PDF_INDEX_RANGE)
    for doc_id in questions["doc_id"].drop_duplicates():
        await _probe_document(doc_id, args.content_type, args.index, args.list)


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
