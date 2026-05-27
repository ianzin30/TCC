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

from hello_raganything import PDF_GRID, _load_questions, _resolve_pdf_path
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
    probe_dir = _probe_working_dir(doc_id, content_type, index)
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, probe_dir)
    return probe_dir, manifest


def _item_label(item: dict[str, Any], type_index: int) -> str:
    page = item.get("page_idx", "unknown")
    content_index = item.get("_content_list_index", "unknown")
    path = item.get("img_path", "")
    path_name = Path(path).name if path else "-"
    return (
        f"{type_index}: page_idx={page}, content_index={content_index}, "
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
            "./output",
            "auto",
            True,
        )
        if content_doc_id != manifest["content_doc_id"]:
            raise RuntimeError(
                "Parsed document id differs from text checkpoint; rebuild the checkpoint."
            )

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
        file_ref = rag._get_file_reference(str(pdf_path))
        rag.set_content_source_for_context(content_list, rag.config.content_format)
        initial_status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        chunks_before = int(initial_status.get("chunks_count", 0))

        print(
            f"[probe] processing {_item_label(selected_item, index)} "
            f"from isolated copy {probe_dir}",
            flush=True,
        )
        await rag._process_multimodal_content_individual(
            [selected_item], file_ref, content_doc_id
        )

        final_status = await rag.lightrag.doc_status.get_by_id(content_doc_id) or {}
        chunks_after = int(final_status.get("chunks_count", 0))
        added_chunks = chunks_after - chunks_before
        result = {
            "document": doc_id,
            "content_type": content_type,
            "index": index,
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
    questions = _load_questions(PDF_GRID)
    for doc_id in questions["doc_id"].drop_duplicates():
        await _probe_document(doc_id, args.content_type, args.index, args.list)


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
