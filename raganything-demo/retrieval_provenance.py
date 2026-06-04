"""Page provenance support for retrieval evaluation checkpoints.

RAG-Anything's bundled Docling adapter currently derives ``page_idx`` from a
block counter.  The helpers in this module preserve Docling's native
``prov.page_no`` instead and attach page metadata to LightRAG chunks without
inserting page markers into their searchable content.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from lightrag import utils as lightrag_utils
from lightrag.operate import chunking_by_token_size, merge_nodes_and_edges
from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock
from raganything.parser import DoclingParser, list_parsers, register_parser
from raganything.utils import get_processor_for_type


PARSER_NAME = "docling_provenance"
PAGE_PROVENANCE = "docling_prov"
INDEXABLE_TYPES = {"text", "image", "table", "equation"}


def _page_number_from_block(block: dict[str, Any], content_type: str) -> int:
    provenance = block.get("prov") or []
    if not provenance or not isinstance(provenance[0], dict):
        raise ValueError(f"Docling {content_type} block has no prov.page_no metadata")
    page_number = provenance[0].get("page_no")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError(
            f"Docling {content_type} block has invalid prov.page_no={page_number!r}"
        )
    return page_number


def require_page_number(item: dict[str, Any]) -> int:
    """Return the 1-based real page number or fail for indexable content."""
    page_number = item.get("page_number")
    if not isinstance(page_number, int) or page_number < 1:
        raise ValueError(
            f"Indexable {item.get('type', 'unknown')} item has no valid "
            f"page_number: {page_number!r}"
        )
    return page_number


class DoclingProvenanceParser(DoclingParser):
    """Docling parser adapter that preserves its native page provenance."""

    def _get_converter(self, **kwargs: Any) -> Any:
        """Build a Docling converter with low-memory options from local_config."""
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.datamodel.settings import settings
        from docling.document_converter import DocumentConverter, PdfFormatOption

        def optional_int(name: str) -> int | None:
            value = kwargs.get(name)
            return None if value is None else int(value)

        table_mode = str(kwargs.get("table_mode", "fast")).lower()
        do_tables = bool(kwargs.get("tables", True))
        do_ocr = bool(kwargs.get("allow_ocr", True))
        artifacts_path = kwargs.get("artifacts_path")
        images_scale = float(kwargs.get("images_scale", 2.0))
        generate_picture_images = bool(kwargs.get("generate_picture_images", True))
        page_batch_size = optional_int("page_batch_size")
        ocr_batch_size = optional_int("ocr_batch_size")
        layout_batch_size = optional_int("layout_batch_size")
        table_batch_size = optional_int("table_batch_size")
        queue_max_size = optional_int("queue_max_size")
        accelerator_num_threads = optional_int("accelerator_num_threads")
        accelerator_device = str(kwargs.get("accelerator_device", "cpu"))
        document_timeout_s = kwargs.get("document_timeout_s")

        if page_batch_size is not None:
            settings.perf.page_batch_size = page_batch_size
        if accelerator_num_threads is not None:
            settings.perf.elements_batch_size = min(
                settings.perf.elements_batch_size, max(1, accelerator_num_threads)
            )

        cache_key = (
            table_mode,
            do_tables,
            do_ocr,
            artifacts_path,
            images_scale,
            generate_picture_images,
            page_batch_size,
            ocr_batch_size,
            layout_batch_size,
            table_batch_size,
            queue_max_size,
            accelerator_num_threads,
            accelerator_device,
            document_timeout_s,
        )
        cached = self._converter_cache.get(cache_key)
        if cached is not None:
            return cached

        pipeline_options = PdfPipelineOptions()
        if hasattr(pipeline_options, "do_ocr"):
            pipeline_options.do_ocr = do_ocr
        if hasattr(pipeline_options, "do_table_structure"):
            pipeline_options.do_table_structure = do_tables
        if hasattr(pipeline_options, "table_structure_options"):
            try:
                pipeline_options.table_structure_options.mode = (
                    TableFormerMode.ACCURATE
                    if table_mode == "accurate"
                    else TableFormerMode.FAST
                )
            except Exception as e:  # pragma: no cover - defensive
                self.logger.debug(f"Could not set TableFormer mode '{table_mode}': {e}")
        if artifacts_path and hasattr(pipeline_options, "artifacts_path"):
            pipeline_options.artifacts_path = artifacts_path
        if hasattr(pipeline_options, "generate_picture_images"):
            pipeline_options.generate_picture_images = generate_picture_images
        if hasattr(pipeline_options, "images_scale"):
            pipeline_options.images_scale = images_scale
        if document_timeout_s is not None and hasattr(
            pipeline_options, "document_timeout"
        ):
            pipeline_options.document_timeout = float(document_timeout_s)
        if ocr_batch_size is not None and hasattr(pipeline_options, "ocr_batch_size"):
            pipeline_options.ocr_batch_size = ocr_batch_size
        if layout_batch_size is not None and hasattr(
            pipeline_options, "layout_batch_size"
        ):
            pipeline_options.layout_batch_size = layout_batch_size
        if table_batch_size is not None and hasattr(
            pipeline_options, "table_batch_size"
        ):
            pipeline_options.table_batch_size = table_batch_size
        if queue_max_size is not None and hasattr(pipeline_options, "queue_max_size"):
            pipeline_options.queue_max_size = queue_max_size
        if accelerator_num_threads is not None and hasattr(
            pipeline_options, "accelerator_options"
        ):
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=accelerator_num_threads,
                device=accelerator_device,
            )

        with self._converter_cache_lock:
            cached = self._converter_cache.get(cache_key)
            if cached is not None:
                return cached
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
            self._converter_cache[cache_key] = converter
            return converter

    def read_from_block_recursive(
        self,
        block: dict[str, Any],
        type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if type == "groups" and not block.get("children"):
            self.logger.debug(
                "Skipping empty Docling group block without page provenance: "
                f"{block.get('self_ref', num)}"
            )
            return []
        return super().read_from_block_recursive(
            block, type, output_dir, cnt, num, docling_content
        )

    def read_from_block(
        self, block: dict[str, Any], type: str, output_dir: Path, cnt: int, num: str
    ) -> dict[str, Any]:
        item = super().read_from_block(block, type, output_dir, cnt, num)
        content_type = item.get("type")
        if content_type in INDEXABLE_TYPES:
            page_number = _page_number_from_block(block, content_type)
            item["page_number"] = page_number
            item["page_idx"] = page_number - 1
            item["page_provenance"] = PAGE_PROVENANCE
        return item


def register_docling_provenance_parser() -> None:
    """Register the corrected parser once in the current Python process."""
    existing = list_parsers().get(PARSER_NAME)
    if existing is None:
        register_parser(PARSER_NAME, DoclingProvenanceParser)
    elif existing != DoclingProvenanceParser.__name__:
        raise RuntimeError(
            f"Parser name {PARSER_NAME!r} is already registered as {existing!r}"
        )


def validate_content_pages(content_list: list[dict[str, Any]]) -> None:
    """Reject parsed indexable content without deterministic page provenance."""
    for item in content_list:
        if item.get("type") in INDEXABLE_TYPES:
            require_page_number(item)


@dataclass(frozen=True)
class TextPageSpan:
    start: int
    end: int
    page_number: int


def _sanitize_text_fragment_like_lightrag(text: str) -> str:
    """Apply LightRAG's unsafe-character cleanup without per-block stripping."""
    if not text:
        return text
    text = html.unescape(text)
    text = lightrag_utils._SURROGATE_PATTERN.sub("", text)
    text = lightrag_utils._CONTROL_CHAR_PATTERN_ALL.sub("", text)
    return text


def build_text_content_with_page_spans(
    content_list: list[dict[str, Any]], *, sanitize_for_lightrag: bool = False
) -> tuple[str, list[TextPageSpan]]:
    """Build the same text as ``separate_content`` plus source-page spans."""
    text_parts: list[str] = []
    spans: list[TextPageSpan] = []
    position = 0

    for item in content_list:
        if item.get("type") != "text":
            continue
        text = item.get("text", "")
        if not text.strip():
            continue
        page_number = require_page_number(item)
        if sanitize_for_lightrag:
            text = _sanitize_text_fragment_like_lightrag(text)
        if text_parts:
            position += 2  # ``separate_content`` joins text parts with "\n\n".
        start = position
        text_parts.append(text)
        position += len(text)
        spans.append(TextPageSpan(start=start, end=position, page_number=page_number))

    text_content = "\n\n".join(text_parts)
    if not sanitize_for_lightrag:
        return text_content, spans

    left_trim = len(text_content) - len(text_content.lstrip())
    stripped_text = text_content.strip()
    stripped_len = len(stripped_text)
    adjusted_spans = [
        TextPageSpan(
            start=max(0, span.start - left_trim),
            end=min(stripped_len, span.end - left_trim),
            page_number=span.page_number,
        )
        for span in spans
        if max(0, span.start - left_trim) < min(stripped_len, span.end - left_trim)
    ]
    return stripped_text, adjusted_spans


def _pages_for_range(
    start: int, end: int, spans: list[TextPageSpan]
) -> list[int]:
    pages: list[int] = []
    for span in spans:
        if span.start < end and start < span.end and span.page_number not in pages:
            pages.append(span.page_number)
    if not pages:
        raise ValueError(
            f"Chunk character range {start}:{end} did not overlap any source page"
        )
    return pages


def build_page_aware_chunking_func(
    expected_content: str, spans: list[TextPageSpan]
) -> Callable[..., list[dict[str, Any]]]:
    """Return standard token chunking with persisted page metadata."""

    def page_aware_chunking(
        tokenizer,
        content: str,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
    ) -> list[dict[str, Any]]:
        if content != expected_content:
            raise ValueError("Page-aware chunking received an unexpected document body")
        if split_by_character is not None or split_by_character_only:
            raise ValueError("Page-aware chunking requires token-based splitting only")

        chunks = chunking_by_token_size(
            tokenizer,
            content,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            chunk_overlap_token_size=chunk_overlap_token_size,
            chunk_token_size=chunk_token_size,
        )
        tokens = tokenizer.encode(content)
        step = chunk_token_size - chunk_overlap_token_size
        starts = range(0, len(tokens), step)

        for chunk, token_start in zip(chunks, starts):
            token_end = min(token_start + chunk_token_size, len(tokens))
            char_start = len(tokenizer.decode(tokens[:token_start]))
            char_end = len(tokenizer.decode(tokens[:token_end]))
            chunk["page_numbers"] = _pages_for_range(char_start, char_end, spans)
            chunk["content_type"] = "text"
            chunk["page_provenance"] = PAGE_PROVENANCE
        return chunks

    return page_aware_chunking


async def process_multimodal_content_individual_with_pages(
    rag, multimodal_items: list[dict[str, Any]], file_path: str, doc_id: str
) -> None:
    """Run RAG-Anything's resilient individual path and persist item pages."""
    file_name = rag._get_file_reference(file_path)
    all_chunk_results: list[Any] = []
    multimodal_chunk_ids: list[str] = []
    existing_doc_status = await rag.lightrag.doc_status.get_by_id(doc_id)
    existing_chunks_count = (
        existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
    )

    for index, item in enumerate(multimodal_items):
        page_number = require_page_number(item)
        content_type = item.get("type", "unknown")
        try:
            rag.logger.info(
                f"Processing item {index + 1}/{len(multimodal_items)}: "
                f"{content_type} content"
            )
            processor = get_processor_for_type(rag.modal_processors, content_type)
            if not processor:
                rag.logger.warning(
                    f"No suitable processor found for {content_type} type content"
                )
                continue

            item_info = {
                "page_idx": page_number - 1,
                "index": item.get("_content_list_index", index),
                "type": content_type,
            }
            _, entity_info, chunk_results = await processor.process_multimodal_content(
                modal_content=item,
                content_type=content_type,
                file_path=file_name,
                item_info=item_info,
                batch_mode=True,
                doc_id=doc_id,
                chunk_order_index=existing_chunks_count + index,
            )
            all_chunk_results.extend(chunk_results)

            chunk_id = entity_info.get("chunk_id") if entity_info else None
            if chunk_id:
                stored_chunk = await rag.lightrag.text_chunks.get_by_id(chunk_id)
                if not stored_chunk:
                    raise RuntimeError(
                        f"Multimodal processor returned missing chunk {chunk_id}"
                    )
                stored_chunk.pop("_id", None)
                stored_chunk.update(
                    {
                        "page_numbers": [page_number],
                        "content_type": content_type,
                        "page_provenance": PAGE_PROVENANCE,
                    }
                )
                await rag.lightrag.text_chunks.upsert({chunk_id: stored_chunk})
                multimodal_chunk_ids.append(chunk_id)

            rag.logger.info(
                f"{content_type} processing complete: "
                f"{entity_info.get('entity_name', 'Unknown')}"
            )
        except Exception as exc:
            rag.logger.error(f"Error processing multimodal content: {str(exc)}")
            rag.logger.debug("Exception details:", exc_info=True)
            continue

    if multimodal_chunk_ids:
        current_status = await rag.lightrag.doc_status.get_by_id(doc_id)
        if current_status:
            current_status.pop("_id", None)
            existing_chunks_list = current_status.get("chunks_list", [])
            current_status.update(
                {
                    "chunks_list": existing_chunks_list + multimodal_chunk_ids,
                    "chunks_count": current_status.get("chunks_count", 0)
                    + len(multimodal_chunk_ids),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                }
            )
            await rag.lightrag.doc_status.upsert({doc_id: current_status})
            await rag.lightrag.doc_status.index_done_callback()
            rag.logger.info(
                f"Updated doc_status with {len(multimodal_chunk_ids)} "
                "multimodal chunks integrated into chunks_list"
            )

    if all_chunk_results:
        workspace = getattr(rag.lightrag, "workspace", "")
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=workspace
        )
        pipeline_status_lock = get_pipeline_status_lock(workspace=workspace)
        await merge_nodes_and_edges(
            chunk_results=all_chunk_results,
            knowledge_graph_inst=rag.lightrag.chunk_entity_relation_graph,
            entity_vdb=rag.lightrag.entities_vdb,
            relationships_vdb=rag.lightrag.relationships_vdb,
            global_config=rag.lightrag.__dict__,
            full_entities_storage=rag.lightrag.full_entities,
            full_relations_storage=rag.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=rag.lightrag.llm_response_cache,
            entity_chunks_storage=rag.lightrag.entity_chunks,
            relation_chunks_storage=rag.lightrag.relation_chunks,
            current_file_number=1,
            total_files=1,
            file_path=file_name,
        )
        await rag.lightrag._insert_done()
    else:
        await rag.lightrag.text_chunks.index_done_callback()

    rag.logger.info("Individual multimodal content processing complete")
    await rag._mark_multimodal_processing_complete(doc_id)
