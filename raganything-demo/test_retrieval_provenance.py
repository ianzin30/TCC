"""Fast validation tests for page-aware retrieval preparation."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from raganything.utils import separate_content

from local_config import (
    EXTRACTION_QUALITY,
    ExtractionQualityStats,
    LLMConfig,
    PARSER_NAME,
    RETRIEVAL_EVAL,
    RUNTIME,
    TEXT_LLM,
    VISION_LLM,
    _extraction_llm_call_with_metadata,
    _llm_call,
    _quality_checked_llm_call,
    _vision_call,
    analyze_extraction_output,
    build_lightrag_kwargs,
    build_parser_kwargs,
    model_manifest_fields,
    require_current_model_manifest,
)
from evaluate_retrieval_pages import _metrics_for_question, _ranked_pages
from retrieval_provenance import (
    DoclingProvenanceParser,
    TextPageSpan,
    build_page_aware_chunking_func,
)
from smoke_text_checkpoint import (
    DoclingParseIssueCapture,
    _summarize_docling_parse_issues,
)


ROOT = Path(__file__).resolve().parent
DOCLING_JSON = (
    ROOT
    / "output"
    / "PH_2016.06.08_Economy-Final_9acff927"
    / "PH_2016.06.08_Economy-Final"
    / "docling"
    / "PH_2016.06.08_Economy-Final.json"
)
PARSE_CACHE = (
    ROOT
    / "rag_storage"
    / "ollama-qwen-bge-text-checkpoint"
    / "PH_2016.06.08_Economy-Final"
    / "kv_store_parse_cache.json"
)


class ProvenanceParserTests(unittest.TestCase):
    def test_retrieval_profile_does_not_change_default_demo_config(self) -> None:
        self.assertEqual(PARSER_NAME, "docling")
        self.assertEqual(RUNTIME.chunk_token_size, 1000)
        self.assertEqual(RETRIEVAL_EVAL.parser_name, "docling_provenance")
        self.assertEqual(RETRIEVAL_EVAL.chunk_token_size, 250)
        self.assertIsNone(build_lightrag_kwargs()["default_llm_timeout"])

    def test_docling_parser_kwargs_use_low_memory_batches(self) -> None:
        parser_kwargs = build_parser_kwargs()
        self.assertLessEqual(parser_kwargs["images_scale"], 2.0)
        self.assertLessEqual(parser_kwargs["page_batch_size"], 4)
        self.assertLessEqual(parser_kwargs["ocr_batch_size"], 4)
        self.assertLessEqual(parser_kwargs["layout_batch_size"], 4)
        self.assertLessEqual(parser_kwargs["table_batch_size"], 4)
        self.assertLessEqual(parser_kwargs["queue_max_size"], 100)

    def test_docling_native_pages_are_preserved_for_tables_and_images(self) -> None:
        if not DOCLING_JSON.exists():
            self.skipTest(f"Fixture not found: {DOCLING_JSON}")
        document = json.loads(DOCLING_JSON.read_text(encoding="utf-8"))
        parser = DoclingProvenanceParser()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            texts = [
                parser.read_from_block(block, "texts", output_dir, 0, str(index))
                for index, block in enumerate(document["texts"])
            ]
            tables = [
                parser.read_from_block(block, "tables", output_dir, 0, str(index))
                for index, block in enumerate(document["tables"])
            ]
            images = [
                parser.read_from_block(block, "pictures", output_dir, 0, str(index))
                for index, block in enumerate(document["pictures"])
            ]
        self.assertEqual([item["page_number"] for item in tables], [3, 19, 20])
        self.assertEqual(
            [item["page_number"] for item in images],
            [4, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 17],
        )
        all_items = texts + tables + images
        self.assertEqual(min(item["page_number"] for item in all_items), 1)
        self.assertEqual(max(item["page_number"] for item in all_items), 23)
        for item in all_items:
            self.assertEqual(item["page_idx"], item["page_number"] - 1)

    def test_docling_leaf_groups_are_skipped_instead_of_fake_tables(self) -> None:
        parser = DoclingProvenanceParser()
        document = {
            "body": {"children": [{"$ref": "#/groups/0"}, {"$ref": "#/tables/0"}]},
            "groups": [
                {
                    "self_ref": "#/groups/0",
                    "children": [],
                    "content_layer": "body",
                    "name": "group",
                    "label": "form_area",
                }
            ],
            "tables": [
                {
                    "self_ref": "#/tables/0",
                    "children": [],
                    "label": "table",
                    "prov": [{"page_no": 7}],
                    "data": {"table_cells": [], "num_rows": 0, "num_cols": 0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            items = parser.read_from_block_recursive(
                document["body"], "body", Path(temp_dir), 0, "0", document
            )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "table")
        self.assertEqual(items[0]["page_number"], 7)

    def test_docling_memory_logs_are_captured_for_later_reruns(self) -> None:
        capture = DoclingParseIssueCapture()
        logger = logging.getLogger("docling.pipeline.standard_pdf_pipeline.test")
        old_level = logger.level
        old_propagate = logger.propagate
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(capture)
        try:
            logger.error(
                "Stage preprocess failed for run 1, pages [29]: std::bad_alloc"
            )
            logger.error(
                "Stage ocr failed for run 1: ONNXRuntimeError: bad allocation"
            )
        finally:
            logger.removeHandler(capture)
            logger.setLevel(old_level)
            logger.propagate = old_propagate

        summary = _summarize_docling_parse_issues(capture, status="parsed")
        self.assertEqual(summary["issue_count"], 2)
        self.assertIn("std::bad_alloc", summary["matched_patterns"])
        self.assertIn("bad allocation", summary["matched_patterns"])
        self.assertIn("onnxruntimeerror", summary["matched_patterns"])

    def test_250_token_chunking_keeps_standard_content_and_adds_pages(self) -> None:
        if not PARSE_CACHE.exists():
            self.skipTest(f"Fixture not found: {PARSE_CACHE}")
        cache = json.loads(PARSE_CACHE.read_text(encoding="utf-8"))
        parsed = next(iter(cache.values()))
        text_content, _ = separate_content(parsed["content_list"])
        tokenizer = TiktokenTokenizer("gpt-4o-mini")
        standard_1000 = chunking_by_token_size(
            tokenizer, text_content, chunk_overlap_token_size=100, chunk_token_size=1000
        )
        standard_250 = chunking_by_token_size(
            tokenizer, text_content, chunk_overlap_token_size=100, chunk_token_size=250
        )
        annotate = build_page_aware_chunking_func(
            text_content, [TextPageSpan(0, len(text_content), 1)]
        )
        annotated_250 = annotate(
            tokenizer,
            text_content,
            chunk_overlap_token_size=100,
            chunk_token_size=250,
        )

        self.assertEqual(len(standard_1000), 10)
        self.assertEqual(len(standard_250), 60)
        self.assertEqual(
            [chunk["content"] for chunk in standard_250],
            [chunk["content"] for chunk in annotated_250],
        )
        self.assertTrue(
            all(chunk["page_numbers"] == [1] for chunk in annotated_250)
        )

    def test_metrics_count_chunk_coverage_without_mislabeling_pages(self) -> None:
        chunks = [
            {
                "rank": 1,
                "chunk_id": "chunk-a",
                "content_type": "text",
                "page_numbers": [4, 5],
                "matches_evidence": True,
            },
            {
                "rank": 2,
                "chunk_id": "chunk-b",
                "content_type": "image",
                "page_numbers": [7],
                "matches_evidence": False,
            },
        ]
        metrics = _metrics_for_question(chunks, [5])
        pages = _ranked_pages(chunks, [5])
        self.assertTrue(metrics["hit_at_1"])
        self.assertEqual(metrics["first_hit_rank"], 1)
        self.assertFalse(pages[0]["matches_evidence"])
        self.assertTrue(pages[1]["matches_evidence"])


class ExtractionQualityTests(unittest.TestCase):
    def test_truncated_extraction_output_is_rejected(self) -> None:
        result = (
            "entity<|#|>Latinos<|#|>concept<|#|>Population described in report.\n"
            "relation<|#|>Latinos<|#|>Economic Mobility<|#"
        )
        analysis = analyze_extraction_output(result)
        self.assertFalse(analysis["valid"])
        self.assertIn("missing <|COMPLETE|>", analysis["issues"])
        self.assertEqual(analysis["relation_records"], 0)

    def test_complete_zero_relation_output_is_not_treated_as_malformed(self) -> None:
        result = (
            "entity<|#|>Pew Research Center<|#|>organization<|#|>"
            "Pew Research Center publishes reports.\n"
            "<|COMPLETE|>"
        )
        analysis = analyze_extraction_output(result)
        self.assertTrue(analysis["valid"])
        self.assertEqual(analysis["relation_records"], 0)
        self.assertEqual(EXTRACTION_QUALITY.base_num_predict, 4096)
        self.assertEqual(EXTRACTION_QUALITY.base_num_ctx, 8192)
        self.assertEqual(EXTRACTION_QUALITY.elevated_num_predict, 8192)
        self.assertEqual(EXTRACTION_QUALITY.elevated_num_ctx, 12288)
        self.assertEqual(EXTRACTION_QUALITY.attempt_timeout_s, 210)
        self.assertGreaterEqual(EXTRACTION_QUALITY.max_format_retries, 0)
        self.assertGreaterEqual(EXTRACTION_QUALITY.max_dropped_malformed_records, 0)
        self.assertGreaterEqual(EXTRACTION_QUALITY.max_elevated_retries, 0)
        self.assertGreaterEqual(EXTRACTION_QUALITY.max_empty_entity_chunks, 0)

    def test_only_missing_completion_is_distinguishable_from_truncation(self) -> None:
        result = (
            "entity<|#|>Gestalt<|#|>concept<|#|>Gestalt concerns perception.\n"
            "relation<|#|>Gestalt<|#|>Perception<|#|>theory<|#|>"
            "Gestalt is a theory of perception."
        )
        analysis = analyze_extraction_output(result)
        self.assertEqual(analysis["issues"], ["missing <|COMPLETE|>"])
        self.assertEqual(analysis["relation_records"], 1)


class ModelRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_automatic_gpu_placement_omits_num_gpu_option(self) -> None:
        automatic = LLMConfig(
            name="text",
            temperature=0.1,
            top_p=0.9,
            num_ctx=8192,
            num_batch=256,
            num_gpu=None,
            num_predict=1024,
        )
        forced = LLMConfig(
            name="text",
            temperature=0.1,
            top_p=0.9,
            num_ctx=8192,
            num_batch=256,
            num_gpu=4,
            num_predict=1024,
        )
        self.assertNotIn("num_gpu", automatic.options())
        self.assertEqual(forced.options()["num_gpu"], 4)

    def test_text_and_vision_model_blocks_are_independently_configurable(self) -> None:
        self.assertTrue(TEXT_LLM.name)
        self.assertEqual(VISION_LLM.name, "qwen2.5vl:7b")
        self.assertEqual(TEXT_LLM.num_ctx, 8192)
        self.assertEqual(VISION_LLM.num_ctx, 8192)
        self.assertIsNone(TEXT_LLM.num_gpu)
        self.assertIsNone(VISION_LLM.num_gpu)
        self.assertFalse(TEXT_LLM.think)
        self.assertIsNone(VISION_LLM.think)

    def test_manifest_validation_rejects_old_model_routing(self) -> None:
        manifest = model_manifest_fields()
        require_current_model_manifest(manifest, "Checkpoint")
        manifest["text_llm_model"] = "qwen2.5vl:7b"
        with self.assertRaisesRegex(RuntimeError, "text/vision model configuration"):
            require_current_model_manifest(manifest, "Checkpoint")

    async def test_text_completion_uses_text_model(self) -> None:
        mock_call = AsyncMock(return_value="answer")
        with patch("local_config._ollama_model_if_cache", mock_call):
            result = await _llm_call("prompt")

        self.assertEqual(result, "answer")
        self.assertEqual(mock_call.await_args.args[0], TEXT_LLM.name)

    async def test_structured_extraction_uses_text_model(self) -> None:
        response = MagicMock()
        response.__getitem__.return_value = {"content": "<|COMPLETE|>"}
        response.done_reason = "stop"
        response.eval_count = 1
        response.prompt_eval_count = 2
        client = MagicMock()
        client.chat = AsyncMock(return_value=response)
        client._client.aclose = AsyncMock()
        with patch("local_config.ollama.AsyncClient", return_value=client):
            await _extraction_llm_call_with_metadata(
                "prompt", num_ctx=8192, num_predict=4096
            )

        self.assertEqual(client.chat.await_args.kwargs["model"], TEXT_LLM.name)
        self.assertNotEqual(client.chat.await_args.kwargs["model"], VISION_LLM.name)
        self.assertFalse(client.chat.await_args.kwargs["think"])

    async def test_regular_text_completion_disables_configured_thinking(self) -> None:
        mock_call = AsyncMock(return_value="answer")
        with patch("local_config._ollama_model_if_cache", mock_call):
            await _llm_call("prompt")

        self.assertFalse(mock_call.await_args.kwargs["think"])

    async def test_image_completion_uses_vision_model(self) -> None:
        response = MagicMock()
        response.__getitem__.return_value = {"content": "image description"}
        client = MagicMock()
        client.chat = AsyncMock(return_value=response)
        client._client.aclose = AsyncMock()
        with patch("local_config.ollama.AsyncClient", return_value=client):
            result = await _vision_call("describe", image_data="base64")

        self.assertEqual(result, "image description")
        self.assertEqual(client.chat.await_args.kwargs["model"], VISION_LLM.name)

    async def test_vision_text_fallback_delegates_to_text_path(self) -> None:
        mock_call = AsyncMock(return_value="text answer")
        with patch("local_config._llm_call", mock_call):
            result = await _vision_call("text only")

        self.assertEqual(result, "text answer")
        mock_call.assert_awaited_once()


class ExtractionRetryTests(unittest.IsolatedAsyncioTestCase):
    extraction_prompt = "Extract entities and relationships from the input text"
    valid_result = (
        "entity<|#|>Gestalt<|#|>concept<|#|>Gestalt concerns perception.\n"
        "relation<|#|>Gestalt<|#|>Perception<|#|>theory<|#|>"
        "Gestalt is a theory of perception.\n"
        "<|COMPLETE|>"
    )
    malformed_result = (
        "entity<|#|>Gestalt<|#|>concept<|#|>Gestalt concerns perception.\n"
        "relation<|#|>Gestalt<|#|>Perception<|#"
    )

    async def asyncSetUp(self) -> None:
        self.quality_patch = patch(
            "local_config.EXTRACTION_QUALITY",
            replace(
                EXTRACTION_QUALITY,
                max_format_retries=2,
                max_dropped_malformed_records=1,
                max_elevated_retries=1,
            ),
        )
        self.quality_patch.start()
        self.addCleanup(self.quality_patch.stop)

    async def test_length_limited_output_uses_high_retry_budget(self) -> None:
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "length", 4096, 1800),
                (self.valid_result, "stop", 5000, 1900),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertEqual(result, self.valid_result)
        self.assertEqual(mock_call.await_count, 2)
        self.assertEqual(mock_call.await_args_list[0].kwargs["num_ctx"], 8192)
        self.assertEqual(mock_call.await_args_list[0].kwargs["num_predict"], 4096)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 12288)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 8192)
        self.assertEqual(stats.elevated_retry_chunks, 1)
        self.assertEqual(stats.elevated_retry_attempts, 1)
        self.assertEqual(stats.format_retry_chunks, 0)

    async def test_length_retry_amount_is_configurable(self) -> None:
        configured_quality = replace(EXTRACTION_QUALITY, max_elevated_retries=2)
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "length", 4096, 1800),
                (self.malformed_result, "length", 8192, 1900),
                (self.valid_result, "stop", 5000, 1900),
            ]
        )
        with (
            patch("local_config.EXTRACTION_QUALITY", configured_quality),
            patch("local_config._extraction_llm_call_with_metadata", mock_call),
        ):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertEqual(result, self.valid_result)
        self.assertEqual(mock_call.await_count, 3)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 12288)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 8192)
        self.assertEqual(mock_call.await_args_list[2].kwargs["num_ctx"], 12288)
        self.assertEqual(mock_call.await_args_list[2].kwargs["num_predict"], 8192)
        self.assertEqual(stats.elevated_retry_chunks, 1)
        self.assertEqual(stats.elevated_retry_attempts, 2)

    async def test_missing_marker_is_repaired_without_elevated_retry(self) -> None:
        stats = ExtractionQualityStats()
        without_marker = self.valid_result.replace("\n<|COMPLETE|>", "")
        mock_call = AsyncMock(return_value=(without_marker, "stop", 700, 1800))
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertTrue(result.endswith("<|COMPLETE|>"))
        self.assertEqual(mock_call.await_count, 1)
        self.assertEqual(stats.completion_marker_repairs, 1)
        self.assertEqual(stats.elevated_retry_chunks, 0)

    async def test_format_error_without_length_retries_at_base_resources(self) -> None:
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "stop", 900, 1800),
                (self.valid_result, "stop", 1000, 1900),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertEqual(result, self.valid_result)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 8192)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 4096)
        self.assertEqual(stats.format_retry_chunks, 1)
        self.assertEqual(stats.elevated_retry_chunks, 0)

    async def test_attempt_timeout_retries_at_base_resources(self) -> None:
        configured_quality = replace(
            EXTRACTION_QUALITY, attempt_timeout_s=0.001, max_format_retries=1
        )
        stats = ExtractionQualityStats()
        calls: list[dict[str, int]] = []

        async def fake_call(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                await asyncio.sleep(0.05)
                return self.valid_result, "stop", 1000, 1800
            return self.valid_result, "stop", 1000, 1900

        with (
            patch("local_config.EXTRACTION_QUALITY", configured_quality),
            patch("local_config._extraction_llm_call_with_metadata", fake_call),
        ):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertEqual(result, self.valid_result)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["num_ctx"], 8192)
        self.assertEqual(calls[1]["num_predict"], 4096)
        self.assertEqual(stats.timed_out_attempts, 1)
        self.assertEqual(stats.format_retry_chunks, 1)

    async def test_single_malformed_record_is_dropped_after_two_format_retries(
        self,
    ) -> None:
        stats = ExtractionQualityStats()
        almost_valid = self.valid_result.replace(
            "\n<|COMPLETE|>",
            "\nrelation<|#|>Broken<|#\n<|COMPLETE|>",
        )
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "stop", 900, 1800),
                (almost_valid, "stop", 1200, 1900),
                (self.malformed_result, "stop", 900, 1800),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertTrue(analyze_extraction_output(result)["valid"])
        self.assertNotIn("Broken", result)
        self.assertEqual(mock_call.await_count, 3)
        self.assertEqual(stats.format_retry_chunks, 1)
        self.assertEqual(stats.salvaged_output_chunks, 1)
        self.assertEqual(stats.discarded_malformed_records, 1)

    async def test_multiple_malformed_records_still_fail_after_format_retries(
        self,
    ) -> None:
        too_malformed = self.valid_result.replace(
            "\n<|COMPLETE|>",
            "\nrelation<|#|>Broken One<|#\n"
            "relation<|#|>Broken Two<|#\n<|COMPLETE|>",
        )
        mock_call = AsyncMock(
            side_effect=[
                (too_malformed, "stop", 1200, 1800),
                (too_malformed, "stop", 1200, 1800),
                (too_malformed, "stop", 1200, 1800),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            with self.assertRaisesRegex(RuntimeError, "format retries"):
                await _quality_checked_llm_call(self.extraction_prompt)

        self.assertEqual(mock_call.await_count, 3)

    async def test_format_retry_amount_is_configurable(self) -> None:
        configured_quality = replace(EXTRACTION_QUALITY, max_format_retries=3)
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "stop", 900, 1800),
                (self.malformed_result, "stop", 900, 1800),
                (self.malformed_result, "stop", 900, 1800),
                (self.valid_result, "stop", 1000, 1900),
            ]
        )
        with (
            patch("local_config.EXTRACTION_QUALITY", configured_quality),
            patch("local_config._extraction_llm_call_with_metadata", mock_call),
        ):
            result = await _quality_checked_llm_call(self.extraction_prompt)

        self.assertEqual(result, self.valid_result)
        self.assertEqual(mock_call.await_count, 4)

    async def test_complete_empty_extraction_is_accepted_after_format_retries(
        self,
    ) -> None:
        configured_quality = replace(
            EXTRACTION_QUALITY, max_format_retries=2, max_empty_entity_chunks=1
        )
        stats = ExtractionQualityStats()
        empty_result = "<|COMPLETE|>"
        mock_call = AsyncMock(
            side_effect=[
                (empty_result, "stop", 6, 1800),
                (empty_result, "stop", 6, 1900),
                (empty_result, "stop", 6, 1900),
            ]
        )
        with (
            patch("local_config.EXTRACTION_QUALITY", configured_quality),
            patch("local_config._extraction_llm_call_with_metadata", mock_call),
        ):
            result = await _quality_checked_llm_call(
                self.extraction_prompt, quality_stats=stats
            )

        self.assertEqual(result, empty_result)
        self.assertEqual(mock_call.await_count, 3)
        self.assertEqual(stats.format_retry_chunks, 1)
        self.assertEqual(stats.empty_entity_output_chunks, 1)

    async def test_complete_empty_extraction_limit_is_enforced(self) -> None:
        configured_quality = replace(
            EXTRACTION_QUALITY, max_format_retries=1, max_empty_entity_chunks=0
        )
        mock_call = AsyncMock(
            side_effect=[
                ("<|COMPLETE|>", "stop", 6, 1800),
                ("<|COMPLETE|>", "stop", 6, 1900),
            ]
        )
        with (
            patch("local_config.EXTRACTION_QUALITY", configured_quality),
            patch("local_config._extraction_llm_call_with_metadata", mock_call),
        ):
            with self.assertRaisesRegex(RuntimeError, "no valid entity records"):
                await _quality_checked_llm_call(self.extraction_prompt)

        self.assertEqual(mock_call.await_count, 2)

    async def test_high_retry_runner_failure_is_propagated(self) -> None:
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "length", 4096, 1800),
                RuntimeError("model runner has unexpectedly stopped"),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            with self.assertRaisesRegex(RuntimeError, "model runner"):
                await _quality_checked_llm_call(
                    self.extraction_prompt, quality_stats=stats
                )

        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 12288)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 8192)
        self.assertEqual(stats.elevated_retry_chunks, 1)
        self.assertEqual(stats.elevated_retry_attempts, 1)
        self.assertEqual(stats.format_retry_chunks, 0)


if __name__ == "__main__":
    unittest.main()
