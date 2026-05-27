"""Fast validation tests for page-aware retrieval preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from raganything.utils import separate_content

from local_config import (
    EXTRACTION_QUALITY,
    ExtractionQualityStats,
    PARSER_NAME,
    RETRIEVAL_EVAL,
    RUNTIME,
    _quality_checked_llm_call,
    analyze_extraction_output,
)
from evaluate_retrieval_pages import _metrics_for_question, _ranked_pages
from retrieval_provenance import (
    DoclingProvenanceParser,
    TextPageSpan,
    build_page_aware_chunking_func,
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
        self.assertEqual(RETRIEVAL_EVAL.chunk_token_size, 400)

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

    def test_400_token_chunking_keeps_standard_content_and_adds_pages(self) -> None:
        if not PARSE_CACHE.exists():
            self.skipTest(f"Fixture not found: {PARSE_CACHE}")
        cache = json.loads(PARSE_CACHE.read_text(encoding="utf-8"))
        parsed = next(iter(cache.values()))
        text_content, _ = separate_content(parsed["content_list"])
        tokenizer = TiktokenTokenizer("gpt-4o-mini")
        standard_1000 = chunking_by_token_size(
            tokenizer, text_content, chunk_overlap_token_size=100, chunk_token_size=1000
        )
        standard_400 = chunking_by_token_size(
            tokenizer, text_content, chunk_overlap_token_size=100, chunk_token_size=400
        )
        annotate = build_page_aware_chunking_func(
            text_content, [TextPageSpan(0, len(text_content), 1)]
        )
        annotated_400 = annotate(
            tokenizer,
            text_content,
            chunk_overlap_token_size=100,
            chunk_token_size=400,
        )

        self.assertEqual(len(standard_1000), 10)
        self.assertEqual(len(standard_400), 30)
        self.assertEqual(
            [chunk["content"] for chunk in standard_400],
            [chunk["content"] for chunk in annotated_400],
        )
        self.assertTrue(
            all(chunk["page_numbers"] == [1] for chunk in annotated_400)
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
        self.assertEqual(EXTRACTION_QUALITY.elevated_num_predict, 6144)
        self.assertEqual(EXTRACTION_QUALITY.elevated_num_ctx, 8192)

    def test_only_missing_completion_is_distinguishable_from_truncation(self) -> None:
        result = (
            "entity<|#|>Gestalt<|#|>concept<|#|>Gestalt concerns perception.\n"
            "relation<|#|>Gestalt<|#|>Perception<|#|>theory<|#|>"
            "Gestalt is a theory of perception."
        )
        analysis = analyze_extraction_output(result)
        self.assertEqual(analysis["issues"], ["missing <|COMPLETE|>"])
        self.assertEqual(analysis["relation_records"], 1)


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

    async def test_length_limited_output_increases_only_generation_budget(self) -> None:
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "length", 4096),
                (self.valid_result, "stop", 5000),
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
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 8192)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 6144)
        self.assertEqual(stats.elevated_retry_chunks, 1)
        self.assertEqual(stats.format_retry_chunks, 0)

    async def test_missing_marker_is_repaired_without_elevated_retry(self) -> None:
        stats = ExtractionQualityStats()
        without_marker = self.valid_result.replace("\n<|COMPLETE|>", "")
        mock_call = AsyncMock(return_value=(without_marker, "stop", 700))
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
                (self.malformed_result, "stop", 900),
                (self.valid_result, "stop", 1000),
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

    async def test_elevated_runner_failure_is_propagated(self) -> None:
        stats = ExtractionQualityStats()
        mock_call = AsyncMock(
            side_effect=[
                (self.malformed_result, "length", 4096),
                RuntimeError("model runner has unexpectedly stopped"),
            ]
        )
        with patch("local_config._extraction_llm_call_with_metadata", mock_call):
            with self.assertRaisesRegex(RuntimeError, "model runner"):
                await _quality_checked_llm_call(
                    self.extraction_prompt, quality_stats=stats
                )

        self.assertEqual(mock_call.await_args_list[1].kwargs["num_ctx"], 8192)
        self.assertEqual(mock_call.await_args_list[1].kwargs["num_predict"], 6144)
        self.assertEqual(stats.elevated_retry_chunks, 1)


if __name__ == "__main__":
    unittest.main()
