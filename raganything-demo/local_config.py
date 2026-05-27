"""
Central configuration for the RAG-Anything flow using local models via Ollama.

This module is the single source of truth for:
  - LLM / vision model identity and generation params (Qwen 2.5 VL 7B)
  - Embedding model identity and params (BGE-M3)
  - LightRAG runtime knobs (token limits, batch sizes, async concurrency)
  - Adapter functions that translate RAG-Anything's expected call signatures
    into the Ollama Python client's call signatures.

============================================================
RAG-Anything paper settings (Guo et al., 2025, arXiv:2510.12323)
============================================================
The values below are what the paper actually used in their experiments.
They are documented here so you can see, side-by-side, how the local
configuration deviates.

  Backbone LLM / Vision .... GPT-4o-mini (128K context)
  Embedding model .......... text-embedding-3-large, 3072-dim
  Reranker ................. bge-reranker-v2-m3
  Parser ................... MinerU
  Entity + relation tokens . 20,000 (combined)
  Chunk token limit ........ 12,000
  GPT-4o-mini baseline ..... up to 50 pages/doc, rendered at 144 dpi

Local replacements used here:
  LLM / Vision ............. qwen2.5vl:7b   (one Ollama tag covers both)
  Embedding ................ bge-m3:latest  (1024-dim, NOT 3072-dim)
  Reranker ................. (not wired; stub at bottom of file)
  Parser ................... docling

Deviations worth noting:
  * Embedding dimension drops from 3072 -> 1024. This propagates into
    LightRAG via `embedding_dim` and changes vector-store geometry, so do
    NOT mix indexes built with different embedding models.
  * Backbone is a 7B local VLM instead of GPT-4o-mini; expect slower
    indexing and a quality gap on long-context multimodal questions.
"""

# --- Tiktoken offline cache (must run before importing lightrag/raganything) -
# LightRAG instantiates a TiktokenTokenizer("gpt-4o-mini") in __post_init__
# just to count tokens during chunking -- it does NOT call the OpenAI API.
# tiktoken downloads the BPE vocab (~1.7 MB) from
# openaipublic.blob.core.windows.net the first time and caches it. Pin that
# cache to a project-local directory so the download is a one-shot event and
# every subsequent run stays offline. Run once on a network that can reach
# the blob; after that .tiktoken_cache/ holds everything needed.
import os
from pathlib import Path as _Path
_TIKTOKEN_CACHE = _Path(__file__).resolve().parent / ".tiktoken_cache"
_TIKTOKEN_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))
# ----------------------------------------------------------------------------

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional

import numpy as np
import ollama
from lightrag.llm.ollama import _ollama_model_if_cache, ollama_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnythingConfig


# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------

OLLAMA_HOST: str = "http://localhost:11434"
OLLAMA_TIMEOUT: int = 600  # seconds; local VLM calls can be slow on CPU


# ---------------------------------------------------------------------------
# LLM / Vision model (Qwen 2.5 VL 7B)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    # Ollama model tag. Quantization variants -- uncomment one to switch.
    name: str = "qwen2.5vl:7b"

    # name: str = "qwen2.5vl:7b-fp16"     # fp16   (~16 GB VRAM)
    # name: str = "qwen2.5vl:7b-q8_0"     # 8-bit  (~9  GB VRAM)
    # name: str = "qwen2.5vl:7b-q4_K_M"   # 4-bit  (~6  GB VRAM) # essa aqui e a normal são a mesma, na verdade isso deveria ter sido explicado, a padrão já é quantizada.

    # Generation params (low temperature -> stable graph extraction).
    temperature: float = 0.1
    top_p: float = 0.9

    # Ollama context window.
    # Paper uses 12k-token chunks + up to 20k graph tokens. Ideally we'd run
    # 32k, but the Ollama runner does a *pre-load* VRAM estimate that
    # includes the full KV cache; on a 12 GB GPU running Qwen 2.5 VL 7B Q4
    # it concludes that 32k won't fit and silently falls back to CPU+RAM
    # (which is what produced the "model requires more system memory" crash).
    # The stable base profile uses 8192; extraction-quality retries may request
    # a larger per-call window only after confirmed truncation. If you want to
    # push toward 32k:
    #   * stop ollama (tray -> Quit Ollama), then re-launch with
    #       $env:OLLAMA_FLASH_ATTENTION = "1"
    #       $env:OLLAMA_KV_CACHE_TYPE   = "q8_0"
    #     ollama serve
    #   These shrink the KV cache enough that Ollama will accept 32k on 12 GB.
    num_ctx: int = 8192

    # Prompt-eval batch size: how many tokens Ollama processes per forward
    # pass during prefill. Bigger = faster prefill, more VRAM.
    num_batch: int = 256

    # Number of model layers to offload to GPU. Ollama convention:
    #   -1   -> auto (let Ollama choose partial offload if VRAM is tight)
    #    0   -> CPU only
    #    N>0 -> exactly N layers on GPU
    # Ollama 0.24 reports qwen2.5vl:7b as 29 loadable layers. Request all
    # 29 layers for full GPU offload; if it cannot fit, fail visibly instead
    # of silently running partly on CPU.
    num_gpu: int = 29
    #37

    # Default answer/analysis output budget. Textual graph extraction uses a
    # larger, separately configured budget because its output is a long list
    # of structured entity and relation records.
    num_predict: int = 1024

    def options(
        self, *, num_ctx: int | None = None, num_predict: int | None = None
    ) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": num_ctx if num_ctx is not None else self.num_ctx,
            "num_batch": self.num_batch,
            "num_gpu": self.num_gpu,
            "num_predict": num_predict if num_predict is not None else self.num_predict,
        }


LLM = LLMConfig()


# ---------------------------------------------------------------------------
# Embedding model (BGE-M3)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    # Ollama model tag. Quantization variants -- uncomment one to switch.
    name: str = "bge-m3:latest"
    # name: str = "bge-m3:567m-fp16"      # fp16    (~1.2 GB)
    # name: str = "bge-m3:567m-q8_0"      # 8-bit   (~700 MB)
    # name: str = "bge-m3:567m-q4_K_M"    # 4-bit   (~400 MB)

    # BGE-M3 native dim is 1024.
    embedding_dim: int = 1024

    # BGE-M3 supports up to 8192 input tokens.
    max_token_size: int = 8192

    # Keep embeddings off the GPU so the Qwen VLM can retain its full
    # 29-layer offload on a 12 GB card. Embedding quality is unchanged;
    # only embedding throughput is traded for VRAM headroom.
    num_gpu: int = 0

    # Embedding batch size -> LightRAG's `embedding_batch_num`. Default
    # LightRAG is 10; 32 is a reasonable bump for local Ollama throughput.
    batch_size: int = 32

    # Concurrent embedding requests -> LightRAG's `embedding_func_max_async`.
    # Local Ollama serializes everything behind a single runner; high
    # concurrency just inflates per-call latency and risks hitting the
    # per-call timeout below. 2 is a comfortable ceiling for bge-m3 on a
    # single GPU; bump back up if you switch to a hosted embedding API.
    max_async: int = 2

    # Per-call timeout for an embedding request, in seconds. LightRAG's
    # default is 30 (DEFAULT_EMBEDDING_TIMEOUT in lightrag/constants.py),
    # and the worker-level timeout is derived as 2x this value. Local
    # Ollama cold-starts (loading bge-m3 + qwen2.5vl at the same time) can
    # easily exceed 30s on the first call. 180 gives plenty of headroom
    # without masking real hangs.
    request_timeout_s: int = 180


EMBED = EmbeddingConfig()


# ---------------------------------------------------------------------------
# LightRAG runtime knobs (mirror the paper's token limits)
# ---------------------------------------------------------------------------

@dataclass
class LightRAGRuntime:
    # Smaller local chunks keep extraction within Qwen's constrained 8K
    # context. The paper's 12K setting assumes a much larger hosted-model
    # context window.
    chunk_token_size: int = 1_000

    # Retrieval budgets must fit inside max_total_tokens below.
    max_entity_tokens: int = 2_000
    max_relation_tokens: int = 2_000

    # Reserve space in num_ctx=8192 for instructions and a generated answer.
    max_total_tokens: int = 5_000

    # Disable the optional correction pass while building local checkpoints:
    # repeated gleaning was slow and did not repair malformed extractions.
    entity_extract_max_gleaning: int = 0
    max_extract_input_tokens: int = 7_000

    # Concurrent LLM calls. Local Ollama serves one request at a time well;
    # keep this small to avoid thrashing.
    llm_model_max_async: int = 1


RUNTIME = LightRAGRuntime()


# ---------------------------------------------------------------------------
# Retrieval-by-page evaluation profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalEvaluationProfile:
    """Overrides used only by the checkpoint and retrieval evaluation flow."""

    parser_name: str = "docling_provenance"
    chunk_token_size: int = 400
    page_provenance: str = "docling_prov"


RETRIEVAL_EVAL = RetrievalEvaluationProfile()


# ---------------------------------------------------------------------------
# Text extraction quality profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionQualityProfile:
    """Controls selective validation for text checkpoint graph extraction."""

    # A 3072-token output truncated structured extraction for dense chunks.
    # When Ollama explicitly reports a length-limited response, increase only
    # the generation budget first, isolating it from context-window pressure.
    base_num_ctx: int = 8192
    base_num_predict: int = 4096
    elevated_num_ctx: int = 8192
    elevated_num_predict: int = 6144
    max_format_retries: int = 1
    max_elevated_retries: int = 1


EXTRACTION_QUALITY = ExtractionQualityProfile()


@dataclass
class ExtractionQualityStats:
    """Per-checkpoint counters for adaptive text extraction behavior."""

    initial_attempts: int = 0
    format_retry_chunks: int = 0
    elevated_retry_chunks: int = 0
    completion_marker_repairs: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "initial_attempts": self.initial_attempts,
            "format_retry_chunks": self.format_retry_chunks,
            "elevated_retry_chunks": self.elevated_retry_chunks,
            "completion_marker_repairs": self.completion_marker_repairs,
        }


# ---------------------------------------------------------------------------
# MMLongBench PDF selection
# ---------------------------------------------------------------------------

# One-based, inclusive positions among the distinct `doc_id` values in the
# parquet file. The previously processed document
# `PH_2016.06.08_Economy-Final.pdf` is index 1.
#
# Example: `(2, 5)` processes PDFs 2, 3, 4 and 5, including all of their
# answerable questions.
PDF_INDEX_RANGE: tuple[int, int] = (3, 3)


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

# Kwargs that LightRAG / RAG-Anything pass into the LLM hook assuming an
# OpenAI-compatible signature. The Ollama Python client rejects them with
# "AsyncClient.chat() got an unexpected keyword argument ...". We strip
# them here and translate the only one with a real Ollama equivalent
# (`keyword_extraction` -> `format="json"`), matching the behavior of the
# native `lightrag.llm.ollama.ollama_model_complete` wrapper.
_OPENAI_ONLY_KWARGS = (
    "enable_cot",
    "base_url",
    "api_key",
    "token_tracker",
    "use_azure",
    "azure_deployment",
    "api_version",
    "hashing_kv",
)

_ENTITY_EXTRACTION_TASK_MARKER = "Extract entities and relationships from the input text"
_EXTRACTION_TUPLE_DELIMITER = "<|#|>"
_EXTRACTION_COMPLETION_DELIMITER = "<|COMPLETE|>"


def _normalize_ollama_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop OpenAI-only kwargs and translate keyword_extraction -> JSON format."""
    if kwargs.pop("keyword_extraction", False):
        kwargs.setdefault("format", "json")
    for k in _OPENAI_ONLY_KWARGS:
        kwargs.pop(k, None)
    return kwargs


def analyze_extraction_output(result: str) -> dict[str, Any]:
    """Return structural quality indicators for a LightRAG extraction output."""
    entity_records = 0
    relation_records = 0
    malformed_records: list[str] = []
    for line in result.splitlines():
        record = line.strip()
        if record.startswith("entity"):
            if (
                record.startswith(f"entity{_EXTRACTION_TUPLE_DELIMITER}")
                and len(record.split(_EXTRACTION_TUPLE_DELIMITER)) == 4
            ):
                entity_records += 1
            else:
                malformed_records.append(record[:120])
        elif record.startswith("relation") or record.startswith("relationship"):
            if (
                (
                    record.startswith(f"relation{_EXTRACTION_TUPLE_DELIMITER}")
                    or record.startswith(
                        f"relationship{_EXTRACTION_TUPLE_DELIMITER}"
                    )
                )
                and len(record.split(_EXTRACTION_TUPLE_DELIMITER)) == 5
            ):
                relation_records += 1
            else:
                malformed_records.append(record[:120])

    issues: list[str] = []
    if _EXTRACTION_COMPLETION_DELIMITER not in result:
        issues.append(f"missing {_EXTRACTION_COMPLETION_DELIMITER}")
    if malformed_records:
        issues.append(f"{len(malformed_records)} malformed record(s)")
    if entity_records == 0:
        issues.append("no valid entity records")
    return {
        "valid": not issues,
        "issues": issues,
        "entity_records": entity_records,
        "relation_records": relation_records,
        "malformed_records": malformed_records,
    }


def _is_entity_extraction_prompt(prompt: str) -> bool:
    return _ENTITY_EXTRACTION_TASK_MARKER in prompt


async def _extraction_llm_call_with_metadata(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    *,
    num_ctx: int,
    num_predict: int,
    **kwargs: Any,
) -> tuple[str, str | None, int | None]:
    """Call Ollama for extraction while retaining its completion reason."""
    kwargs = _normalize_ollama_kwargs(kwargs)
    kwargs.pop("max_tokens", None)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
    try:
        response = await client.chat(
            model=LLM.name,
            messages=messages,
            options=LLM.options(num_ctx=num_ctx, num_predict=num_predict),
            **kwargs,
        )
        return (
            response["message"]["content"],
            response.done_reason,
            response.eval_count,
        )
    finally:
        try:
            await client._client.aclose()
        except Exception:
            pass


async def _llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    _num_ctx: int | None = None,
    _num_predict: int | None = None,
    **kwargs: Any,
) -> str:
    """Text-only call. Matches RAG-Anything's `llm_model_func` signature."""
    kwargs = _normalize_ollama_kwargs(kwargs)
    return await _ollama_model_if_cache(
        LLM.name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        host=OLLAMA_HOST,
        timeout=OLLAMA_TIMEOUT,
        options=LLM.options(num_ctx=_num_ctx, num_predict=_num_predict),
        **kwargs,
    )


async def _quality_checked_llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    quality_stats: ExtractionQualityStats | None = None,
    **kwargs: Any,
) -> str:
    """Validate and selectively retry structured entity/relation extraction."""
    if not _is_entity_extraction_prompt(prompt):
        return await _llm_call(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    stats = quality_stats or ExtractionQualityStats()

    def strengthened_prompt() -> str:
        return (
            prompt
            + "\n\n---Mandatory Output Validation---\n"
            "Repeat the full extraction from the source text. Your output "
            "must end with <|COMPLETE|>, and every entity or relation must "
            "be a complete single-line record using <|#|> fields. Do not "
            "stop midway through a record."
        )

    async def run_attempt(
        attempt_prompt: str, *, num_ctx: int, num_predict: int
    ) -> tuple[str, str | None, int | None, dict[str, Any]]:
        result, done_reason, eval_count = await _extraction_llm_call_with_metadata(
            attempt_prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            **kwargs,
        )
        analysis = analyze_extraction_output(result)
        return result, done_reason, eval_count, analysis

    def accept_or_repair(
        result: str, done_reason: str | None, eval_count: int | None, analysis: dict[str, Any]
    ) -> str | None:
        if analysis["valid"]:
            return result
        if (
            analysis["issues"] == [f"missing {_EXTRACTION_COMPLETION_DELIMITER}"]
            and done_reason != "length"
        ):
            stats.completion_marker_repairs += 1
            print(
                "[extract quality] repaired omitted completion marker on "
                f"otherwise valid output (done_reason={done_reason or 'unknown'}, "
                f"generated_tokens={eval_count or 'unknown'})",
                flush=True,
            )
            return result.rstrip() + "\n" + _EXTRACTION_COMPLETION_DELIMITER
        return None

    async def elevated_retry(trigger_reason: str) -> str:
        stats.elevated_retry_chunks += 1
        print(
            "[extract quality] retrying length-limited chunk with larger output "
            "budget only: "
            f"{trigger_reason}; retry resources "
            f"num_ctx={EXTRACTION_QUALITY.elevated_num_ctx}, "
            f"num_predict={EXTRACTION_QUALITY.elevated_num_predict}",
            flush=True,
        )
        result, done_reason, eval_count, analysis = await run_attempt(
            strengthened_prompt(),
            num_ctx=EXTRACTION_QUALITY.elevated_num_ctx,
            num_predict=EXTRACTION_QUALITY.elevated_num_predict,
        )
        accepted = accept_or_repair(result, done_reason, eval_count, analysis)
        if accepted is not None:
            return accepted
        raise RuntimeError(
            "Text extraction remained structurally invalid after elevated retry: "
            + ", ".join(analysis["issues"])
            + f" (done_reason={done_reason or 'unknown'}, "
            f"generated_tokens={eval_count or 'unknown'})"
        )

    stats.initial_attempts += 1
    result, done_reason, eval_count, analysis = await run_attempt(
        prompt,
        num_ctx=EXTRACTION_QUALITY.base_num_ctx,
        num_predict=EXTRACTION_QUALITY.base_num_predict,
    )
    accepted = accept_or_repair(result, done_reason, eval_count, analysis)
    if accepted is not None:
        return accepted
    if done_reason == "length":
        if EXTRACTION_QUALITY.max_elevated_retries > 0:
            return await elevated_retry(
                ", ".join(analysis["issues"])
                + f" (done_reason=length, generated_tokens={eval_count or 'unknown'})"
            )
        raise RuntimeError(
            "Text extraction hit its output limit and elevated retry is disabled: "
            + ", ".join(analysis["issues"])
        )

    if EXTRACTION_QUALITY.max_format_retries > 0:
        stats.format_retry_chunks += 1
        print(
            "[extract quality] retrying invalid structured output with base "
            "resources: "
            + ", ".join(analysis["issues"])
            + f" (done_reason={done_reason or 'unknown'}, "
            f"generated_tokens={eval_count or 'unknown'})",
            flush=True,
        )
        result, done_reason, eval_count, analysis = await run_attempt(
            strengthened_prompt(),
            num_ctx=EXTRACTION_QUALITY.base_num_ctx,
            num_predict=EXTRACTION_QUALITY.base_num_predict,
        )
        accepted = accept_or_repair(result, done_reason, eval_count, analysis)
        if accepted is not None:
            return accepted
        if done_reason == "length" and EXTRACTION_QUALITY.max_elevated_retries > 0:
            return await elevated_retry(
                ", ".join(analysis["issues"])
                + f" after format retry (done_reason=length, "
                f"generated_tokens={eval_count or 'unknown'})"
            )

    raise RuntimeError(
        "Text extraction remained structurally invalid after base retry: "
        + ", ".join(analysis["issues"])
        + f" (done_reason={done_reason or 'unknown'}, "
        f"generated_tokens={eval_count or 'unknown'})"
    )


def build_llm_func(
    *,
    enforce_extraction_quality: bool = False,
    quality_stats: ExtractionQualityStats | None = None,
):
    """Return the callable RAG-Anything will use for text completions."""
    if enforce_extraction_quality:
        stats = quality_stats or ExtractionQualityStats()

        async def quality_checked_call(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[list[dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> str:
            return await _quality_checked_llm_call(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                quality_stats=stats,
                **kwargs,
            )

        return quality_checked_call
    return _llm_call


def _extract_text_and_images_from_openai_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Translate OpenAI-style vision messages into Ollama's flat shape.

    RAG-Anything's VLM-enhanced query path (query.py) emits OpenAI-format
    messages where the user content is a list of {"type": "text"|"image_url", ...}
    blocks. Ollama's chat API instead takes plain-text `content` plus a
    parallel `images=[base64, ...]` list, so we flatten here.
    """
    ollama_messages: list[dict[str, Any]] = []
    trailing_images: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            ollama_messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        images: list[str] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                # Strip "data:image/<fmt>;base64," prefix if present.
                if "," in url:
                    images.append(url.split(",", 1)[1])
                else:
                    images.append(url)

        entry: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
        if images:
            entry["images"] = images
            trailing_images = images
        ollama_messages.append(entry)

    return ollama_messages, trailing_images


async def _vision_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    image_data: Optional[str] = None,
    messages: Optional[list[dict[str, Any]]] = None,
    **kwargs: Any,
) -> str:
    """Vision-or-text call. Matches RAG-Anything's `vision_model_func` signature.

    Three input shapes are handled (see raganything/query.py and
    raganything/modalprocessors.py):
      1. `messages=[...]` already formatted in OpenAI style -> translate to
         Ollama's chat shape and call ollama.AsyncClient.chat directly.
      2. `image_data=<base64>` -> single user turn with one image.
      3. text-only fallback -> delegate to the text LLM path.
    """
    # Strip OpenAI-only kwargs and translate keyword_extraction the same way
    # the text path does, so both vision shapes accept whatever LightRAG sends.
    kwargs = _normalize_ollama_kwargs(kwargs)

    # Shape 1: pre-formatted OpenAI messages (VLM enhanced retrieval path).
    if messages:
        ollama_messages, _ = _extract_text_and_images_from_openai_messages(messages)
        client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        try:
            response = await client.chat(
                model=LLM.name,
                messages=ollama_messages,
                options=LLM.options(),
                **kwargs,
            )
            return response["message"]["content"]
        finally:
            try:
                await client._client.aclose()
            except Exception:
                pass

    # Shape 2: explicit base64 image (multimodal processors path).
    if image_data:
        user_msg: dict[str, Any] = {
            "role": "user",
            "content": prompt,
            "images": [image_data],
        }
        chat_messages: list[dict[str, Any]] = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            chat_messages.extend(history_messages)
        chat_messages.append(user_msg)

        client = ollama.AsyncClient(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        try:
            response = await client.chat(
                model=LLM.name,
                messages=chat_messages,
                options=LLM.options(),
                **kwargs,
            )
            return response["message"]["content"]
        finally:
            try:
                await client._client.aclose()
            except Exception:
                pass

    # Shape 3: no image, fall through to the text path.
    return await _llm_call(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def build_vision_func():
    """Return the callable RAG-Anything will use for vision + text completions."""
    return _vision_call


def _sanitize_embedding_inputs(texts: list[str]) -> list[str]:
    """Replace empty / whitespace-only strings with a single space.

    bge-m3 (especially the quantized variant on Ollama) can produce NaN
    vectors when fed empty or all-whitespace input. A single space yields
    a finite vector with neutral semantics, which downstream cosine
    similarity tolerates.
    """
    return [t if (t and t.strip()) else " " for t in texts]


def build_embedding_func() -> EmbeddingFunc:
    """Return the EmbeddingFunc RAG-Anything will pass to LightRAG.

    Wraps the bare ollama_embed in two defensive layers:
      1. Input sanitization (empty -> single space).
      2. Output NaN/Inf clamping (poisons JSON serialization downstream;
         was the root cause of `failed to encode response: json: unsupported
         value: NaN` errors observed during entity upsert).
    """
    inner = partial(
        ollama_embed.func,
        embed_model=EMBED.name,
        host=OLLAMA_HOST,
        options={"num_gpu": EMBED.num_gpu},
    )

    async def safe_embed(texts, **kwargs):
        sanitized = _sanitize_embedding_inputs(list(texts))
        vectors = await inner(sanitized, **kwargs)
        arr = np.asarray(vectors, dtype=np.float32)
        bad_rows = ~np.isfinite(arr).all(axis=1)
        if bad_rows.any():
            patched_idxs = np.flatnonzero(bad_rows).tolist()
            print(
                f"[embed] WARNING: clamped NaN/Inf in {len(patched_idxs)} "
                f"embedding row(s) (idx={patched_idxs}); inputs may be too "
                f"short or unusual.",
                flush=True,
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    return EmbeddingFunc(
        embedding_dim=EMBED.embedding_dim,
        max_token_size=EMBED.max_token_size,
        func=safe_embed,
    )


# ---------------------------------------------------------------------------
# RAGAnything + LightRAG init helpers
# ---------------------------------------------------------------------------

PARSER_NAME: str = "docling"
PARSER_OUTPUT_DIR: str = "./output"


def build_parser_kwargs() -> dict[str, Any]:
    """Return additional parser invocation settings shared by every runner."""
    return {}


def build_rag_config(
    working_dir: str = "./rag_storage", *, parser_name: str | None = None
) -> RAGAnythingConfig:
    """Parser + multimodal flags. Nothing model-related lives here.

    `working_dir` is scoped per-(architecture, PDF) by the smoke test so
    different runs don't share indexes. Defaults preserve demo behavior.
    """
    return RAGAnythingConfig(
        working_dir=working_dir,
        parser_output_dir=PARSER_OUTPUT_DIR,
        parser=parser_name or PARSER_NAME,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )


def build_lightrag_kwargs(*, chunk_token_size: int | None = None) -> dict[str, Any]:
    """LightRAG init kwargs that carry the paper's token limits + batch knobs."""
    return {
        "chunk_token_size": chunk_token_size or RUNTIME.chunk_token_size,
        "max_entity_tokens": RUNTIME.max_entity_tokens,
        "max_relation_tokens": RUNTIME.max_relation_tokens,
        "max_total_tokens": RUNTIME.max_total_tokens,
        "entity_extract_max_gleaning": RUNTIME.entity_extract_max_gleaning,
        "max_extract_input_tokens": RUNTIME.max_extract_input_tokens,
        "embedding_batch_num": EMBED.batch_size,
        "embedding_func_max_async": EMBED.max_async,
        "default_embedding_timeout": EMBED.request_timeout_s,
        "llm_model_max_async": RUNTIME.llm_model_max_async,
        # Per-call LLM timeout, in seconds. LightRAG's default is 180
        # (DEFAULT_LLM_TIMEOUT). For local Ollama with a cold-started 7B
        # VLM and long prompts (12k-token chunks + system prompt + image
        # tokens), 300 gives headroom for the first call without realistically
        # slowing a healthy run. Worker timeout is derived as 2x this value.
        "default_llm_timeout": 300,
    }


# ---------------------------------------------------------------------------
# Reranker stub (paper used bge-reranker-v2-m3; see ablation Table 4)
# ---------------------------------------------------------------------------
# When you want to add the reranker, wire something like:
#
#   from FlagEmbedding import FlagReranker
#   _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
#
#   async def rerank_model_func(query, documents, top_n=None, **_):
#       pairs = [[query, d["content"]] for d in documents]
#       scores = _reranker.compute_score(pairs, normalize=True)
#       ranked = sorted(zip(documents, scores), key=lambda x: -x[1])
#       if top_n: ranked = ranked[:top_n]
#       return [{**d, "rerank_score": s} for d, s in ranked]
#
# then add `"rerank_model_func": rerank_model_func` to build_lightrag_kwargs().
