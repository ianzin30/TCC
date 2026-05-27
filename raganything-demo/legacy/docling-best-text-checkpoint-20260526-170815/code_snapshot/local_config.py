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
    # 16384 fits comfortably (KV cache ~1 GB on disk + model ~5 GB + overhead
    # well under 10 GB free VRAM) and still leaves room for a 12k chunk plus
    # system prompt + response. If you want to push toward 32k:
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

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "num_batch": self.num_batch,
            "num_gpu": self.num_gpu,
            # Leave room for LightRAG's follow-up gleaning request in the
            # 8K context window while allowing structured extraction to finish.
            "num_predict": 1024,
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


def _normalize_ollama_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop OpenAI-only kwargs and translate keyword_extraction -> JSON format."""
    if kwargs.pop("keyword_extraction", False):
        kwargs.setdefault("format", "json")
    for k in _OPENAI_ONLY_KWARGS:
        kwargs.pop(k, None)
    return kwargs


async def _llm_call(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
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
        options=LLM.options(),
        **kwargs,
    )


def build_llm_func():
    """Return the callable RAG-Anything will use for text completions."""
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


def build_rag_config(working_dir: str = "./rag_storage") -> RAGAnythingConfig:
    """Parser + multimodal flags. Nothing model-related lives here.

    `working_dir` is scoped per-(architecture, PDF) by the smoke test so
    different runs don't share indexes. Defaults preserve demo behavior.
    """
    return RAGAnythingConfig(
        working_dir=working_dir,
        parser_output_dir=PARSER_OUTPUT_DIR,
        parser=PARSER_NAME,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )


def build_lightrag_kwargs() -> dict[str, Any]:
    """LightRAG init kwargs that carry the paper's token limits + batch knobs."""
    return {
        "chunk_token_size": RUNTIME.chunk_token_size,
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
