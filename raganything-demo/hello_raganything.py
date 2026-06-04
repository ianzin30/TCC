"""
RAG-Anything runner with two modes:

  1. Demo mode (default): processes ./sample.pdf and asks one hard-coded
     question. Same flow as the original demo.

  2. Smoke-test mode (SMOKE_TEST=1 or --smoke): loads the MMLongBench-Doc
     parquet, selects the inclusive PDF_INDEX_RANGE configured in
     local_config.py, drops "Not answerable" rows, then indexes + queries
     each PDF and dumps results to JSONL.

Pick mode via env var or CLI flag:
    uv run python hello_raganything.py             # demo
    SMOKE_TEST=1 uv run python hello_raganything.py
    uv run python hello_raganything.py --smoke
"""

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import ollama
import pandas as pd

from raganything import RAGAnything

from local_config import (
    EMBED,
    OLLAMA_HOST,
    PDF_INDEX_RANGE,
    PARSER_OUTPUT_DIR,
    TEXT_LLM,
    VISION_LLM,
    build_embedding_func,
    build_lightrag_kwargs,
    build_llm_func,
    build_parser_kwargs,
    build_rag_config,
    build_vision_func,
)


# ---------------------------------------------------------------------------
# Demo-mode config
# ---------------------------------------------------------------------------

PDF_PATH = "sample.pdf"


# ---------------------------------------------------------------------------
# Smoke-test config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MMLONGBENCH_PARQUET = PROJECT_ROOT / "MMLongBench-Doc" / "data" / "train-00000-of-00001.parquet"
MMLONGBENCH_PDFS_DIR = PROJECT_ROOT / "MMLongBench-Doc" / "documents"

# Tag used to scope rag_storage + smoke_results so different architectures
# (RAG-Anything local Qwen+BGE, RAG-Anything OpenAI, ColQwen) don't collide.
ARCH_NAME = "ollama-qwen-bge"

# Dataset marker for impossible questions. Filtered out before querying.
UNANSWERABLE_MARKER = "Not answerable"

# Poll cadence while waiting for a long-running local model response.
QUERY_HEARTBEAT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Dataset helpers (standalone so the ColQwen runner can copy them verbatim)
# ---------------------------------------------------------------------------

def _load_questions(pdf_index_range: tuple[int, int]) -> pd.DataFrame:
    """Load answerable QA pairs for a one-based inclusive PDF index range."""
    df = pd.read_parquet(MMLONGBENCH_PARQUET)
    document_ids = df["doc_id"].drop_duplicates().reset_index(drop=True)

    if len(pdf_index_range) != 2:
        raise ValueError("PDF_INDEX_RANGE must contain exactly (start_index, end_index).")
    start_index, end_index = pdf_index_range
    if (
        not isinstance(start_index, int)
        or not isinstance(end_index, int)
        or start_index < 1
        or end_index < start_index
        or end_index > len(document_ids)
    ):
        raise ValueError(
            "PDF_INDEX_RANGE must be a valid one-based inclusive range; "
            f"received {pdf_index_range} for {len(document_ids)} PDF(s)."
        )

    selected_documents = document_ids.iloc[start_index - 1 : end_index].tolist()
    print(
        f"[grid] PDF_INDEX_RANGE={pdf_index_range} -> "
        f"{len(selected_documents)} PDF(s)",
        flush=True,
    )
    for index, doc_id in enumerate(selected_documents, start=start_index):
        print(f"[grid]   {index}: {doc_id}", flush=True)

    mask = df["doc_id"].isin(selected_documents) & (df["answer"] != UNANSWERABLE_MARKER)
    filtered = df.loc[mask].reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            f"No answerable questions found for PDF_INDEX_RANGE={pdf_index_range}, "
            f"documents={selected_documents}. All selected questions may be "
            f"marked '{UNANSWERABLE_MARKER}'."
        )

    return filtered


def _resolve_pdf_path(doc_id: str) -> Path:
    """Return the on-disk path for a doc_id; raises if the PDF is missing."""
    path = MMLONGBENCH_PDFS_DIR / doc_id
    if not path.exists():
        raise FileNotFoundError(f"PDF not present on disk: {path}")
    return path


def _arch_working_dir(doc_id: str) -> str:
    """Per-architecture, per-PDF rag_storage dir so reruns reuse caches."""
    return f"./rag_storage/{ARCH_NAME}/{Path(doc_id).stem}"


# ---------------------------------------------------------------------------
# Query observability
# ---------------------------------------------------------------------------

def _format_bytes(size: int | None) -> str:
    """Return Ollama byte counts in a compact human-readable form."""
    if size is None:
        return "unknown"
    gib = int(size) / (1024 ** 3)
    return f"{gib:.2f} GiB"


def _print_ollama_request() -> None:
    """Print the model request; process status later confirms GPU placement."""
    text_gpu = TEXT_LLM.num_gpu if TEXT_LLM.num_gpu is not None else "auto"
    vision_gpu = VISION_LLM.num_gpu if VISION_LLM.num_gpu is not None else "auto"
    print(
        f"[ollama] TEXT_LLM={TEXT_LLM.name}, num_ctx={TEXT_LLM.num_ctx}, "
        f"num_gpu={text_gpu}; VISION_LLM={VISION_LLM.name}, "
        f"num_ctx={VISION_LLM.num_ctx}, num_gpu={vision_gpu}; "
        f"EMBED={EMBED.name}, num_gpu={EMBED.num_gpu}. Actual GPU placement "
        "is confirmed once Ollama reports a loaded model.",
        flush=True,
    )


async def _ollama_runtime_status() -> str:
    """Return status for Ollama models currently loaded into memory."""
    client = ollama.AsyncClient(host=OLLAMA_HOST)
    try:
        response = await client.ps()
    finally:
        await client._client.aclose()

    if not response.models:
        return "model not loaded yet"

    statuses: list[str] = []
    for model in response.models:
        name = model.name or model.model or "unknown-model"
        size = int(model.size) if model.size is not None else None
        size_vram = int(model.size_vram) if model.size_vram is not None else None
        if size and size_vram is not None:
            gpu_percent = f"{(size_vram / size) * 100:.0f}% GPU"
        else:
            gpu_percent = "GPU allocation unknown"
        context = model.context_length if model.context_length is not None else "unknown"
        statuses.append(
            f"{name}: VRAM {_format_bytes(size_vram)} / {_format_bytes(size)} "
            f"({gpu_percent}), context={context}"
        )
    return "; ".join(statuses)


async def _query_heartbeat(label: str, started: float, query_task: asyncio.Task[str]) -> None:
    """Report elapsed time and Ollama placement while a query is pending."""
    while True:
        await asyncio.sleep(QUERY_HEARTBEAT_SECONDS)
        if query_task.done():
            return

        elapsed = time.perf_counter() - started
        try:
            runtime_status = await _ollama_runtime_status()
        except Exception as exc:
            runtime_status = f"Ollama status unavailable: {type(exc).__name__}: {exc}"
        print(f"{label} RUNNING ({elapsed:.1f}s) {runtime_status}", flush=True)


async def _aquery_with_progress(
    rag: RAGAnything,
    question: str,
    doc_id: str,
    question_number: int,
    question_total: int,
) -> str:
    """Run one query while printing immediate and periodic progress."""
    label = f"[query {question_number}/{question_total}]"
    print(f"{label} START doc={doc_id}: {question}", flush=True)

    started = time.perf_counter()
    query_task = asyncio.create_task(rag.aquery(question, mode="hybrid"))
    heartbeat_task = asyncio.create_task(_query_heartbeat(label, started, query_task))
    try:
        return await query_task
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

async def demo() -> None:
    rag = RAGAnything(
        config=build_rag_config(),
        llm_model_func=build_llm_func(),
        vision_model_func=build_vision_func(),
        embedding_func=build_embedding_func(),
        lightrag_kwargs=build_lightrag_kwargs(),
    )

    print("Processando documento...")
    await rag.process_document_complete(
        file_path=PDF_PATH,
        output_dir=PARSER_OUTPUT_DIR,
        parse_method="auto",
        **build_parser_kwargs(),
    )

    print("Documento processado.")
    print("Fazendo pergunta...")
    result = await _aquery_with_progress(
        rag,
        "What is this document mainly about?",
        PDF_PATH,
        1,
        1,
    )

    print("\nResposta:")
    print(result)


# ---------------------------------------------------------------------------
# Smoke-test mode
# ---------------------------------------------------------------------------

async def smoke_test() -> None:
    df = _load_questions(PDF_INDEX_RANGE)
    print(f"[grid] {df['doc_id'].nunique()} PDF(s), {len(df)} answerable questions")

    results: list[dict] = []
    question_number = 0

    for doc_id, group in df.groupby("doc_id", sort=False):
        try:
            pdf_path = _resolve_pdf_path(doc_id)
        except FileNotFoundError as exc:
            print(f"[skip] {doc_id}: {exc}")
            continue

        # Auto-purge any stale state from a previous failed run so the new
        # run actually indexes (instead of LightRAG dedupe-skipping it).
        # Smoke tests are meant to be reproducible from scratch each time;
        # production indexing should use its own persistent working dir.
        working_dir = _arch_working_dir(doc_id)
        if Path(working_dir).exists():
            shutil.rmtree(working_dir)
            print(f"[clean] purged stale {working_dir}", flush=True)

        cfg = build_rag_config(working_dir=working_dir)
        rag = RAGAnything(
            config=cfg,
            llm_model_func=build_llm_func(),
            vision_model_func=build_vision_func(),
            embedding_func=build_embedding_func(),
            lightrag_kwargs=build_lightrag_kwargs(),
        )

        print(f"[index] {doc_id}  ({len(group)} questions)")
        await rag.process_document_complete(
            file_path=str(pdf_path),
            output_dir=PARSER_OUTPUT_DIR,
            parse_method="auto",
            **build_parser_kwargs(),
        )

        for _, row in group.iterrows():
            question_number += 1
            t0 = time.perf_counter()
            try:
                model_ans = await _aquery_with_progress(
                    rag,
                    row["question"],
                    doc_id,
                    question_number,
                    len(df),
                )
                err = None
            except Exception as exc:
                model_ans, err = None, repr(exc)
            dt = round(time.perf_counter() - t0, 2)

            status = "ERR" if err else "OK "
            preview = (model_ans or err or "")[:80].replace("\n", " ")
            print(f"  [{status}] ({dt:>6.2f}s) {row['question'][:60]!r} -> {preview!r}")

            results.append({
                "arch": ARCH_NAME,
                "doc_id": doc_id,
                "question": row["question"],
                "ground_truth": row["answer"],
                "model_answer": model_ans,
                "evidence_sources": row.get("evidence_sources"),
                "evidence_pages": row.get("evidence_pages"),
                "answer_format": row.get("answer_format"),
                "duration_s": dt,
                "error": err,
            })

    out_dir = Path("./smoke_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ARCH_NAME}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    print(f"[done] {len(results)} answers -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    _print_ollama_request()
    if os.getenv("SMOKE_TEST") == "1" or "--smoke" in sys.argv:
        await smoke_test()
    else:
        await demo()


if __name__ == "__main__":
    asyncio.run(main())


# --- Previous OpenAI wiring, kept for reference ---
#
# import os
# from functools import partial
# from dotenv import load_dotenv
# from raganything import RAGAnythingConfig
# from lightrag.llm.openai import openai_complete_if_cache, openai_embed
# from lightrag.utils import EmbeddingFunc
#
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
#
# config = RAGAnythingConfig(
#     working_dir="./rag_storage",
#     parser="docling",
#     parse_method="auto",
#     enable_image_processing=True,
#     enable_table_processing=True,
#     enable_equation_processing=True,
# )
#
# def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
#     return openai_complete_if_cache(
#         "gpt-4o-mini",
#         prompt,
#         system_prompt=system_prompt,
#         history_messages=history_messages,
#         api_key=api_key,
#         **kwargs,
#     )
#
# def vision_model_func(prompt, system_prompt=None, history_messages=[],
#                       image_data=None, messages=None, **kwargs):
#     if messages:
#         return openai_complete_if_cache(
#             "gpt-4o-mini", "", messages=messages, api_key=api_key, **kwargs,
#         )
#     if image_data:
#         return openai_complete_if_cache(
#             "gpt-4o-mini", "",
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url",
#                      "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
#                 ],
#             }],
#             api_key=api_key, **kwargs,
#         )
#     return llm_model_func(prompt, system_prompt=system_prompt,
#                           history_messages=history_messages, **kwargs)
#
# embedding_func = EmbeddingFunc(
#     embedding_dim=1536,
#     max_token_size=8192,
#     func=partial(openai_embed.func, model="text-embedding-3-small", api_key=api_key),
# )
