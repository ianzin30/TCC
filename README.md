# TCC — Recuperação Multimodal de Documentos (Retrieval sobre MMLongBench-Doc)

Este repositório reúne o código experimental do meu Trabalho de Conclusão de Curso, cujo objetivo é comparar diferentes estratégias de **recuperação de informação (retrieval)** em documentos longos e multimodais (texto, tabelas, gráficos e imagens), avaliadas sobre o benchmark [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc).

A pergunta central do trabalho é: **dada uma pergunta sobre um PDF longo, o sistema consegue recuperar as páginas corretas de evidência?** A avaliação é feita no nível de página, comparando o ranking produzido por cada pipeline com a coluna `evidence_pages` do dataset, usando as métricas **MRR**, **Hit@k** e **Evidence Page Recall@k** para `k ∈ {1, 3, 5, 10, 20}`.

## Abordagens comparadas

O repositório é dividido em três subprojetos independentes (cada um com seu próprio ambiente `uv`), correspondendo a três estratégias distintas de retrieval:

| Subprojeto | Estratégia | Stack principal |
|---|---|---|
| [`colqwen/`](colqwen) | Retrieval visual "page-as-image" com late interaction (sem parsing textual) | ColPali / ColQwen2.5 (`vidore/colqwen2.5-v0.2`) |
| [`docling-langchain/`](docling-langchain) | Baseline puramente textual: parsing + embeddings densos + vetor store | Docling, LangChain, BGE-M3, Milvus Lite |
| [`raganything-demo/`](raganything-demo) | RAG multimodal com grafo de conhecimento, adaptado para execução 100% local | RAG-Anything, LightRAG, Docling, Ollama (Qwen), BGE-M3, reranker `bge-reranker-v2-m3` |

### `colqwen/` — Retrieval visual (late interaction)

Renderiza cada página do PDF como imagem e usa modelos da família ColPali/ColQwen (via `colpali-engine`) para gerar embeddings multivetoriais de página, comparando-os diretamente com o embedding da pergunta — sem qualquer etapa de parsing ou OCR. Contém scripts de avaliação para várias variantes do modelo (`colqwen2.5`, `colqwen2.5-v0.2`, `colpali-v1.1`, `colsmol-256m`, `colsmol-500m`) e os resultados brutos em `smoke_results/`.

### `docling-langchain/` — Baseline textual

Faz parsing dos PDFs com [Docling](https://github.com/docling-project/docling), divide o conteúdo em chunks, gera embeddings densos com `BAAI/bge-m3` e indexa em um vetor store Milvus Lite via LangChain. Serve como baseline "texto puro" para comparação com as abordagens multimodais.

### `raganything-demo/` — RAG multimodal com grafo (adaptação local do RAG-Anything)

Reimplementa a arquitetura do artigo [RAG-Anything (Guo et al., 2025)](https://arxiv.org/abs/2510.12323) para rodar inteiramente em hardware local, sem depender de APIs pagas:

- **Parsing:** Docling (com pós-processamento próprio para compactar tabelas em Markdown e evitar prompts excessivos).
- **Modelos:** `qwen3:8b` (texto) e `qwen2.5vl:7b` (visão) via Ollama, no lugar do GPT-4o-mini do artigo original.
- **Embeddings:** `bge-m3:latest` (1024-d) no lugar do `text-embedding-3-large` (3072-d).
- **Grafo de conhecimento + recuperação híbrida:** LightRAG, combinando chunks textuais e multimodais.
- **Reranking (opcional):** `BAAI/bge-reranker-v2-m3`, aplicado sobre os top-20 candidatos recuperados.
- **Pipeline dividido em checkpoints** (textual → multimodal) para permitir reprocessamento parcial sem refazer o grafo inteiro.

As decisões de adaptação, limitações encontradas (ex.: falhas do runner do Ollama ao gerar respostas com muitas imagens) e a mudança de foco avaliativo — de "geração de resposta final" para "qualidade do retrieval por página" — estão documentadas em detalhe em [`raganything-demo/mudancas.md`](raganything-demo/mudancas.md).

## Estrutura do repositório

```
TCC/
├── colqwen/              # Retrieval visual com ColPali/ColQwen (uv project próprio)
├── docling-langchain/     # Baseline textual: Docling + LangChain + BGE-M3 + Milvus
├── raganything-demo/      # RAG multimodal com grafo (RAG-Anything adaptado, local)
└── MMLongBench-Doc/       # Dataset (não versionado, ver "Dados" abaixo)
```

Cada subprojeto contém seu próprio `pyproject.toml`/`uv.lock`, scripts `evaluate_*_pages.py` para rodar a avaliação de retrieval e um diretório `smoke_results/` com as saídas (`.jsonl`) já produzidas.

## Configuração experimental (resumo)

- **Recorte avaliado:** 30 documentos e 172 perguntas respondíveis do MMLongBench-Doc.
- **Unidade de avaliação:** página do documento, comparada com `evidence_pages`.
- **Métricas:** MRR, Hit@k e Evidence Page Recall@k, para `k = 1, 3, 5, 10, 20`.

## Como rodar

Cada subprojeto usa [`uv`](https://docs.astral.sh/uv/) para gerenciar dependências e é executado isoladamente:

```bash
cd colqwen
uv sync
uv run python evaluate_colqwen2_5_v0_2_pages.py
```

```bash
cd docling-langchain
uv sync
uv run python evaluate_docling_langchain_pages.py
```

```bash
cd raganything-demo
uv sync
uv run python smoke_text_checkpoint.py            # 1. constrói o checkpoint textual
uv run python smoke_multimodal_from_checkpoint.py  # 2. processa conteúdo multimodal
uv run python evaluate_retrieval_pages.py          # 3. avalia o retrieval por página
```

O pipeline `raganything-demo` depende de um servidor [Ollama](https://ollama.com) local com os modelos `qwen3:8b` e `qwen2.5vl:7b` disponíveis; veja os comentários de configuração em [`raganything-demo/local_config.py`](raganything-demo/local_config.py).

### Dados

Os experimentos usam o dataset [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) (parquet + PDFs), que **não é versionado** neste repositório (ver `.gitignore`) por ser um conjunto de dados de terceiros. Baixe-o separadamente e posicione-o em `MMLongBench-Doc/` na raiz do projeto, seguindo a estrutura esperada pelos `local_config.py` de cada subprojeto.

## Status

Trabalho em andamento. O texto do TCC (capítulos em `.tex`) está sendo escrito em paralelo ao código, em [`raganything-demo/raganything.tex`](raganything-demo/raganything.tex) e [`docling-langchain/docling.tex`](docling-langchain/docling.tex).
