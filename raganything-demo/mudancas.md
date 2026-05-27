**Esboço Para Documentação No TCC**

A implementação original descrita no artigo [RAG-ANYTHING.pdf](</c:/Users/Ian Gabriel/Downloads/RAG-ANYTHING.pdf>) utiliza uma arquitetura multimodal baseada em extração de conteúdo, construção de grafo textual/visual, recuperação híbrida e geração final de respostas. Nos experimentos do artigo, foram utilizados:

| Componente | Configuração do artigo |
|---|---|
| Parser documental | MinerU |
| LLM/VLM principal | GPT-4o-mini |
| Embeddings | `text-embedding-3-large`, dimensão `3072` |
| Reranker | `bge-reranker-v2-m3` |
| Limite combinado de entidades e relações | `20.000` tokens |
| Limite de chunks | `12.000` tokens |
| Benchmarks | DocBench e MMLongBench |

A implementação desenvolvida no experimento local preserva a ideia geral do RAG-Anything, mas foi adaptada para execução local com Ollama, limitação de memória de GPU e avaliação específica de retrieval no MMLongBench-Doc.

## 1. Substituição Dos Modelos Proprietários Por Modelos Locais

| Elemento | Artigo original | Implementação local |
|---|---|---|
| LLM/VLM | GPT-4o-mini | `qwen2.5vl:7b` via Ollama |
| Embeddings | `text-embedding-3-large`, 3072 dimensões | `bge-m3:latest`, 1024 dimensões |
| Execução | API externa | Máquina local com GPU limitada |

A motivação principal foi executar todo o pipeline localmente, sem depender de APIs pagas ou serviços externos. O modelo `qwen2.5vl:7b` foi escolhido por suportar texto e visão, permitindo processar gráficos e imagens extraídas do PDF.

Durante os testes, também foi avaliada a versão `qwen2.5vl:3b`, buscando reduzir consumo de memória. Entretanto, os erros iniciais envolvendo tabelas também ocorreram com o modelo menor, indicando que o problema não era exclusivamente causado pelo tamanho do modelo.

O `bge-m3` foi adotado para embeddings por permitir execução local. Como sua utilização em GPU concorria por VRAM com o Qwen, ele foi configurado para executar na CPU:

```python
bge-m3:latest
embedding_dim = 1024
num_gpu = 0
```

A motivação foi liberar memória da GPU para o modelo multimodal, que era o componente mais exigente do pipeline.

## 2. Redução Dos Limites De Contexto E De Extração

No artigo, o sistema utiliza limites consideravelmente maiores:

```text
entity + relation tokens: 20.000
chunk tokens: 12.000
```

Na execução local, esses valores se mostraram incompatíveis com o contexto e a memória disponíveis para o Qwen via Ollama. A configuração atual foi reduzida em [local_config.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/local_config.py:188>):

| Parâmetro | Valor local |
|---|---:|
| `chunk_token_size` | `1000` |
| `max_entity_tokens` | `2000` |
| `max_relation_tokens` | `2000` |
| `max_total_tokens` | `5000` |
| `max_extract_input_tokens` | `7000` |
| `num_ctx` do Qwen | `8192` |

A motivação foi evitar prompts excessivamente grandes, timeouts e crashes do Ollama, mantendo espaço no contexto para instruções e geração.

Essa alteração reduz o custo computacional, mas também pode reduzir a quantidade de informação disponível em cada consulta ou extração. Portanto, representa uma adaptação necessária ao ambiente local, e não uma reprodução exata das condições do artigo.

## 3. Uso Do Qwen Totalmente Na GPU E BGE Na CPU

Inicialmente, houve investigação sobre a divisão do Qwen entre CPU e GPU, pois a execução estava extremamente lenta. O modelo visual foi configurado para utilizar suas camadas na GPU:

```python
qwen2.5vl:7b
num_gpu = 29
```

Durante as consultas finais, o Ollama reportou:

```text
qwen2.5vl:7b: VRAM 14.31 GiB / 14.31 GiB, 100% GPU
bge-m3:latest: VRAM 0.00 GiB / 1.06 GiB, 0% GPU
```

A motivação dessa distribuição foi utilizar a GPU para a etapa mais custosa, a inferência multimodal, mantendo o modelo de embeddings na CPU.

Apesar disso, a geração final de respostas ainda falhou quando muitos elementos visuais eram enviados simultaneamente ao Qwen. Isso demonstra que utilizar o modelo inteiramente na GPU melhora a velocidade potencial, mas não resolve sozinho os limites de memória/contexto.

## 4. Investigação Entre MinerU E Docling

O artigo utiliza **MinerU** para parsing dos documentos. Inicialmente, a implementação local utilizava **Docling**, mas foi realizado um experimento com MinerU em modo `pipeline` para aproximar o ambiente daquele descrito no artigo.

### Resultado Com MinerU

O MinerU processou o PDF com sucesso, produzindo:

```text
239 blocos
153 blocos textuais
14 charts
2 tabelas
86 itens considerados multimodais pelo fluxo atual
```

Entretanto, houve dois problemas práticos:

1. Elementos como `header`, `footer`, `page_number` e `page_footnote` também foram encaminhados como multimodais pelo pipeline atual.
2. O grafo textual resultante foi menor e aparentemente menos informativo do que o gerado com Docling.

Um dos resultados com MinerU produziu:

```text
84 nós
42 relações
```

### Resultado Com Docling

Com Docling, o mesmo documento produziu:

```text
635 blocos
620 blocos textuais
12 imagens
3 tabelas
15 itens multimodais
```

O melhor checkpoint textual gerado com Docling apresentou:

```text
128 nós
69 relações
```

Por esse motivo, decidiu-se retornar ao Docling. Embora MinerU corresponda à configuração declarada no artigo, no ambiente e na integração atuais o Docling apresentou melhor separação operacional entre texto e conteúdo visual e produziu um grafo textual mais completo.

Essa escolha deve ser descrita como uma decisão experimental baseada em comportamento observado, e não como superioridade geral de um parser sobre o outro.

## 5. Identificação Do Problema De Tabelas Do Docling

Ao iniciar a fase multimodal com Docling, as imagens funcionavam em vários casos, mas as tabelas provocavam erros imediatos no Ollama:

```text
model runner has unexpectedly stopped
status code: 500
```

Inicialmente, suspeitou-se de falta de VRAM ou de tamanho excessivo do Qwen. Contudo, o mesmo erro ocorreu com o modelo de `3b`.

A investigação mostrou que o problema estava na integração entre a estrutura de tabela retornada pelo Docling e o RAG-Anything:

- O Docling retorna tabelas contendo texto das células e metadados extensos de layout.
- Esses metadados incluem posições, bounding boxes e spans.
- O RAG-Anything convertia toda essa estrutura diretamente para texto.
- Esse conteúdo bruto era enviado ao modelo como prompt.

Os tamanhos observados foram:

| Tabela | Representação bruta | Conteúdo textual necessário |
|---|---:|---:|
| 1 | `20.144` caracteres | `730` caracteres |
| 2 | `11.019` caracteres | `380` caracteres |
| 3 | `20.256` caracteres | `400` caracteres |

O Ollama chegou a registrar truncamento de prompts e falha interna do runner. Portanto, o erro não indicava que as tabelas fossem incompatíveis com o Qwen; indicava que elas estavam sendo enviadas em uma representação excessivamente detalhada e inadequada para inferência.

## 6. Pós-Processamento Das Tabelas Do Docling

Para corrigir o problema sem abandonar o Docling, foi implementada uma etapa de normalização em [content_postprocessing.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/content_postprocessing.py:33>).

Essa etapa:

1. Detecta itens do tipo `table`.
2. Lê somente a grade textual das células.
3. Remove os metadados geométricos e estruturais desnecessários.
4. Converte a tabela para Markdown compacto.
5. Entrega essa versão reduzida à fase multimodal.

A normalização foi aplicada nos runners:

- [smoke_multimodal_from_checkpoint.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/smoke_multimodal_from_checkpoint.py:108>)
- [smoke_multimodal_probe.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/smoke_multimodal_probe.py:114>)

### Validação Da Correção

Antes da correção, a primeira tabela causava crash imediato do modelo.

Após a correção, a mesma tabela foi processada com sucesso:

```text
table processing complete:
Pew Research Center Survey on Latino Financial Confidence (table)

Chunk 1 of 1 extracted 6 Ent + 4 Rel
[probe SUCCESS] table #1: added_chunks=1
```

O grafo isolado do teste passou de:

```text
128 nós, 69 relações
```

para:

```text
131 nós, 76 relações
```

Essa alteração constitui uma adaptação importante da implementação: o parser continua sendo Docling, mas a saída tabular é convertida para um formato textual apropriado antes de ser enviada ao módulo multimodal.

## 7. Separação Do Pipeline Em Checkpoints

O fluxo original executava em sequência:

```text
parsing
 -> extração textual
 -> construção do grafo textual
 -> processamento multimodal
 -> consultas
 -> geração de respostas
```

Como cada tentativa completa era demorada e erros multimodais obrigavam a repetir etapas já concluídas, o pipeline foi dividido em partes reutilizáveis.

### Checkpoint Textual

Foi criado [smoke_text_checkpoint.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/smoke_text_checkpoint.py>), responsável por:

- processar o PDF;
- inserir apenas o conteúdo textual;
- construir o grafo textual;
- salvar esse estado antes do processamento multimodal.

O melhor checkpoint Docling salvo possui:

```text
128 nós
69 relações
10 chunks textuais
15 itens multimodais adiados
```

### Processamento Multimodal A Partir Do Checkpoint

Foi criado [smoke_multimodal_from_checkpoint.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/smoke_multimodal_from_checkpoint.py>), responsável por:

- copiar o checkpoint textual;
- processar apenas os itens multimodais;
- preservar o checkpoint textual original;
- permitir novas tentativas sem reconstruir todo o grafo textual.

### Probe Multimodal Isolado

Foi criado [smoke_multimodal_probe.py](</c:/Users/Ian Gabriel/Desktop/TCC/raganything-demo/smoke_multimodal_probe.py>), permitindo testar isoladamente:

```powershell
uv run python smoke_multimodal_probe.py TABLE --index 1
uv run python smoke_multimodal_probe.py IMAGE --index 1
```

A motivação foi localizar erros por modalidade e por elemento específico, evitando executar todas as imagens e tabelas em cada diagnóstico.

## 8. Preservação De Resultados Em Diretório Legacy

Como diferentes configurações poderiam sobrescrever resultados anteriores, foram criadas cópias de segurança dos checkpoints relevantes em diretórios `legacy`.

Entre os estados preservados está o melhor checkpoint textual Docling:

```text
legacy/docling-best-text-checkpoint-20260526-170815
```

Ele preserva:

```text
128 nós
69 relações
```

A motivação foi manter resultados reproduzíveis para comparação posterior, sobretudo durante experimentos com MinerU, normalização de tabelas e mudanças de prompt.

## 9. Desativação Do Gleaning

Durante a extração textual, o LightRAG realizava chamadas adicionais de *gleaning*, isto é, tentativas extras de extrair entidades e relações após a primeira resposta do modelo.

Esse mecanismo pode melhorar completude em modelos robustos, mas no ambiente local provocava:

- aumento substancial do tempo de execução;
- repetição de warnings de formato;
- maior custo de inferência;
- pouca evidência de correção efetiva dos registros malformados.

Por isso, o parâmetro foi alterado para:

```python
entity_extract_max_gleaning = 0
```

A motivação foi tornar o experimento viável e previsível em tempo de execução. Essa mudança pode diminuir a cobertura final do grafo, sendo importante documentá-la como um compromisso entre desempenho computacional e profundidade de extração.

## 10. Warnings De Formato De Saída Do LLM

Durante a construção do grafo textual, foram observados avisos como:

```text
Complete delimiter can not be found in extraction result
LLM output format error; found 2/5 fields on RELATION
```

Esses avisos indicam que o modelo local nem sempre obedeceu integralmente ao formato estruturado esperado pelo LightRAG.

A interpretação adotada foi:

- O aviso de delimiter ausente refere-se à resposta completa gerada para um chunk.
- Mesmo com esse aviso, entidades e relações válidas anteriores podem ser extraídas e utilizadas.
- O aviso de quantidade incorreta de campos refere-se a registros específicos malformados, provavelmente descartados.
- Chunks com `0 Rel` podem indicar perda de informação relacional, mas não necessariamente invalidam todo o grafo.

Esses warnings foram aceitos inicialmente porque o pipeline ainda conseguia persistir grafos utilizáveis. Contudo, eles devem ser considerados uma limitação da substituição de GPT-4o-mini por um modelo local menor e quantizado.

## 11. Resultado Da Indexação Multimodal Completa

Após a correção das tabelas, o processamento multimodal completo foi concluído com sucesso:

```text
Individual multimodal content processing complete
Writing graph with 228 nodes, 341 edges
```

Isso significa que:

- o grafo textual foi carregado;
- elementos visuais foram transformados em descrições;
- tabelas corrigidas foram integradas;
- entidades e relações multimodais foram incorporadas ao grafo;
- o índice foi persistido em disco.

Esse foi um resultado importante, pois a fase multimodal anteriormente falhava durante o processamento das tabelas.

## 12. Falha Na Geração Final De Respostas

Após a indexação multimodal, o runner executou sete perguntas do dataset. Para cada pergunta, a recuperação híbrida encontrou contexto relevante, incluindo chunks e imagens.

Por exemplo:

```text
Final context: 66 entities, 142 relations, 20 chunks
Processed 9 images for VLM
```

Entretanto, todas as chamadas finais ao VLM falharam:

```text
ERROR: VLM call failed:
model runner has unexpectedly stopped
```

A quantidade de imagens enviadas ao Qwen por consulta variou entre `7` e `12`, além do contexto textual e relacional recuperado.

| Pergunta | Imagens enviadas ao VLM | Resultado final |
|---|---:|---|
| 1 | 9 | Falha |
| 2 | 7 | Falha |
| 3 | 11 | Falha |
| 4 | 11 | Falha |
| 5 | 11 | Falha |
| 6 | 11 | Falha |
| 7 | 12 | Falha |

A interpretação é que:

- a indexação multimodal funcionou;
- a recuperação híbrida funcionou;
- a falha ocorreu apenas na geração da resposta final;
- o volume de imagens e contexto excedeu a capacidade prática do Qwen local no ambiente disponível.

Essa limitação é especialmente relevante porque o objetivo atual do TCC não exige necessariamente gerar respostas completas, mas avaliar se o retrieval localiza as páginas corretas.

## 13. Mudança De Objetivo Avaliativo: Retrieval Por Página

O artigo avalia a capacidade final do sistema de responder perguntas multimodais. No experimento local, decidiu-se avaliar especificamente a etapa de recuperação, verificando se o sistema encontra as páginas que contêm a evidência correta.

O dataset MMLongBench-Doc já disponibiliza as páginas de referência por pergunta na coluna:

```text
evidence_pages
```

Para o documento analisado, exemplos incluem:

| Tipo de evidência | Página(s) correta(s) |
|---|---|
| Chart | `[5]` |
| Table | `[19, 20]` |
| Pure text | `[2, 22, 23]` |
| Chart + layout text | `[4, 7, 9, 10, 13, 14]` |

A etapa de retrieval do LightRAG já existe internamente por meio de:

```python
rag.lightrag.aquery_data(...)
```

Essa função retorna entidades, relações e chunks recuperados sem executar a geração final de resposta.

Entretanto, ainda falta implementar um runner específico que:

1. carregue o checkpoint multimodal completo;
2. execute somente o retrieval para cada pergunta;
3. identifique as páginas associadas aos chunks recuperados;
4. compare essas páginas com `evidence_pages`;
5. compute métricas como `Hit@1`, `Hit@k`, `Recall@k` e `MRR`.

## 14. Reranker Ainda Não Implementado

O artigo utiliza:

```text
bge-reranker-v2-m3
```

Na implementação atual, o reranker ainda não foi conectado. Durante consultas, aparece o aviso:

```text
Rerank is enabled but no rerank model is configured.
```

Isso significa que os chunks são recuperados, mas não passam pela etapa de reranqueamento adotada no artigo.

A ausência do reranker não causou os crashes do VLM, mas pode afetar a qualidade da ordenação dos resultados recuperados. Portanto, para uma comparação mais fiel com o artigo, a inclusão futura do `bge-reranker-v2-m3` é desejável.

## 15. Síntese Das Principais Diferenças

| Aspecto | Artigo | Experimento local | Motivação |
|---|---|---|---|
| LLM/VLM | GPT-4o-mini | Qwen2.5-VL 7B via Ollama | Execução local |
| Embeddings | text-embedding-3-large, 3072-d | BGE-M3, 1024-d | Execução local |
| Parser | MinerU | Docling | Melhor resultado observado no documento testado |
| Reranker | bge-reranker-v2-m3 | Ainda ausente | Implementação futura |
| Limites de tokens | 20k relações/entidades, 12k chunks | 5k total, chunks de 1k | Limitação de contexto/VRAM |
| Gleaning | Fluxo padrão | Desativado | Reduzir tempo de execução |
| GPU | Infraestrutura não detalhada | Qwen GPU, BGE CPU | Administrar VRAM |
| Tabelas | Fluxo nativo | Normalização Docling -> Markdown | Corrigir crash por prompt excessivo |
| Pipeline | Execução integral | Checkpoints textual/multimodal | Evitar reprocessamento demorado |
| Avaliação | Resposta final | Retrieval por página planejado | Evitar limitação do VLM e avaliar recuperação |

## Texto Curto Para Incorporar Ao TCC

Neste trabalho, a arquitetura RAG-Anything foi adaptada para execução integralmente local. Diferentemente da configuração experimental original, que utiliza MinerU, GPT-4o-mini, text-embedding-3-large e bge-reranker-v2-m3, a implementação local empregou Docling para parsing, Qwen2.5-VL 7B via Ollama para tarefas textuais e visuais e BGE-M3 para embeddings. Essas alterações foram motivadas pela necessidade de eliminar dependências de APIs externas e executar o experimento em hardware disponível localmente.

Durante a implementação, verificou-se que a extração textual com Docling produziu um grafo mais completo para o documento avaliado do que o teste realizado com MinerU. Entretanto, as tabelas extraídas pelo Docling continham metadados estruturais extensos que eram enviados integralmente ao modelo multimodal, produzindo prompts excessivos e falhas do runner do Ollama. Para corrigir esse problema, foi introduzida uma etapa de pós-processamento que converte tabelas para uma representação Markdown compacta, preservando apenas seu conteúdo textual. Após essa alteração, tabelas que anteriormente causavam falha puderam ser inseridas com sucesso no grafo multimodal.

Devido ao elevado custo computacional da indexação, o pipeline foi separado em checkpoints textuais e multimodais, permitindo repetir experimentos visuais sem reconstruir o grafo textual. A indexação multimodal final foi concluída com sucesso, resultando em um grafo com 228 nós e 341 arestas. Contudo, a etapa posterior de geração de respostas falhou ao enviar múltiplas imagens recuperadas ao Qwen local, evidenciando uma limitação prática de memória/contexto do ambiente experimental. Como consequência, a avaliação subsequente será direcionada à qualidade do retrieval, utilizando as páginas de evidência já fornecidas pelo MMLongBench-Doc e métricas baseadas na recuperação das páginas corretas.