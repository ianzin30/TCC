import asyncio
import os
from functools import partial

from dotenv import load_dotenv

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# Carrega variáveis do arquivo .env
load_dotenv()

# Caminho do PDF que será processado
PDF_PATH = "sample.pdf"


async def main():
    # Lê a chave da API
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY não foi encontrada. Verifique o arquivo .env.")

    # Configuração básica do RAG-Anything
    config = RAGAnythingConfig(
        working_dir="./rag_storage",

        # Parser usado para extrair conteúdo do documento
        # Opções comuns: "mineru", "docling", "paddleocr"
        parser="docling",

        # auto deixa o parser escolher a melhor estratégia
        parse_method="auto",

        # Ativa processamento multimodal
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Função de LLM para respostas textuais
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kwargs,
        )

    # Função de visão para imagens, gráficos e conteúdo visual
    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        messages=None,
        **kwargs,
    ):
        # Alguns modos multimodais já passam a mensagem pronta
        if messages:
            return openai_complete_if_cache(
                "gpt-4o-mini",
                "",
                messages=messages,
                api_key=api_key,
                **kwargs,
            )

        # Caso exista imagem codificada em base64
        if image_data:
            return openai_complete_if_cache(
                "gpt-4o-mini",
                "",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                api_key=api_key,
                **kwargs,
            )

        # Caso seja apenas texto
        return llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # Função de embedding
    # text-embedding-3-small usa dimensão 1536
    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model="text-embedding-3-small",
            api_key=api_key,
        ),
    )

    # Inicializa o RAG-Anything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    print("Processando documento...")

    # Processa o PDF e cria a base de conhecimento
    await rag.process_document_complete(
        file_path=PDF_PATH,
        output_dir="./output",
        parse_method="auto",
    )

    print("Documento processado.")
    print("Fazendo pergunta...")

    # Pergunta simples sobre o documento
    result = await rag.aquery(
        "What is this document mainly about?",
        mode="hybrid",
    )

    print("\nResposta:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())