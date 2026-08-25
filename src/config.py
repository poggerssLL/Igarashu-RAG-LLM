"""Configurações centrais do projeto RAG local."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ_PROJETO / ".env")

PASTA_DOCUMENTOS = RAIZ_PROJETO / "Documentos"
PASTA_DADOS = RAIZ_PROJETO / "Dados"
PASTA_CHROMA = PASTA_DADOS / "chroma"
ARQUIVO_MANIFESTO_INDICE = PASTA_DADOS / "manifesto_indice.json"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODELO_CONVERSA = os.getenv("MODELO_CONVERSA", "qwen2.5:3b")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "nomic-embed-text")
MODELOS_EMBEDDINGS_SUPORTADOS = ("nomic-embed-text", "embeddinggemma")
NOME_COLECAO = os.getenv("NOME_COLECAO", "materiais_estudo")

TAMANHO_TRECHO = 1_200
SOBREPOSICAO_TRECHO = 200
TAMANHO_LOTE_EMBEDDINGS = 16
MINIMO_CANDIDATOS = 20


def preparar_pastas() -> None:
    """Garante que as pastas locais necessárias existam."""
    PASTA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
