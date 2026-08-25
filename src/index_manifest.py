"""Manifesto e proteção de compatibilidade do índice vetorial local."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    ARQUIVO_MANIFESTO_INDICE,
    NOME_COLECAO,
    SOBREPOSICAO_TRECHO,
    TAMANHO_TRECHO,
)


class ErroManifesto(RuntimeError):
    """Incompatibilidade que poderia misturar embeddings no mesmo índice."""


def normalizar_modelo(nome: str) -> str:
    nome = nome.strip().casefold()
    return nome if ":" in nome else f"{nome}:latest"


@dataclass(frozen=True)
class ManifestoIndice:
    versao: int
    modelo_embeddings: str
    dimensao: int
    tamanho_trecho: int
    sobreposicao_trecho: int
    data_indexacao: str
    colecao: str
    trechos: int


def criar_manifesto(modelo: str, dimensao: int, trechos: int) -> ManifestoIndice:
    return ManifestoIndice(
        versao=1,
        modelo_embeddings=normalizar_modelo(modelo),
        dimensao=dimensao,
        tamanho_trecho=TAMANHO_TRECHO,
        sobreposicao_trecho=SOBREPOSICAO_TRECHO,
        data_indexacao=datetime.now(timezone.utc).isoformat(),
        colecao=NOME_COLECAO,
        trechos=trechos,
    )


def salvar_manifesto(
    manifesto: ManifestoIndice, caminho: Path = ARQUIVO_MANIFESTO_INDICE
) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(asdict(manifesto), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporario.replace(caminho)


def carregar_manifesto(
    caminho: Path = ARQUIVO_MANIFESTO_INDICE,
) -> ManifestoIndice | None:
    if not caminho.exists():
        return None
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return ManifestoIndice(**dados)
    except (OSError, json.JSONDecodeError, TypeError) as erro:
        raise ErroManifesto(f"Manifesto do índice inválido em '{caminho}': {erro}") from erro


def validar_compatibilidade(
    modelo_configurado: str,
    manifesto: ManifestoIndice | None,
    *,
    indice_tem_dados: bool = True,
) -> None:
    if not indice_tem_dados:
        return
    if manifesto is None:
        raise ErroManifesto(
            "O índice existente não possui manifesto. Reindexe toda a biblioteca com "
            "confirmação para registrar o modelo e impedir a mistura de embeddings."
        )
    configurado = normalizar_modelo(modelo_configurado)
    if configurado != normalizar_modelo(manifesto.modelo_embeddings):
        raise ErroManifesto(
            "Modelo de embeddings incompatível: o índice usa "
            f"'{manifesto.modelo_embeddings}', mas a configuração solicita "
            f"'{configurado}'. Se quiser trocar o modelo, faça uma reindexação completa "
            "e confirmada; embeddings antigos e novos não serão misturados."
        )
    if (
        manifesto.tamanho_trecho != TAMANHO_TRECHO
        or manifesto.sobreposicao_trecho != SOBREPOSICAO_TRECHO
    ):
        raise ErroManifesto(
            "A configuração de trechos difere daquela registrada no índice. "
            "Faça uma reindexação completa e confirmada."
        )
