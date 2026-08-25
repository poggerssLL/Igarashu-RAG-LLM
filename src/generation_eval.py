"""Avaliação local da geração fundamentada e da linha de base compatível."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ollama import Client

from .chat import (
    TrechoRecuperado,
    abrir_colecao,
    consultar,
    resposta_no_idioma,
)
from .config import OLLAMA_HOST, RAIZ_PROJETO
from .grounded import (
    AfirmacaoVerificada,
    ResultadoFundamentado,
    consultar_fundamentado,
    verificar_afirmacoes,
)


ARQUIVO_CASOS_GERACAO = RAIZ_PROJETO / "avaliacao" / "casos_geracao.json"
ARQUIVO_LINHA_BASE = RAIZ_PROJETO / "avaliacao" / "linha_base_geracao.json"


@dataclass(frozen=True)
class ResultadoGeracao:
    pergunta: str
    modo: str
    recuperou_pagina: bool
    conceitos_presentes: bool
    citacoes_validas: bool
    idioma_correto: bool
    recusa_correta: bool
    nao_sustentadas_detectadas: int
    nao_sustentadas_publicadas: int
    paginas_retornadas: tuple[int, ...]
    documento: str
    resposta: str
    observacao: str


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(item for item in texto if not unicodedata.combining(item))
    return " ".join(texto.split())


def carregar_casos_geracao(
    caminho: Path = ARQUIVO_CASOS_GERACAO,
) -> list[dict]:
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, list) or not dados:
        raise ValueError("A avaliação da geração exige uma lista não vazia de casos.")
    return dados


def extrair_citacoes(resposta: str) -> list[tuple[str, int]]:
    return [
        (arquivo.strip(), int(pagina))
        for arquivo, pagina in re.findall(
            r"\[([^\],]+),\s*página do PDF\s+(\d+)\]", resposta
        )
    ]


def validar_citacoes(
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    exigir: bool,
) -> bool:
    citacoes = extrair_citacoes(resposta)
    permitidas = {(item.arquivo, item.pagina) for item in trechos}
    return (bool(citacoes) or not exigir) and all(
        citacao in permitidas for citacao in citacoes
    )


def resposta_recusou(resposta: str) -> bool:
    texto = normalizar(resposta)
    return any(
        trecho in texto
        for trecho in (
            "nao encontrei evidencia suficiente",
            "nao encontrei a resposta",
            "informacao faltante",
            "could not find sufficient evidence",
        )
    )


def auditar_resposta_compatibilidade(
    cliente: Client,
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    idioma: str,
) -> list[AfirmacaoVerificada]:
    corpo = resposta.split("\n\nFontes", 1)[0]
    sentencas = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", corpo)
        if len(item.strip()) > 8
    ][:6]
    rascunho = [
        {
            "texto": sentenca,
            "secao": "explicacao",
            "paginas": list({item.pagina for item in trechos}),
            "natureza": "texto_explicito",
        }
        for sentenca in sentencas
    ]
    return verificar_afirmacoes(cliente, rascunho, trechos, idioma)


def avaliar_saida(
    caso: dict,
    modo: str,
    trechos: Sequence[TrechoRecuperado],
    resposta: str,
    documento: str,
    afirmacoes: Sequence[AfirmacaoVerificada],
    insuficiente: bool,
) -> ResultadoGeracao:
    paginas = tuple(dict.fromkeys(item.pagina for item in trechos))
    esperadas = {int(item) for item in caso.get("paginas_esperadas", [])}
    termos = [normalizar(str(item)) for item in caso.get("conceitos_esperados", [])]
    texto = normalizar(resposta)
    espera_recusa = bool(caso.get("espera_recusa"))
    recusou = insuficiente or resposta_recusou(resposta)
    reprovadas = sum(
        item.classificacao == "não sustentada" for item in afirmacoes
    )
    publicadas = reprovadas if modo == "compatibilidade" else 0
    return ResultadoGeracao(
        pergunta=caso["pergunta"],
        modo=modo,
        recuperou_pagina=(not esperadas or bool(esperadas & set(paginas))),
        conceitos_presentes=(not termos or all(item in texto for item in termos)),
        citacoes_validas=validar_citacoes(
            resposta, trechos, exigir=not espera_recusa
        ),
        idioma_correto=resposta_no_idioma(resposta, caso.get("idioma", "Português")),
        recusa_correta=(recusou == espera_recusa),
        nao_sustentadas_detectadas=reprovadas,
        nao_sustentadas_publicadas=publicadas,
        paginas_retornadas=paginas,
        documento=documento,
        resposta=resposta,
        observacao=str(caso.get("observacao") or ""),
    )


def executar_avaliacao_geracao(
    modo: str = "fundamentado",
    casos: Sequence[dict] | None = None,
) -> list[ResultadoGeracao]:
    casos = list(casos or carregar_casos_geracao())
    cliente = Client(host=OLLAMA_HOST, timeout=180)
    colecao = abrir_colecao()
    resultados: list[ResultadoGeracao] = []
    for indice, caso in enumerate(casos, start=1):
        print(
            f"[{modo}] caso {indice}/{len(casos)}: {caso['pergunta']}",
            flush=True,
        )
        if modo == "fundamentado":
            resultado = consultar_fundamentado(
                caso["pergunta"],
                disciplina=caso.get("disciplina"),
                arquivo=caso.get("arquivo"),
                idioma=caso.get("idioma", "Português"),
                nivel_detalhe=caso.get("nivel_detalhe", "Explicado"),
                incluir_vizinhas=bool(caso.get("incluir_vizinhas", True)),
                cliente=cliente,
                colecao=colecao,
            )
            resultados.append(
                avaliar_saida(
                    caso,
                    modo,
                    resultado.trechos,
                    resultado.resposta,
                    resultado.documento,
                    resultado.afirmacoes,
                    resultado.insuficiente,
                )
            )
        elif modo == "compatibilidade":
            trechos, resposta = consultar(
                caso["pergunta"],
                4,
                cliente_ollama=cliente,
                colecao=colecao,
                disciplina=caso.get("disciplina"),
                incluir_vizinhas=bool(caso.get("incluir_vizinhas", False)),
                idioma_resposta=caso.get("idioma", "Português"),
            )
            auditoria = auditar_resposta_compatibilidade(
                cliente, resposta, trechos, caso.get("idioma", "Português")
            )
            resultados.append(
                avaliar_saida(
                    caso,
                    modo,
                    trechos,
                    resposta,
                    "Vários PDFs possíveis",
                    auditoria,
                    resposta_recusou(resposta),
                )
            )
        else:
            raise ValueError(f"Modo inválido: {modo}")
    return resultados


def resumo_metricas(resultados: Sequence[ResultadoGeracao]) -> dict:
    total = len(resultados)
    if not total:
        return {"casos": 0}
    return {
        "casos": total,
        "recuperacao_pagina": sum(item.recuperou_pagina for item in resultados) / total,
        "conceitos": sum(item.conceitos_presentes for item in resultados) / total,
        "citacoes": sum(item.citacoes_validas for item in resultados) / total,
        "idioma": sum(item.idioma_correto for item in resultados) / total,
        "recusa": sum(item.recusa_correta for item in resultados) / total,
        "casos_sem_afirmacao_publicada_nao_sustentada": sum(
            item.nao_sustentadas_publicadas == 0 for item in resultados
        ) / total,
        "afirmacoes_nao_sustentadas_detectadas": sum(
            item.nao_sustentadas_detectadas for item in resultados
        ),
        "afirmacoes_nao_sustentadas_publicadas": sum(
            item.nao_sustentadas_publicadas for item in resultados
        ),
    }


def salvar_linha_base(resultados: Sequence[ResultadoGeracao]) -> None:
    ARQUIVO_LINHA_BASE.write_text(
        json.dumps(
            {
                "modo": "compatibilidade",
                "metricas": resumo_metricas(resultados),
                "casos": [item.__dict__ for item in resultados],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
