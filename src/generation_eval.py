"""Avaliação reproduzível da resposta final dos modos de geração locais."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ollama import Client

from .chat import TrechoRecuperado, abrir_colecao, consultar, remover_secao_fontes
from .config import (
    MINIMO_CANDIDATOS,
    MODELO_CONVERSA,
    MODELO_EMBEDDINGS,
    OLLAMA_HOST,
    RAIZ_PROJETO,
)
from .grounded import (
    AfirmacaoVerificada,
    DiagnosticoEstrutural,
    EvidenciaOrganizada,
    TrechoRotulado,
    consultar_fundamentado,
    verificar_afirmacoes,
)
from .index_manifest import carregar_manifesto


ARQUIVO_CASOS_GERACAO = RAIZ_PROJETO / "avaliacao" / "casos_geracao.json"
ARQUIVO_LINHA_BASE = RAIZ_PROJETO / "avaliacao" / "linha_base_geracao.json"
PASTA_RESULTADOS_GERACAO = RAIZ_PROJETO / "avaliacao" / "resultados"
VERSAO_ESQUEMA_AVALIACAO = "2.1"
AVISO_AUDITORIA_QWEN = (
    "A auditoria semântica é uma métrica auxiliar produzida por um LLM local. "
    "Ela não é uma validação independente e não substitui gabarito ou revisão humana."
)

_PADRAO_CITACAO = re.compile(
    r"\[([^\],\n]+),\s*página do PDF\s+(\d+)\]", re.IGNORECASE
)
_PADRAO_CITACAO_APARENTE = re.compile(
    r"\[[^\]\n]*(?:página|pagina|\bp\.|\.pdf\b)[^\]\n]*\]", re.IGNORECASE
)


@dataclass(frozen=True)
class ResultadoGeracao:
    pergunta: str
    modo: str
    tipo_caso: str
    expectativa: dict
    arquivo_correto: bool | None
    pagina_correta: bool | None
    fonte_correta: bool | None
    conceitos_presentes: bool | None
    citacao_formal_valida: bool | None
    citacao_recuperada: bool | None
    citacao_sustenta_afirmacao: bool | None
    idioma_correto: bool | None
    resposta_presente: bool | None
    recusa_correta: bool
    nao_sustentadas_detectadas: int
    parcialmente_sustentadas_detectadas: int
    nao_sustentadas_publicadas: int
    paginas_retornadas: tuple[int, ...]
    fontes_retornadas: tuple[tuple[str, int], ...]
    citacoes: tuple[tuple[str, int], ...]
    documento: str
    resposta: str
    observacao: str
    afirmacoes_publicadas: tuple[str, ...]
    afirmacoes_auditadas: tuple[AfirmacaoVerificada, ...]
    trechos: tuple[TrechoRecuperado, ...]
    duracao_segundos: float
    modelo_gerador: str
    modelo_auditor: str
    gerador_e_auditor_iguais: bool
    avaliacao_independente: bool
    aviso_auditoria: str = AVISO_AUDITORIA_QWEN
    evidencias_geracao: tuple[EvidenciaOrganizada, ...] = ()
    afirmacoes_geracao: tuple[AfirmacaoVerificada, ...] = ()
    trechos_rotulados: tuple[TrechoRotulado, ...] = ()
    diagnostico_estrutural: DiagnosticoEstrutural = field(
        default_factory=DiagnosticoEstrutural
    )
    origem_vinculos_evidencia: str = "nao_aplicavel"
    afirmacoes_com_evidencia_valida: bool | None = None
    evidencias_com_trechos_validos: bool | None = None
    citacoes_derivadas_evidencias: bool | None = None
    cobertura_evidencias_afirmacoes: float | None = None
    tentativas_evidencia_inexistente: int | None = None
    tentativas_trecho_inexistente: int | None = None
    tentativas_mistura_arquivos: int | None = None
    afirmacoes_publicadas_sem_evidencia: int | None = None
    arquivo_recuperado: bool | None = None
    pagina_recuperada: bool | None = None
    fonte_recuperada: bool | None = None
    citacao_pagina_esperada: bool | None = None
    citacao_fonte_esperada: bool | None = None
    citacoes_inline: tuple[tuple[str, int], ...] = ()
    citacoes_bibliografia: tuple[tuple[str, int], ...] = ()
    quantidade_citacoes_unicas: int | None = None
    citacoes_duplicadas_removidas: int | None = None
    trechos_suporte_por_afirmacao: tuple[int, ...] | None = None
    paginas_citadas_por_afirmacao: tuple[int, ...] | None = None

    @property
    def recuperou_pagina(self) -> bool | None:
        """Alias legado para consumidores antigos da avaliação."""
        return self.pagina_recuperada

    @property
    def citacoes_validas(self) -> bool | None:
        """Compatibilidade: exige formato e pertencimento ao contexto recuperado."""
        if self.citacao_formal_valida is None or self.citacao_recuperada is None:
            return None
        return self.citacao_formal_valida and self.citacao_recuperada

    @property
    def metricas_deterministicas(self) -> dict[str, bool | None]:
        return {
            "arquivo_recuperado": self.arquivo_recuperado,
            "pagina_recuperada": self.pagina_recuperada,
            "fonte_recuperada": self.fonte_recuperada,
            "citacao_pagina_esperada": self.citacao_pagina_esperada,
            "citacao_fonte_esperada": self.citacao_fonte_esperada,
            "arquivo_correto": self.arquivo_correto,
            "pagina_correta": self.pagina_correta,
            "fonte_correta": self.fonte_correta,
            "conceitos_presentes": self.conceitos_presentes,
            "citacao_formal_valida": self.citacao_formal_valida,
            "citacao_recuperada": self.citacao_recuperada,
            "citacoes_validas": self.citacoes_validas,
            "recusa_correta": self.recusa_correta,
            "idioma_correto": self.idioma_correto,
            "resposta_presente": self.resposta_presente,
            "afirmacoes_com_evidencia_valida": self.afirmacoes_com_evidencia_valida,
            "evidencias_com_trechos_validos": self.evidencias_com_trechos_validos,
            "citacoes_derivadas_evidencias": self.citacoes_derivadas_evidencias,
        }

    @property
    def metricas_rastreabilidade_deterministicas(self) -> dict:
        return {
            "origem_vinculos": self.origem_vinculos_evidencia,
            "cobertura_evidencias_afirmacoes": self.cobertura_evidencias_afirmacoes,
            "tentativas_evidencia_inexistente": self.tentativas_evidencia_inexistente,
            "tentativas_trecho_inexistente": self.tentativas_trecho_inexistente,
            "tentativas_mistura_arquivos": self.tentativas_mistura_arquivos,
            "afirmacoes_publicadas_sem_evidencia": self.afirmacoes_publicadas_sem_evidencia,
            "quantidade_citacoes_unicas": self.quantidade_citacoes_unicas,
            "citacoes_duplicadas_removidas": self.citacoes_duplicadas_removidas,
            "trechos_suporte_por_afirmacao": (
                list(self.trechos_suporte_por_afirmacao)
                if self.trechos_suporte_por_afirmacao is not None
                else None
            ),
            "paginas_citadas_por_afirmacao": (
                list(self.paginas_citadas_por_afirmacao)
                if self.paginas_citadas_por_afirmacao is not None
                else None
            ),
            "ids_rejeitados": self.diagnostico_estrutural.como_dict(),
        }

    @property
    def metricas_auxiliares_qwen(self) -> dict:
        return {
            "citacao_sustenta_afirmacao": self.citacao_sustenta_afirmacao,
            "afirmacoes_sustentadas": sum(
                item.classificacao == "sustentada" for item in self.afirmacoes_auditadas
            ),
            "afirmacoes_parcialmente_sustentadas": self.parcialmente_sustentadas_detectadas,
            "afirmacoes_nao_sustentadas": self.nao_sustentadas_detectadas,
            "afirmacoes_inseguras_publicadas": self.nao_sustentadas_publicadas,
            "modelo_gerador": self.modelo_gerador,
            "modelo_auditor": self.modelo_auditor,
            "gerador_e_auditor_iguais": self.gerador_e_auditor_iguais,
            "avaliacao_independente": self.avaliacao_independente,
            "aviso": self.aviso_auditoria,
        }


class ResultadosGeracao(list[ResultadoGeracao]):
    """Lista compatível com a API anterior, com o caminho do relatório salvo."""

    def __init__(
        self,
        itens: Sequence[ResultadoGeracao] = (),
        *,
        relatorio: Path | None = None,
    ) -> None:
        super().__init__(itens)
        self.relatorio = relatorio


@dataclass(frozen=True)
class AnaliseCitacoes:
    formal_valida: bool | None
    pertencem_recuperacao: bool | None
    ocorrencias: tuple[tuple[str, int], ...]
    unicas: tuple[tuple[str, int], ...]
    inline: tuple[tuple[str, int], ...]
    bibliografia: tuple[tuple[str, int], ...]
    duplicadas_removidas: int


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(item for item in texto if not unicodedata.combining(item))
    return " ".join(texto.split())


def normalizar_caminho(caminho: str) -> str:
    return caminho.strip().replace("\\", "/").casefold()


def _normalizar_decimal_textual(texto: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", ".", normalizar(texto))


def _alternativa_presente(texto: str, alternativa: object) -> bool:
    texto_normalizado = _normalizar_decimal_textual(texto)
    if isinstance(alternativa, str):
        return _normalizar_decimal_textual(alternativa) in texto_normalizado
    if not isinstance(alternativa, dict):
        return False
    if alternativa.get("texto") is not None:
        return _normalizar_decimal_textual(str(alternativa["texto"])) in texto_normalizado
    try:
        esperado = float(str(alternativa["valor"]).replace(",", "."))
        tolerancia = float(
            str(alternativa.get("tolerancia", 0.0)).replace(",", ".")
        )
    except (KeyError, TypeError, ValueError):
        return False
    unidade = str(alternativa.get("unidade") or "").strip()
    if not unidade or tolerancia < 0:
        return False
    unidade_padrao = re.escape(_normalizar_decimal_textual(unidade)).replace(
        r"\ ", r"\s*"
    )
    padrao = re.compile(
        rf"(?<![\w.,])([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*{unidade_padrao}(?!\w)",
        re.IGNORECASE,
    )
    for encontrado in padrao.finditer(texto_normalizado):
        try:
            valor = float(encontrado.group(1))
        except ValueError:
            continue
        if abs(valor - esperado) <= tolerancia:
            return True
    return False


def avaliar_conceitos(texto: str, conceitos: Sequence[object]) -> bool | None:
    """Avalia grupos declarativos sem pedir equivalência ao modelo auditor."""
    if not conceitos:
        return None
    for conceito in conceitos:
        alternativas = (
            conceito.get("qualquer_de", [])
            if isinstance(conceito, dict)
            else [conceito]
        )
        if not isinstance(alternativas, list) or not alternativas:
            return False
        if not any(_alternativa_presente(texto, item) for item in alternativas):
            return False
    return True


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
        for arquivo, pagina in _PADRAO_CITACAO.findall(resposta)
    ]


def _citacoes_unicas(
    citacoes: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    unicas: list[tuple[str, int]] = []
    vistas: set[tuple[str, int]] = set()
    for arquivo, pagina in citacoes:
        chave = (normalizar_caminho(arquivo), pagina)
        if chave in vistas:
            continue
        vistas.add(chave)
        unicas.append((arquivo, pagina))
    return tuple(unicas)


def _separar_corpo_bibliografia(resposta: str) -> tuple[str, str]:
    partes = re.split(
        r"(?im)^\s*(?:#+\s*)?fontes\s*:?\s*$", resposta, maxsplit=1
    )
    return partes[0], partes[1] if len(partes) == 2 else ""


def analisar_citacoes(
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    aplicavel: bool,
) -> AnaliseCitacoes:
    corpo, bibliografia = _separar_corpo_bibliografia(resposta)
    inline = _citacoes_unicas(extrair_citacoes(corpo))
    finais = _citacoes_unicas(extrair_citacoes(bibliografia))
    ocorrencias = tuple(extrair_citacoes(resposta))
    citacoes = _citacoes_unicas(ocorrencias)
    if not aplicavel:
        return AnaliseCitacoes(
            None,
            None,
            ocorrencias,
            citacoes,
            inline,
            finais,
            len(ocorrencias) - len(citacoes),
        )
    aparentes = _PADRAO_CITACAO_APARENTE.findall(resposta)
    formal = bool(citacoes) and len(ocorrencias) == len(aparentes)
    permitidas = {
        (normalizar_caminho(item.arquivo), item.pagina) for item in trechos
    }
    recuperada = formal and all(
        (normalizar_caminho(arquivo), pagina) in permitidas
        for arquivo, pagina in citacoes
    )
    return AnaliseCitacoes(
        formal,
        recuperada,
        ocorrencias,
        citacoes,
        inline,
        finais,
        len(ocorrencias) - len(citacoes),
    )


def validar_citacoes(
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    exigir: bool,
) -> bool:
    """API legada: valida formato e pertencimento aos trechos."""
    analise = analisar_citacoes(resposta, trechos, aplicavel=exigir)
    return True if not exigir else bool(
        analise.formal_valida and analise.pertencem_recuperacao
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
            "could not find the answer",
        )
    )


def extrair_afirmacoes_publicadas(resposta: str) -> list[str]:
    """Extrai o texto realmente publicado, removendo só a seção Fontes."""
    corpo = remover_secao_fontes(resposta)
    afirmacoes: list[str] = []
    for linha in corpo.splitlines():
        linha = linha.strip()
        if not linha or re.fullmatch(r"#{1,6}\s*.*", linha):
            continue
        linha = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", linha)
        linha = re.sub(
            r"^\*\*(?:Texto explícito da fonte|Dedução simples):\*\*\s*",
            "",
            linha,
            flags=re.IGNORECASE,
        )
        partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÀÃÕÂÊÔI])", linha)
        afirmacoes.extend(item.strip() for item in partes if len(item.strip()) >= 3)
    return afirmacoes


def auditar_resposta_publicada(
    cliente: Client,
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    idioma: str,
    *,
    evidencias: Sequence[EvidenciaOrganizada] | None = None,
    afirmacoes_origem: Sequence[AfirmacaoVerificada] = (),
    trechos_rotulados: Sequence[TrechoRotulado] | None = None,
    diagnostico: DiagnosticoEstrutural | None = None,
) -> list[AfirmacaoVerificada]:
    """Audita a mesma resposta pós-publicação nos dois modos."""
    sentencas = [
        sentenca
        for sentenca in extrair_afirmacoes_publicadas(resposta)
        # A recusa é uma mensagem de controle sobre ausência de evidência, não
        # uma afirmação factual sobre o conteúdo do PDF. Ela é medida pela
        # métrica determinística recusa_correta e permanece registrada no relatório.
        if not resposta_recusou(sentenca)
    ]
    if not sentencas:
        return []
    paginas_disponiveis = list(dict.fromkeys(item.pagina for item in trechos))
    rascunho = []
    for sentenca in sentencas:
        texto_sem_citacoes = _PADRAO_CITACAO.sub("", sentenca).strip()
        paginas_citadas = [pagina for _, pagina in extrair_citacoes(sentenca)]
        paginas = [
            pagina for pagina in paginas_citadas if pagina in paginas_disponiveis
        ] or paginas_disponiveis
        evidencia_ids: list[str] = []
        if evidencias is not None:
            texto_normalizado = normalizar(texto_sem_citacoes)
            correspondentes = [
                item
                for item in afirmacoes_origem
                if item.texto_final
                and (
                    normalizar(item.texto_final) in texto_normalizado
                    or texto_normalizado in normalizar(item.texto_final)
                )
            ]
            if correspondentes:
                evidencia_ids = list(correspondentes[0].evidencia_ids)
        rascunho.append(
            {
                "texto": texto_sem_citacoes,
                "texto_original_publicado": sentenca,
                "secao": "resposta_publicada",
                "paginas": paginas,
                "evidencia_ids": evidencia_ids,
                "natureza": "texto_explicito",
            }
        )
    if evidencias is None:
        return verificar_afirmacoes(cliente, rascunho, trechos, idioma)
    return verificar_afirmacoes(
        cliente,
        rascunho,
        trechos,
        idioma,
        evidencias=evidencias,
        trechos_rotulados=trechos_rotulados,
        diagnostico=diagnostico,
        origem_vinculo="geracao_validada",
    )


def auditar_resposta_compatibilidade(
    cliente: Client,
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    idioma: str,
) -> list[AfirmacaoVerificada]:
    """Alias preservado; agora usa o caminho comum pós-publicação."""
    return auditar_resposta_publicada(cliente, resposta, trechos, idioma)


def classificar_idioma(texto: str) -> str | None:
    corpo = remover_secao_fontes(texto)
    corpo = _PADRAO_CITACAO_APARENTE.sub(" ", corpo)
    termos = re.findall(r"[A-Za-zÀ-ÿ]+", corpo.casefold())
    if not termos:
        return None
    marcadores_pt = {
        "não", "encontrei", "evidência", "suficiente", "resposta", "informação",
        "sinal", "periódico", "periódicos", "período", "fonte", "material",
        "uma", "um", "segundo", "texto", "explícito",
    }
    marcadores_en = {
        "not", "could", "find", "found", "evidence", "sufficient", "answer",
        "information", "signal", "periodic", "period", "source", "material",
        "according", "text", "explicit",
    }
    pontos_pt = sum(item in marcadores_pt for item in termos)
    pontos_en = sum(item in marcadores_en for item in termos)
    if pontos_pt >= 2 and pontos_pt > pontos_en:
        return "Português"
    if pontos_en >= 2 and pontos_en > pontos_pt:
        return "English"
    return None


def avaliar_idioma(texto: str, idioma_esperado: str) -> bool | None:
    detectado = classificar_idioma(texto)
    if detectado is None:
        return None
    esperado = "Português" if idioma_esperado == "Português" else "English"
    return detectado == esperado


def metadados_auditoria(
    modelo_gerador: str = MODELO_CONVERSA,
    modelo_auditor: str = MODELO_CONVERSA,
) -> dict:
    iguais = modelo_gerador.strip().casefold() == modelo_auditor.strip().casefold()
    return {
        "modelo_gerador": modelo_gerador,
        "modelo_auditor": modelo_auditor,
        "gerador_e_auditor_iguais": iguais,
        "avaliacao_independente": not iguais,
        "aviso": AVISO_AUDITORIA_QWEN,
    }


def _citacoes_sustentam_afirmacoes(
    afirmacoes: Sequence[AfirmacaoVerificada],
    *,
    aplicavel: bool,
) -> bool | None:
    if not aplicavel:
        return None
    if not afirmacoes:
        return False
    for afirmacao in afirmacoes:
        citacoes = extrair_citacoes(afirmacao.texto_original)
        fontes_citadas = {
            (normalizar_caminho(arquivo), pagina)
            for arquivo, pagina in citacoes
        }
        fontes_associadas = {
            (normalizar_caminho(arquivo), pagina)
            for arquivo, pagina in afirmacao.fontes
        }
        if (
            afirmacao.classificacao != "sustentada"
            or not citacoes
            or not fontes_citadas.issubset(fontes_associadas)
        ):
            return False
    return True


def avaliar_rastreabilidade_estrutural(
    *,
    modo: str,
    resposta: str,
    espera_resposta: bool,
    evidencias: Sequence[EvidenciaOrganizada] = (),
    afirmacoes: Sequence[AfirmacaoVerificada] = (),
    trechos_rotulados: Sequence[TrechoRotulado] = (),
    diagnostico: DiagnosticoEstrutural | None = None,
) -> dict:
    """Mede os vínculos de IDs sem usar julgamento semântico do modelo."""
    if modo != "fundamentado":
        return {
            "origem_vinculos": "reconstruidos_auxiliares",
            "afirmacoes_com_evidencia_valida": None,
            "evidencias_com_trechos_validos": None,
            "citacoes_derivadas_evidencias": None,
            "cobertura_evidencias_afirmacoes": None,
            "tentativas_evidencia_inexistente": None,
            "tentativas_trecho_inexistente": None,
            "tentativas_mistura_arquivos": None,
            "afirmacoes_publicadas_sem_evidencia": None,
            "trechos_suporte_por_afirmacao": None,
            "paginas_citadas_por_afirmacao": None,
        }

    diagnostico = diagnostico or DiagnosticoEstrutural()
    por_rotulo = {item.rotulo: item for item in trechos_rotulados}
    por_evidencia = {item.id: item for item in evidencias}

    def evidencia_valida(item: EvidenciaOrganizada) -> bool:
        if not item.trecho_ids or len(item.trecho_ids) != len(item.ids_chroma):
            return False
        associados = [por_rotulo.get(rotulo) for rotulo in item.trecho_ids]
        contexto = [
            por_rotulo.get(rotulo) for rotulo in item.trecho_ids_contexto
        ]
        return bool(
            all(associados)
            and all(contexto)
            and len(item.trecho_ids_contexto) == len(item.ids_chroma_contexto)
            and all(
                associado is not None
                and associado.id_chroma == id_chroma
                and associado.arquivo == item.arquivo
                and associado.pagina in item.paginas
                for associado, id_chroma in zip(associados, item.ids_chroma)
            )
            and all(
                associado is not None
                and associado.id_chroma == id_chroma
                and associado.arquivo == item.arquivo
                and associado.pagina in item.paginas_contexto
                for associado, id_chroma in zip(
                    contexto, item.ids_chroma_contexto
                )
            )
            and len(
                {
                    associado.arquivo
                    for associado in [*associados, *contexto]
                    if associado
                }
            ) == 1
        )

    validade_evidencias = {
        item.id: evidencia_valida(item) for item in evidencias
    }

    def afirmacao_valida(item: AfirmacaoVerificada) -> bool:
        selecionadas = [
            por_evidencia.get(evidencia_id)
            for evidencia_id in item.evidencia_ids
        ]
        return bool(
            item.evidencia_ids
            and not item.ids_evidencia_invalidos
            and all(selecionadas)
            and all(validade_evidencias.get(evidencia_id, False) for evidencia_id in item.evidencia_ids)
            and len({evidencia.arquivo for evidencia in selecionadas if evidencia}) == 1
        )

    afirmacoes_factuais = [item for item in afirmacoes if item.texto_original]
    afirmacoes_validas = (
        all(afirmacao_valida(item) for item in afirmacoes_factuais)
        if afirmacoes_factuais
        else (False if espera_resposta else None)
    )
    evidencias_validas = (
        all(validade_evidencias.values())
        if evidencias
        else (False if espera_resposta else None)
    )

    publicadas = [
        sentenca
        for sentenca in extrair_afirmacoes_publicadas(resposta)
        if not resposta_recusou(sentenca)
    ]
    publicadas_validas = 0
    publicadas_sem_evidencia = 0
    citacoes_por_afirmacao_validas = True
    fontes_derivadas: set[tuple[str, int]] = set()
    trechos_suporte_por_afirmacao: list[int] = []
    paginas_citadas_por_afirmacao: list[int] = []
    for sentenca in publicadas:
        limpa = normalizar(_PADRAO_CITACAO.sub("", sentenca))
        correspondentes = [
            item
            for item in afirmacoes
            if item.texto_final
            and (
                normalizar(item.texto_final) in limpa
                or limpa in normalizar(item.texto_final)
            )
        ]
        afirmacao = correspondentes[0] if correspondentes else None
        if afirmacao is None or not afirmacao_valida(afirmacao):
            publicadas_sem_evidencia += 1
            citacoes_por_afirmacao_validas = False
            trechos_suporte_por_afirmacao.append(0)
            paginas_citadas_por_afirmacao.append(
                len(_citacoes_unicas(extrair_citacoes(sentenca)))
            )
            continue
        publicadas_validas += 1
        trechos_suporte_por_afirmacao.append(
            len(
                {
                    rotulo
                    for evidencia_id in afirmacao.evidencia_ids
                    if (evidencia := por_evidencia.get(evidencia_id)) is not None
                    for rotulo in evidencia.trecho_ids_suporte
                }
            )
        )
        fontes = set(
            _fontes_da_afirmacao_por_evidencias(afirmacao, por_evidencia)
        )
        paginas_citadas_por_afirmacao.append(len(fontes))
        fontes_derivadas.update(fontes)
        if set(extrair_citacoes(sentenca)) != fontes:
            citacoes_por_afirmacao_validas = False

    todas_citacoes = set(extrair_citacoes(resposta))
    citacoes_derivadas = (
        bool(publicadas)
        and publicadas_sem_evidencia == 0
        and citacoes_por_afirmacao_validas
        and todas_citacoes == fontes_derivadas
        if espera_resposta
        else None
    )
    cobertura = (
        publicadas_validas / len(publicadas) if publicadas else None
    )
    return {
        "origem_vinculos": "ids_usados_na_geracao",
        "afirmacoes_com_evidencia_valida": afirmacoes_validas,
        "evidencias_com_trechos_validos": evidencias_validas,
        "citacoes_derivadas_evidencias": citacoes_derivadas,
        "cobertura_evidencias_afirmacoes": cobertura,
        "tentativas_evidencia_inexistente": len(
            diagnostico.evidencia_ids_invalidos_rejeitados
        ),
        "tentativas_trecho_inexistente": len(
            diagnostico.trecho_ids_invalidos_rejeitados
        ),
        "tentativas_mistura_arquivos": diagnostico.tentativas_mistura_arquivos,
        "afirmacoes_publicadas_sem_evidencia": publicadas_sem_evidencia,
        "trechos_suporte_por_afirmacao": tuple(
            trechos_suporte_por_afirmacao
        ),
        "paginas_citadas_por_afirmacao": tuple(
            paginas_citadas_por_afirmacao
        ),
    }


def _fontes_da_afirmacao_por_evidencias(
    afirmacao: AfirmacaoVerificada,
    por_evidencia: dict[str, EvidenciaOrganizada],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        dict.fromkeys(
            (evidencia.arquivo, pagina)
            for evidencia_id in afirmacao.evidencia_ids
            if (evidencia := por_evidencia.get(evidencia_id)) is not None
            for pagina in evidencia.paginas
        )
    )


def avaliar_saida(
    caso: dict,
    modo: str,
    trechos: Sequence[TrechoRecuperado],
    resposta: str,
    documento: str,
    afirmacoes: Sequence[AfirmacaoVerificada],
    insuficiente: bool,
    *,
    duracao_segundos: float = 0.0,
    modelo_gerador: str = MODELO_CONVERSA,
    modelo_auditor: str = MODELO_CONVERSA,
    evidencias_geracao: Sequence[EvidenciaOrganizada] | None = None,
    afirmacoes_geracao: Sequence[AfirmacaoVerificada] | None = None,
    trechos_rotulados: Sequence[TrechoRotulado] | None = None,
    diagnostico_estrutural: DiagnosticoEstrutural | None = None,
) -> ResultadoGeracao:
    paginas = tuple(dict.fromkeys(item.pagina for item in trechos))
    fontes = tuple(dict.fromkeys((item.arquivo, item.pagina) for item in trechos))
    esperadas = {int(item) for item in caso.get("paginas_esperadas", [])}
    arquivo_esperado = str(caso.get("arquivo") or "").strip()
    arquivo_normalizado = normalizar_caminho(arquivo_esperado)
    conceitos = list(caso.get("conceitos_esperados", []))
    espera_recusa = bool(caso.get("espera_recusa"))
    espera_resposta = not espera_recusa
    recusou = insuficiente or resposta_recusou(resposta)

    arquivos_retornados = {normalizar_caminho(item.arquivo) for item in trechos}
    arquivo_recuperado = (
        arquivo_normalizado in arquivos_retornados
        if espera_resposta and arquivo_normalizado
        else None
    )
    pagina_recuperada = (
        bool(esperadas.intersection(paginas))
        if espera_resposta and esperadas
        else None
    )
    fonte_recuperada = (
        any(
            normalizar_caminho(item.arquivo) == arquivo_normalizado
            and item.pagina in esperadas
            for item in trechos
        )
        if espera_resposta and arquivo_normalizado and esperadas
        else None
    )
    conceitos_presentes = (
        avaliar_conceitos(resposta, conceitos)
        if espera_resposta and conceitos
        else None
    )
    analise_citacoes = analisar_citacoes(
        resposta, trechos, aplicavel=espera_resposta
    )
    citacao_pagina_esperada = (
        any(pagina in esperadas for _, pagina in analise_citacoes.unicas)
        if espera_resposta and esperadas
        else None
    )
    citacao_fonte_esperada = (
        any(
            normalizar_caminho(arquivo) == arquivo_normalizado
            and pagina in esperadas
            for arquivo, pagina in analise_citacoes.unicas
        )
        if espera_resposta and arquivo_normalizado and esperadas
        else None
    )
    afirmacoes = tuple(afirmacoes)
    reprovadas = sum(
        item.classificacao == "não sustentada" for item in afirmacoes
    )
    parciais = sum(
        item.classificacao == "parcialmente sustentada" for item in afirmacoes
    )
    inseguras_publicadas = sum(
        item.classificacao in {"não sustentada", "parcialmente sustentada"}
        for item in afirmacoes
    )
    auditoria = metadados_auditoria(modelo_gerador, modelo_auditor)
    diagnostico_estrutural = diagnostico_estrutural or DiagnosticoEstrutural()
    if (
        modo == "fundamentado"
        and evidencias_geracao is not None
        and afirmacoes_geracao is not None
        and trechos_rotulados is not None
    ):
        rastreabilidade = avaliar_rastreabilidade_estrutural(
            modo=modo,
            resposta=resposta,
            espera_resposta=espera_resposta,
            evidencias=evidencias_geracao,
            afirmacoes=afirmacoes_geracao,
            trechos_rotulados=trechos_rotulados,
            diagnostico=diagnostico_estrutural,
        )
    else:
        rastreabilidade = avaliar_rastreabilidade_estrutural(
            modo="compatibilidade",
            resposta=resposta,
            espera_resposta=espera_resposta,
        )
    expectativa = {
        "arquivo": arquivo_esperado,
        "paginas_esperadas": sorted(esperadas),
        "conceitos_esperados": conceitos,
        "idioma": caso.get("idioma", "Português"),
        "espera_recusa": espera_recusa,
    }
    return ResultadoGeracao(
        pergunta=str(caso["pergunta"]),
        modo=modo,
        tipo_caso=str(caso.get("tipo") or ""),
        expectativa=expectativa,
        # Aliases legados: no esquema 2.1 estes três campos continuam medindo
        # recuperação. Consumidores novos devem usar os nomes inequívocos.
        arquivo_correto=arquivo_recuperado,
        pagina_correta=pagina_recuperada,
        fonte_correta=fonte_recuperada,
        conceitos_presentes=conceitos_presentes,
        citacao_formal_valida=analise_citacoes.formal_valida,
        citacao_recuperada=analise_citacoes.pertencem_recuperacao,
        citacao_sustenta_afirmacao=_citacoes_sustentam_afirmacoes(
            afirmacoes, aplicavel=espera_resposta
        ),
        idioma_correto=avaliar_idioma(
            resposta, str(caso.get("idioma", "Português"))
        ),
        resposta_presente=(
            bool(remover_secao_fontes(resposta).strip()) and not recusou
            if espera_resposta
            else None
        ),
        recusa_correta=(recusou == espera_recusa),
        nao_sustentadas_detectadas=reprovadas,
        parcialmente_sustentadas_detectadas=parciais,
        nao_sustentadas_publicadas=inseguras_publicadas,
        paginas_retornadas=paginas,
        fontes_retornadas=fontes,
        citacoes=analise_citacoes.unicas,
        documento=documento,
        resposta=resposta,
        observacao=str(caso.get("observacao") or ""),
        afirmacoes_publicadas=tuple(extrair_afirmacoes_publicadas(resposta)),
        afirmacoes_auditadas=afirmacoes,
        trechos=tuple(trechos),
        duracao_segundos=round(duracao_segundos, 6),
        modelo_gerador=modelo_gerador,
        modelo_auditor=modelo_auditor,
        gerador_e_auditor_iguais=bool(
            auditoria["gerador_e_auditor_iguais"]
        ),
        avaliacao_independente=bool(auditoria["avaliacao_independente"]),
        evidencias_geracao=tuple(evidencias_geracao or ()),
        afirmacoes_geracao=tuple(afirmacoes_geracao or ()),
        trechos_rotulados=tuple(trechos_rotulados or ()),
        diagnostico_estrutural=diagnostico_estrutural,
        origem_vinculos_evidencia=str(rastreabilidade["origem_vinculos"]),
        afirmacoes_com_evidencia_valida=rastreabilidade[
            "afirmacoes_com_evidencia_valida"
        ],
        evidencias_com_trechos_validos=rastreabilidade[
            "evidencias_com_trechos_validos"
        ],
        citacoes_derivadas_evidencias=rastreabilidade[
            "citacoes_derivadas_evidencias"
        ],
        cobertura_evidencias_afirmacoes=rastreabilidade[
            "cobertura_evidencias_afirmacoes"
        ],
        tentativas_evidencia_inexistente=rastreabilidade[
            "tentativas_evidencia_inexistente"
        ],
        tentativas_trecho_inexistente=rastreabilidade[
            "tentativas_trecho_inexistente"
        ],
        tentativas_mistura_arquivos=rastreabilidade[
            "tentativas_mistura_arquivos"
        ],
        afirmacoes_publicadas_sem_evidencia=rastreabilidade[
            "afirmacoes_publicadas_sem_evidencia"
        ],
        arquivo_recuperado=arquivo_recuperado,
        pagina_recuperada=pagina_recuperada,
        fonte_recuperada=fonte_recuperada,
        citacao_pagina_esperada=citacao_pagina_esperada,
        citacao_fonte_esperada=citacao_fonte_esperada,
        citacoes_inline=analise_citacoes.inline,
        citacoes_bibliografia=analise_citacoes.bibliografia,
        quantidade_citacoes_unicas=(
            len(analise_citacoes.unicas) if espera_resposta else None
        ),
        citacoes_duplicadas_removidas=(
            analise_citacoes.duplicadas_removidas if espera_resposta else None
        ),
        trechos_suporte_por_afirmacao=(
            tuple(rastreabilidade["trechos_suporte_por_afirmacao"])
            if rastreabilidade["trechos_suporte_por_afirmacao"] is not None
            else None
        ),
        paginas_citadas_por_afirmacao=(
            tuple(rastreabilidade["paginas_citadas_por_afirmacao"])
            if rastreabilidade["paginas_citadas_por_afirmacao"] is not None
            else None
        ),
    )


def resultado_aprovado(resultado: ResultadoGeracao) -> bool:
    deterministicas = [
        valor for valor in resultado.metricas_deterministicas.values()
        if valor is not None
    ]
    return (
        all(deterministicas)
        and resultado.citacao_sustenta_afirmacao is not False
        and resultado.nao_sustentadas_publicadas == 0
        and resultado.tentativas_evidencia_inexistente in {None, 0}
        and resultado.tentativas_trecho_inexistente in {None, 0}
        and resultado.tentativas_mistura_arquivos in {None, 0}
        and resultado.afirmacoes_publicadas_sem_evidencia in {None, 0}
    )


def _agregar_booleanos(
    resultados: Sequence[ResultadoGeracao], atributo: str
) -> dict[str, int | float | None]:
    valores = [getattr(item, atributo) for item in resultados]
    aplicaveis = [item for item in valores if item is not None]
    acertos = sum(item is True for item in aplicaveis)
    return {
        "acertos": acertos,
        "aplicaveis": len(aplicaveis),
        "taxa": acertos / len(aplicaveis) if aplicaveis else None,
    }


def resumo_metricas(resultados: Sequence[ResultadoGeracao]) -> dict:
    total = len(resultados)
    nomes_deterministicos = (
        "arquivo_recuperado",
        "pagina_recuperada",
        "fonte_recuperada",
        "citacao_pagina_esperada",
        "citacao_fonte_esperada",
        "arquivo_correto",
        "pagina_correta",
        "fonte_correta",
        "conceitos_presentes",
        "citacao_formal_valida",
        "citacao_recuperada",
        "citacoes_validas",
        "idioma_correto",
        "resposta_presente",
        "recusa_correta",
        "afirmacoes_com_evidencia_valida",
        "evidencias_com_trechos_validos",
        "citacoes_derivadas_evidencias",
    )
    deterministicas = {
        nome: _agregar_booleanos(resultados, nome)
        for nome in nomes_deterministicos
    }
    citacao_semantica = _agregar_booleanos(
        resultados, "citacao_sustenta_afirmacao"
    )
    sem_inseguras = sum(
        item.nao_sustentadas_publicadas == 0 for item in resultados
    )
    auxiliares = {
        "citacao_sustenta_afirmacao": citacao_semantica,
        "casos_sem_afirmacao_publicada_insegura": {
            "acertos": sem_inseguras,
            "aplicaveis": total,
            "taxa": sem_inseguras / total if total else None,
        },
        "afirmacoes_nao_sustentadas_detectadas": sum(
            item.nao_sustentadas_detectadas for item in resultados
        ),
        "afirmacoes_parcialmente_sustentadas_detectadas": sum(
            item.parcialmente_sustentadas_detectadas for item in resultados
        ),
        "afirmacoes_inseguras_publicadas": sum(
            item.nao_sustentadas_publicadas for item in resultados
        ),
        "avaliacao_independente": all(
            item.avaliacao_independente for item in resultados
        ) if resultados else False,
        "aviso": AVISO_AUDITORIA_QWEN,
    }
    coberturas = [
        item.cobertura_evidencias_afirmacoes
        for item in resultados
        if item.cobertura_evidencias_afirmacoes is not None
    ]
    suportes_por_afirmacao = [
        quantidade
        for item in resultados
        if item.trechos_suporte_por_afirmacao is not None
        for quantidade in item.trechos_suporte_por_afirmacao
    ]
    paginas_por_afirmacao = [
        quantidade
        for item in resultados
        if item.paginas_citadas_por_afirmacao is not None
        for quantidade in item.paginas_citadas_por_afirmacao
    ]
    suporte_aplicavel = any(
        item.trechos_suporte_por_afirmacao is not None for item in resultados
    )
    rastreabilidade = {
        "origens_vinculos": sorted(
            {item.origem_vinculos_evidencia for item in resultados}
        ),
        "cobertura_media_evidencias_afirmacoes": (
            sum(coberturas) / len(coberturas) if coberturas else None
        ),
        "casos_cobertura_aplicavel": len(coberturas),
        "tentativas_evidencia_inexistente": sum(
            item.tentativas_evidencia_inexistente or 0 for item in resultados
        ),
        "tentativas_trecho_inexistente": sum(
            item.tentativas_trecho_inexistente or 0 for item in resultados
        ),
        "tentativas_mistura_arquivos": sum(
            item.tentativas_mistura_arquivos or 0 for item in resultados
        ),
        "afirmacoes_publicadas_sem_evidencia": sum(
            item.afirmacoes_publicadas_sem_evidencia or 0 for item in resultados
        ),
        "quantidade_citacoes_unicas": sum(
            item.quantidade_citacoes_unicas or 0 for item in resultados
        ),
        "citacoes_duplicadas_removidas": sum(
            item.citacoes_duplicadas_removidas or 0 for item in resultados
        ),
        "trechos_suporte_por_afirmacao": (
            suportes_por_afirmacao if suporte_aplicavel else None
        ),
        "paginas_citadas_por_afirmacao": (
            paginas_por_afirmacao if suporte_aplicavel else None
        ),
    }

    def taxa(nome: str) -> float | None:
        valor = deterministicas[nome]["taxa"]
        return float(valor) if valor is not None else None

    citacoes_aplicaveis = [
        item for item in resultados if item.citacoes_validas is not None
    ]
    return {
        "versao_esquema": VERSAO_ESQUEMA_AVALIACAO,
        "casos": total,
        "metricas_deterministicas": deterministicas,
        "metricas_rastreabilidade_deterministicas": rastreabilidade,
        "metricas_auxiliares_qwen": auxiliares,
        # Chaves legadas para a interface e consumidores existentes.
        "recuperacao_pagina": taxa("pagina_recuperada"),
        "conceitos": taxa("conceitos_presentes"),
        "citacoes": (
            sum(item.citacoes_validas is True for item in citacoes_aplicaveis)
            / len(citacoes_aplicaveis)
            if citacoes_aplicaveis else None
        ),
        "idioma": taxa("idioma_correto"),
        "recusa": taxa("recusa_correta"),
        "casos_sem_afirmacao_publicada_nao_sustentada": (
            sem_inseguras / total if total else None
        ),
        "afirmacoes_nao_sustentadas_detectadas": auxiliares[
            "afirmacoes_nao_sustentadas_detectadas"
        ],
        "afirmacoes_nao_sustentadas_publicadas": auxiliares[
            "afirmacoes_inseguras_publicadas"
        ],
    }


def _git(comando: str) -> str:
    try:
        processo = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={RAIZ_PROJETO.as_posix()}",
                *comando.split(),
            ],
            cwd=RAIZ_PROJETO,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return ""
    return processo.stdout.strip() if processo.returncode == 0 else ""


def _serializar_afirmacao(item: AfirmacaoVerificada) -> dict:
    return {
        "texto_original": item.texto_original,
        "texto_final_sugerido_pelo_auditor": item.texto_final,
        "classificacao": item.classificacao,
        "paginas": list(item.paginas),
        "natureza": item.natureza,
        "secao": item.secao,
        "justificativa_auditor": item.justificativa,
        "evidencia_ids": list(item.evidencia_ids),
        "fontes_derivadas": [
            {"arquivo": arquivo, "pagina_pdf": pagina}
            for arquivo, pagina in item.fontes
        ],
        "origem_vinculo": item.origem_vinculo,
        "ids_evidencia_invalidos": list(item.ids_evidencia_invalidos),
    }


def _serializar_trecho(item: TrechoRecuperado) -> dict:
    return {
        "id": item.id,
        "arquivo": item.arquivo,
        "disciplina": item.disciplina,
        "pagina_pdf": item.pagina,
        "indice_trecho": item.indice,
        "texto": item.texto,
        "distancia": item.distancia,
        "relevancia": item.relevancia,
        "pontuacao_palavras": item.pontuacao_palavras,
        "pontuacao_fusao": item.pontuacao_fusao,
        "pagina_vizinha": item.pagina_vizinha,
    }


def _serializar_trecho_rotulado(item: TrechoRotulado) -> dict:
    return {
        "rotulo": item.rotulo,
        "id_chroma": item.id_chroma,
        "arquivo": item.arquivo,
        "pagina_pdf": item.pagina,
        "indice_trecho": item.indice,
        "texto": item.texto,
    }


def _serializar_evidencia(item: EvidenciaOrganizada) -> dict:
    return {
        "id": item.id,
        "tipo": item.tipo,
        "conteudo": item.conteudo,
        "natureza": item.natureza,
        "trecho_ids": list(item.trecho_ids),
        "ids_chroma": list(item.ids_chroma),
        "trecho_ids_suporte": list(item.trecho_ids_suporte),
        "ids_chroma_suporte": list(item.ids_chroma_suporte),
        "trecho_ids_contexto": list(item.trecho_ids_contexto),
        "ids_chroma_contexto": list(item.ids_chroma_contexto),
        "arquivo": item.arquivo,
        "paginas_pdf": list(item.paginas),
        "paginas_contexto_pdf": list(item.paginas_contexto),
    }


def serializar_resultado(item: ResultadoGeracao) -> dict:
    evidencias_por_id = {
        evidencia.id: evidencia for evidencia in item.evidencias_geracao
    }
    return {
        "duracao_segundos": item.duracao_segundos,
        "tipo": item.tipo_caso,
        "pergunta": item.pergunta,
        "expectativa": item.expectativa,
        "resposta_final": item.resposta,
        "afirmacoes_publicadas_extraidas": list(item.afirmacoes_publicadas),
        "documento_escolhido": item.documento,
        "trechos_recuperados": [_serializar_trecho(trecho) for trecho in item.trechos],
        "trechos_rotulados": [
            _serializar_trecho_rotulado(trecho)
            for trecho in item.trechos_rotulados
        ],
        "evidencias_geracao": [
            _serializar_evidencia(evidencia)
            for evidencia in item.evidencias_geracao
        ],
        "afirmacoes_geracao": [
            _serializar_afirmacao(afirmacao)
            for afirmacao in item.afirmacoes_geracao
        ],
        "paginas_retornadas": list(item.paginas_retornadas),
        "fontes_retornadas": [
            {"arquivo": arquivo, "pagina_pdf": pagina}
            for arquivo, pagina in item.fontes_retornadas
        ],
        "citacoes": [
            {"arquivo": arquivo, "pagina_pdf": pagina}
            for arquivo, pagina in item.citacoes
        ],
        "citacoes_publicadas": {
            "inline": [
                {"arquivo": arquivo, "pagina_pdf": pagina}
                for arquivo, pagina in item.citacoes_inline
            ],
            "bibliografia_final": [
                {"arquivo": arquivo, "pagina_pdf": pagina}
                for arquivo, pagina in item.citacoes_bibliografia
            ],
            "unicas": [
                {"arquivo": arquivo, "pagina_pdf": pagina}
                for arquivo, pagina in item.citacoes
            ],
            "quantidade_unicas": item.quantidade_citacoes_unicas,
            "duplicadas_removidas": item.citacoes_duplicadas_removidas,
        },
        "afirmacoes_auditadas": [
            _serializar_afirmacao(afirmacao)
            for afirmacao in item.afirmacoes_auditadas
        ],
        "rastreabilidade": {
            "origem_vinculos": item.origem_vinculos_evidencia,
            "afirmacao_para_evidencias": [
                {
                    "texto": afirmacao.texto_original,
                    "evidencia_ids": list(afirmacao.evidencia_ids),
                    "paginas_derivadas": list(afirmacao.paginas),
                }
                for afirmacao in item.afirmacoes_auditadas
            ],
            "evidencia_para_trechos": [
                {
                    "evidencia_id": evidencia.id,
                    "trecho_ids": list(evidencia.trecho_ids),
                    "ids_chroma": list(evidencia.ids_chroma),
                    "trecho_ids_suporte": list(evidencia.trecho_ids_suporte),
                    "ids_chroma_suporte": list(evidencia.ids_chroma_suporte),
                    "trecho_ids_contexto": list(evidencia.trecho_ids_contexto),
                    "ids_chroma_contexto": list(evidencia.ids_chroma_contexto),
                    "arquivo": evidencia.arquivo,
                    "paginas_pdf": list(evidencia.paginas),
                    "paginas_contexto_pdf": list(evidencia.paginas_contexto),
                }
                for evidencia in evidencias_por_id.values()
            ],
            "validacao_estrutural": item.metricas_rastreabilidade_deterministicas,
        },
        "metricas_deterministicas": item.metricas_deterministicas,
        "metricas_rastreabilidade_deterministicas": (
            item.metricas_rastreabilidade_deterministicas
        ),
        "metricas_auxiliares_qwen": item.metricas_auxiliares_qwen,
        "observacao": item.observacao,
        "limitacoes": [item.aviso_auditoria],
    }


def criar_relatorio_detalhado(
    resultados: Sequence[ResultadoGeracao],
    *,
    modo: str,
    data_utc: datetime | None = None,
    tipo: str = "execucao",
    parametros_recuperacao: dict | None = None,
) -> dict:
    agora = data_utc or datetime.now(timezone.utc)
    if agora.tzinfo is None:
        agora = agora.replace(tzinfo=timezone.utc)
    manifesto = carregar_manifesto()
    modelos = metadados_auditoria()
    return {
        "versao_esquema": VERSAO_ESQUEMA_AVALIACAO,
        "tipo_resultado": tipo,
        "data_hora_utc": agora.astimezone(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse HEAD") or None,
            "branch": _git("branch --show-current") or None,
            "working_tree_dirty": bool(_git("status --porcelain")),
        },
        "ambiente": {
            "python": platform.python_version(),
            # sys.platform é estável e não dispara consultas WMI no Windows.
            "sistema_operacional": sys.platform,
        },
        "modelos": {
            "conversa_gerador": modelos["modelo_gerador"],
            "conversa_auditor": modelos["modelo_auditor"],
            "gerador_e_auditor_iguais": modelos["gerador_e_auditor_iguais"],
            "avaliacao_independente": modelos["avaliacao_independente"],
            "embeddings": (
                manifesto.modelo_embeddings if manifesto else MODELO_EMBEDDINGS
            ),
            "dimensao_embeddings": manifesto.dimensao if manifesto else None,
        },
        "parametros_recuperacao": parametros_recuperacao or {},
        "modo_avaliado": modo,
        "duracao_total_segundos": round(
            sum(item.duracao_segundos for item in resultados), 6
        ),
        "metricas": resumo_metricas(resultados),
        "semantica_metricas": {
            "campos_inequivocos": {
                "pagina_recuperada": "página esperada presente nos trechos recuperados",
                "fonte_recuperada": "arquivo e página esperados presentes nos trechos recuperados",
                "citacao_pagina_esperada": "página esperada citada na resposta publicada",
                "citacao_fonte_esperada": "arquivo e página esperados citados na resposta publicada",
            },
            "aliases_legados_recuperacao": {
                "arquivo_correto": "arquivo_recuperado",
                "pagina_correta": "pagina_recuperada",
                "fonte_correta": "fonte_recuperada",
            },
        },
        "casos": [serializar_resultado(item) for item in resultados],
        "observacoes_e_limitacoes": [
            AVISO_AUDITORIA_QWEN,
            "Métricas determinísticas não aplicáveis são registradas como null e excluídas dos denominadores.",
            "Frases programáticas de recusa são registradas como texto publicado, mas avaliadas por recusa_correta em vez da auditoria factual do Qwen.",
            "No modo Fundamentado, citações são derivadas de IDs validados; no modo Compatibilidade, vínculos pós-publicação são apenas reconstruções auxiliares.",
            "O esquema 2.1 adiciona métricas inequívocas de recuperação e citação publicada, preservando aliases legados de recuperação.",
            "Trechos de contexto não originam citações e não são apresentados ao auditor como suporte factual.",
            "A atomicidade semântica do suporte é orientada ao Qwen; as contagens de trechos e páginas são descritivas, não prova de minimalidade.",
            "O arquivo avaliacao/linha_base_geracao.json pertence ao esquema anterior e não é comparável diretamente.",
        ],
    }


def carregar_relatorio_detalhado(caminho: Path) -> dict:
    """Carrega esquemas históricos sem inventar métricas ausentes."""
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, dict):
        raise ValueError("O relatório detalhado deve conter um objeto JSON.")
    return dados


def salvar_resultados_detalhados(
    resultados: Sequence[ResultadoGeracao],
    *,
    modo: str,
    diretorio: Path = PASTA_RESULTADOS_GERACAO,
    data_utc: datetime | None = None,
    tipo: str = "execucao",
    parametros_recuperacao: dict | None = None,
) -> Path:
    agora = data_utc or datetime.now(timezone.utc)
    if agora.tzinfo is None:
        agora = agora.replace(tzinfo=timezone.utc)
    diretorio.mkdir(parents=True, exist_ok=True)
    carimbo = agora.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    base = f"{carimbo}_{tipo}_{modo}_schema-{VERSAO_ESQUEMA_AVALIACAO}"
    destino = diretorio / f"{base}.json"
    sufixo = 2
    while destino.exists():
        destino = diretorio / f"{base}_{sufixo}.json"
        sufixo += 1
    relatorio = criar_relatorio_detalhado(
        resultados,
        modo=modo,
        data_utc=agora,
        tipo=tipo,
        parametros_recuperacao=parametros_recuperacao,
    )
    with destino.open("x", encoding="utf-8", newline="\n") as arquivo:
        json.dump(relatorio, arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return destino


def salvar_linha_base(resultados: Sequence[ResultadoGeracao]) -> Path:
    """Salva a linha de base corrigida sem sobrescrever o artefato legado."""
    return salvar_resultados_detalhados(
        resultados,
        modo="compatibilidade",
        tipo="linha_base_corrigida",
        parametros_recuperacao={
            "top_k": 4,
            "candidatos": MINIMO_CANDIDATOS,
            "busca_hibrida": True,
            "diversificacao_arquivos": True,
        },
    )


def executar_avaliacao_geracao(
    modo: str = "fundamentado",
    casos: Sequence[dict] | None = None,
    *,
    salvar_resultado: bool = True,
) -> ResultadosGeracao:
    casos = list(casos or carregar_casos_geracao())
    cliente = Client(host=OLLAMA_HOST, timeout=180)
    colecao = abrir_colecao()
    resultados: list[ResultadoGeracao] = []
    for indice, caso in enumerate(casos, start=1):
        inicio_caso = time.perf_counter()
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
            # Não reutiliza resultado.afirmacoes: audita novamente a resposta publicada.
            auditoria_final = auditar_resposta_publicada(
                cliente,
                resultado.resposta,
                resultado.trechos,
                caso.get("idioma", "Português"),
                evidencias=resultado.evidencias,
                afirmacoes_origem=resultado.afirmacoes,
                trechos_rotulados=resultado.trechos_rotulados,
                diagnostico=resultado.diagnostico_estrutural,
            )
            item = avaliar_saida(
                caso,
                modo,
                resultado.trechos,
                resultado.resposta,
                resultado.documento,
                auditoria_final,
                resultado.insuficiente,
                duracao_segundos=time.perf_counter() - inicio_caso,
                evidencias_geracao=resultado.evidencias,
                afirmacoes_geracao=resultado.afirmacoes,
                trechos_rotulados=resultado.trechos_rotulados,
                diagnostico_estrutural=resultado.diagnostico_estrutural,
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
            auditoria_final = auditar_resposta_publicada(
                cliente, resposta, trechos, caso.get("idioma", "Português")
            )
            item = avaliar_saida(
                caso,
                modo,
                trechos,
                resposta,
                "Vários PDFs possíveis",
                auditoria_final,
                resposta_recusou(resposta),
                duracao_segundos=time.perf_counter() - inicio_caso,
            )
        else:
            raise ValueError(f"Modo inválido: {modo}")
        resultados.append(item)

    parametros = {
        "top_k_contexto": 4 if modo == "compatibilidade" else "4 a 6",
        "candidatos_iniciais": MINIMO_CANDIDATOS,
        "busca_hibrida": True,
        "paginas_vizinhas_por_caso": {
            str(caso["pergunta"]): bool(
                caso.get("incluir_vizinhas", modo == "fundamentado")
            )
            for caso in casos
        },
        "mesmos_casos_para_comparacao": True,
        "auditoria_pos_publicacao_comum": True,
    }
    caminho = (
        salvar_resultados_detalhados(
            resultados,
            modo=modo,
            parametros_recuperacao=parametros,
        )
        if salvar_resultado else None
    )
    return ResultadosGeracao(resultados, relatorio=caminho)


def formatar_metrica_agregada(metrica: dict) -> str:
    aplicaveis = int(metrica.get("aplicaveis") or 0)
    acertos = int(metrica.get("acertos") or 0)
    if not aplicaveis:
        return "não aplicável"
    return f"{acertos}/{aplicaveis} casos aplicáveis ({acertos / aplicaveis:.1%})"
