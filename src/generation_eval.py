"""Avaliação reproduzível da resposta final dos modos de geração locais."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
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
from .grounded import AfirmacaoVerificada, consultar_fundamentado, verificar_afirmacoes
from .index_manifest import carregar_manifesto


ARQUIVO_CASOS_GERACAO = RAIZ_PROJETO / "avaliacao" / "casos_geracao.json"
ARQUIVO_LINHA_BASE = RAIZ_PROJETO / "avaliacao" / "linha_base_geracao.json"
PASTA_RESULTADOS_GERACAO = RAIZ_PROJETO / "avaliacao" / "resultados"
VERSAO_ESQUEMA_AVALIACAO = "2.0"
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

    @property
    def recuperou_pagina(self) -> bool | None:
        """Alias legado para consumidores antigos da avaliação."""
        return self.pagina_correta

    @property
    def citacoes_validas(self) -> bool | None:
        """Compatibilidade: exige formato e pertencimento ao contexto recuperado."""
        if self.citacao_formal_valida is None or self.citacao_recuperada is None:
            return None
        return self.citacao_formal_valida and self.citacao_recuperada

    @property
    def metricas_deterministicas(self) -> dict[str, bool | None]:
        return {
            "arquivo_correto": self.arquivo_correto,
            "pagina_correta": self.pagina_correta,
            "fonte_correta": self.fonte_correta,
            "conceitos_presentes": self.conceitos_presentes,
            "citacao_formal_valida": self.citacao_formal_valida,
            "citacao_recuperada": self.citacao_recuperada,
            "recusa_correta": self.recusa_correta,
            "idioma_correto": self.idioma_correto,
            "resposta_presente": self.resposta_presente,
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


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(item for item in texto if not unicodedata.combining(item))
    return " ".join(texto.split())


def normalizar_caminho(caminho: str) -> str:
    return caminho.strip().replace("\\", "/").casefold()


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


def analisar_citacoes(
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    aplicavel: bool,
) -> tuple[bool | None, bool | None, tuple[tuple[str, int], ...]]:
    citacoes = tuple(extrair_citacoes(resposta))
    if not aplicavel:
        return None, None, citacoes
    aparentes = _PADRAO_CITACAO_APARENTE.findall(resposta)
    formal = bool(citacoes) and len(citacoes) == len(aparentes)
    permitidas = {
        (normalizar_caminho(item.arquivo), item.pagina) for item in trechos
    }
    recuperada = formal and all(
        (normalizar_caminho(arquivo), pagina) in permitidas
        for arquivo, pagina in citacoes
    )
    return formal, recuperada, citacoes


def validar_citacoes(
    resposta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    exigir: bool,
) -> bool:
    """API legada: valida formato e pertencimento aos trechos."""
    formal, recuperada, _ = analisar_citacoes(resposta, trechos, aplicavel=exigir)
    return True if not exigir else bool(formal and recuperada)


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
        paginas_citadas = [pagina for _, pagina in extrair_citacoes(sentenca)]
        paginas = [
            pagina for pagina in paginas_citadas if pagina in paginas_disponiveis
        ] or paginas_disponiveis
        rascunho.append(
            {
                "texto": sentenca,
                "secao": "resposta_publicada",
                "paginas": paginas,
                "natureza": "texto_explicito",
            }
        )
    return verificar_afirmacoes(cliente, rascunho, trechos, idioma)


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
        paginas_citadas = {pagina for _, pagina in citacoes}
        if (
            afirmacao.classificacao != "sustentada"
            or not citacoes
            or not paginas_citadas.intersection(afirmacao.paginas)
        ):
            return False
    return True


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
) -> ResultadoGeracao:
    paginas = tuple(dict.fromkeys(item.pagina for item in trechos))
    fontes = tuple(dict.fromkeys((item.arquivo, item.pagina) for item in trechos))
    esperadas = {int(item) for item in caso.get("paginas_esperadas", [])}
    arquivo_esperado = str(caso.get("arquivo") or "").strip()
    arquivo_normalizado = normalizar_caminho(arquivo_esperado)
    termos = [normalizar(str(item)) for item in caso.get("conceitos_esperados", [])]
    texto = normalizar(resposta)
    espera_recusa = bool(caso.get("espera_recusa"))
    espera_resposta = not espera_recusa
    recusou = insuficiente or resposta_recusou(resposta)

    arquivos_retornados = {normalizar_caminho(item.arquivo) for item in trechos}
    arquivo_correto = (
        arquivo_normalizado in arquivos_retornados
        if espera_resposta and arquivo_normalizado
        else None
    )
    pagina_correta = (
        bool(esperadas.intersection(paginas))
        if espera_resposta and esperadas
        else None
    )
    fonte_correta = (
        any(
            normalizar_caminho(item.arquivo) == arquivo_normalizado
            and item.pagina in esperadas
            for item in trechos
        )
        if espera_resposta and arquivo_normalizado and esperadas
        else None
    )
    conceitos_presentes = (
        all(item in texto for item in termos)
        if espera_resposta and termos
        else None
    )
    formal, recuperada, citacoes = analisar_citacoes(
        resposta, trechos, aplicavel=espera_resposta
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
    expectativa = {
        "arquivo": arquivo_esperado,
        "paginas_esperadas": sorted(esperadas),
        "conceitos_esperados": list(caso.get("conceitos_esperados", [])),
        "idioma": caso.get("idioma", "Português"),
        "espera_recusa": espera_recusa,
    }
    return ResultadoGeracao(
        pergunta=str(caso["pergunta"]),
        modo=modo,
        tipo_caso=str(caso.get("tipo") or ""),
        expectativa=expectativa,
        arquivo_correto=arquivo_correto,
        pagina_correta=pagina_correta,
        fonte_correta=fonte_correta,
        conceitos_presentes=conceitos_presentes,
        citacao_formal_valida=formal,
        citacao_recuperada=recuperada,
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
        citacoes=citacoes,
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
        "arquivo_correto",
        "pagina_correta",
        "fonte_correta",
        "conceitos_presentes",
        "citacao_formal_valida",
        "citacao_recuperada",
        "idioma_correto",
        "resposta_presente",
        "recusa_correta",
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
        "metricas_auxiliares_qwen": auxiliares,
        # Chaves legadas para a interface e consumidores existentes.
        "recuperacao_pagina": taxa("pagina_correta"),
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


def serializar_resultado(item: ResultadoGeracao) -> dict:
    return {
        "duracao_segundos": item.duracao_segundos,
        "tipo": item.tipo_caso,
        "pergunta": item.pergunta,
        "expectativa": item.expectativa,
        "resposta_final": item.resposta,
        "afirmacoes_publicadas_extraidas": list(item.afirmacoes_publicadas),
        "documento_escolhido": item.documento,
        "trechos_recuperados": [_serializar_trecho(trecho) for trecho in item.trechos],
        "paginas_retornadas": list(item.paginas_retornadas),
        "fontes_retornadas": [
            {"arquivo": arquivo, "pagina_pdf": pagina}
            for arquivo, pagina in item.fontes_retornadas
        ],
        "citacoes": [
            {"arquivo": arquivo, "pagina_pdf": pagina}
            for arquivo, pagina in item.citacoes
        ],
        "afirmacoes_auditadas": [
            _serializar_afirmacao(afirmacao)
            for afirmacao in item.afirmacoes_auditadas
        ],
        "metricas_deterministicas": item.metricas_deterministicas,
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
        "casos": [serializar_resultado(item) for item in resultados],
        "observacoes_e_limitacoes": [
            AVISO_AUDITORIA_QWEN,
            "Métricas determinísticas não aplicáveis são registradas como null e excluídas dos denominadores.",
            "Frases programáticas de recusa são registradas como texto publicado, mas avaliadas por recusa_correta em vez da auditoria factual do Qwen.",
            "O arquivo avaliacao/linha_base_geracao.json pertence ao esquema anterior e não é comparável diretamente.",
        ],
    }


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
