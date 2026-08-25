"""Consulta RAG local com recuperação híbrida, fontes verificadas e Ollama."""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import chromadb
from ollama import Client, ResponseError

from .config import (
    MINIMO_CANDIDATOS,
    MODELO_CONVERSA,
    MODELO_EMBEDDINGS,
    NOME_COLECAO,
    OLLAMA_HOST,
    PASTA_CHROMA,
)
from .disciplinas import SEM_DISCIPLINA, TODAS_DISCIPLINAS
from .index_manifest import (
    ErroManifesto,
    ManifestoIndice,
    carregar_manifesto,
    validar_compatibilidade,
)

IDIOMAS_RESPOSTA = ("Português", "English")


class ErroConsulta(RuntimeError):
    """Erro esperado, apresentado ao usuário sem traceback."""


@dataclass(frozen=True)
class TrechoRecuperado:
    texto: str
    arquivo: str
    pagina: int
    indice: int
    distancia: float | None = None
    disciplina: str = SEM_DISCIPLINA
    id: str = ""
    pontuacao_palavras: float = 0.0
    pontuacao_fusao: float = 0.0
    pagina_vizinha: bool = False

    @property
    def relevancia(self) -> float | None:
        if self.distancia is not None:
            return 1.0 / (1.0 + max(0.0, self.distancia))
        return self.pontuacao_fusao or None


def _nome_modelo(modelo: object) -> str:
    if isinstance(modelo, dict):
        return str(modelo.get("model") or modelo.get("name") or "")
    return str(getattr(modelo, "model", None) or getattr(modelo, "name", ""))


def _modelo_instalado(nome: str, instalados: set[str]) -> bool:
    base = nome.split(":", 1)[0]
    return any(item == nome or item.split(":", 1)[0] == base for item in instalados)


def verificar_ollama_e_modelos(
    cliente: Client, modelo_embeddings: str = MODELO_EMBEDDINGS
) -> None:
    try:
        resposta = cliente.list()
    except Exception as erro:
        raise ErroConsulta(
            f"Ollama indisponível em {OLLAMA_HOST}. Inicie o Ollama e tente novamente."
        ) from erro
    modelos = getattr(resposta, "models", None)
    if modelos is None and isinstance(resposta, dict):
        modelos = resposta.get("models", [])
    instalados = {_nome_modelo(modelo) for modelo in (modelos or [])}
    ausentes = [
        nome
        for nome in (modelo_embeddings, MODELO_CONVERSA)
        if not _modelo_instalado(nome, instalados)
    ]
    if ausentes:
        comandos = ", ".join(f"ollama pull {nome}" for nome in ausentes)
        raise ErroConsulta(
            f"Modelo(s) ausente(s): {', '.join(ausentes)}. Instale com: {comandos}"
        )


def abrir_colecao(pasta_chroma: Path = PASTA_CHROMA) -> object:
    if not pasta_chroma.exists() or not (pasta_chroma / "chroma.sqlite3").exists():
        raise ErroConsulta(
            f"Banco vetorial inexistente em '{pasta_chroma}'. Execute primeiro a ingestão."
        )
    try:
        cliente = chromadb.PersistentClient(path=pasta_chroma)
        colecao = cliente.get_collection(name=NOME_COLECAO)
    except Exception as erro:
        raise ErroConsulta(
            f"A coleção '{NOME_COLECAO}' não foi encontrada em '{pasta_chroma}'. "
            "Execute novamente a ingestão."
        ) from erro
    if colecao.count() == 0:
        raise ErroConsulta("O banco vetorial existe, mas não contém documentos indexados.")
    return colecao


def manifesto_compativel(
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> ManifestoIndice:
    try:
        manifesto = carregar_manifesto()
        validar_compatibilidade(modelo_embeddings, manifesto, indice_tem_dados=True)
    except ErroManifesto as erro:
        raise ErroConsulta(str(erro)) from erro
    assert manifesto is not None
    return manifesto


def gerar_embedding_pergunta(
    cliente: Client, pergunta: str, modelo_embeddings: str = MODELO_EMBEDDINGS
) -> list[float]:
    try:
        resposta = cliente.embed(model=modelo_embeddings, input=pergunta)
    except ResponseError as erro:
        if erro.status_code == 404:
            raise ErroConsulta(f"Modelo de embeddings ausente: {modelo_embeddings}.") from erro
        raise ErroConsulta(f"O Ollama recusou a geração do embedding: {erro.error}") from erro
    except Exception as erro:
        raise ErroConsulta(
            f"Ollama indisponível em {OLLAMA_HOST} durante a geração do embedding."
        ) from erro
    if not resposta.embeddings:
        raise ErroConsulta("O Ollama não retornou o embedding da pergunta.")
    return list(resposta.embeddings[0])


def _chave(trecho: TrechoRecuperado) -> str:
    return trecho.id or f"{trecho.arquivo.casefold()}|{trecho.pagina}|{trecho.indice}"


def _converter_resultado(resultado: dict) -> list[TrechoRecuperado]:
    documentos = (resultado.get("documents") or [[]])[0]
    metadados = (resultado.get("metadatas") or [[]])[0]
    distancias = (resultado.get("distances") or [[]])[0]
    ids = (resultado.get("ids") or [[]])[0]
    trechos: list[TrechoRecuperado] = []
    for posicao, (texto, metadado) in enumerate(zip(documentos, metadados)):
        if not texto or not metadado:
            continue
        trechos.append(
            TrechoRecuperado(
                texto=str(texto),
                arquivo=str(
                    metadado.get("arquivo")
                    or metadado.get("caminho_relativo")
                    or metadado.get("nome_arquivo")
                    or "desconhecido"
                ),
                pagina=int(metadado.get("pagina", 0)),
                indice=int(metadado.get("indice_trecho", 0)),
                distancia=float(distancias[posicao]) if posicao < len(distancias) else None,
                disciplina=str(metadado.get("disciplina") or SEM_DISCIPLINA),
                id=str(ids[posicao]) if posicao < len(ids) else "",
            )
        )
    return trechos


def filtro_chroma(
    disciplina: str | None = None, arquivo: str | None = None
) -> dict | None:
    filtros = []
    if disciplina:
        filtros.append({"disciplina": disciplina})
    if arquivo:
        filtros.append({"arquivo": arquivo})
    if not filtros:
        return None
    return filtros[0] if len(filtros) == 1 else {"$and": filtros}


def listar_arquivos_indexados(
    colecao: object, disciplina: str | None = None
) -> list[str]:
    argumentos: dict = {"include": ["metadatas"]}
    if disciplina:
        argumentos["where"] = {"disciplina": disciplina}
    dados = colecao.get(**argumentos)
    return sorted(
        {
            str(metadata.get("arquivo") or metadata.get("caminho_relativo"))
            for metadata in (dados.get("metadatas") or [])
            if metadata.get("arquivo") or metadata.get("caminho_relativo")
        },
        key=str.casefold,
    )


def buscar_trechos(
    colecao: object,
    embedding: Sequence[float],
    top_k: int = 4,
    disciplina: str | None = None,
    arquivo: str | None = None,
) -> list[TrechoRecuperado]:
    """Busca vetorial básica, mantida como bloco reutilizável e compatível."""
    if top_k <= 0:
        raise ErroConsulta("--top-k deve ser um número inteiro maior que zero.")
    filtro = filtro_chroma(disciplina, arquivo)
    if filtro:
        ids = colecao.get(where=filtro, limit=top_k, include=[]).get("ids") or []
        if not ids:
            raise ErroConsulta(
                "Não há documentos indexados para os filtros selecionados "
                f"(disciplina={disciplina or 'todas'}, arquivo={arquivo or 'automático'})."
            )
        quantidade = min(top_k, len(ids))
    else:
        quantidade = min(top_k, colecao.count())
    argumentos = {
        "query_embeddings": [list(embedding)],
        "n_results": quantidade,
        "include": ["documents", "metadatas", "distances"],
    }
    if filtro:
        argumentos["where"] = filtro
    try:
        resultado = colecao.query(**argumentos)
    except Exception as erro:
        raise ErroConsulta(f"Falha ao consultar o banco vetorial: {erro}") from erro
    trechos = _converter_resultado(resultado)
    if not trechos:
        raise ErroConsulta("A busca não retornou nenhum trecho utilizável.")
    return trechos


_STOPWORDS = {
    "a", "as", "o", "os", "de", "da", "das", "do", "dos", "e", "em", "um", "uma",
    "que", "qual", "quais", "como", "para", "por", "the", "an", "of", "and", "in",
    "what", "which", "how", "is", "are", "to",
}
_LEXICO_BILINGUE = {
    "sinal": {"signal"}, "sinais": {"signal", "signals"},
    "periodico": {"periodic"}, "periodicos": {"periodic"},
    "caracteriza": {"characterizes", "characterized", "class"},
    "classe": {"class"}, "tempo": {"time"}, "continuo": {"continuous"},
    "discreto": {"discrete"}, "frequencia": {"frequency"}, "periodo": {"period"},
    "amostragem": {"sampling"}, "sistema": {"system"}, "sistemas": {"systems"},
    "repete": {"repeat", "repeats", "periodic"},
    "repetir": {"repeat", "periodic"},
    "regularmente": {"periodic"},
    "condicao": {"condition", "property"},
    "inalterado": {"unchanged"},
    "deslocamento": {"shift"},
    "define": {"defined", "definition", "smallest", "value"},
    "definido": {"defined", "smallest", "value"},
    "definicao": {"definition", "smallest", "value"},
}


def normalizar_termos(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [
        termo for termo in re.findall(r"[a-z0-9]+", texto)
        if len(termo) > 1 and termo not in _STOPWORDS
    ]


def termos_consulta(pergunta: str) -> set[str]:
    termos = set(normalizar_termos(pergunta))
    for termo in tuple(termos):
        termos.update(_LEXICO_BILINGUE.get(termo, set()))
    return termos


def buscar_por_palavras_chave(
    colecao: object,
    pergunta: str,
    quantidade: int,
    disciplina: str | None = None,
    arquivo: str | None = None,
) -> list[TrechoRecuperado]:
    """Ranking lexical local com peso IDF sobre todos os chunks filtrados."""
    argumentos: dict = {"include": ["documents", "metadatas"]}
    if filtro := filtro_chroma(disciplina, arquivo):
        argumentos["where"] = filtro
    try:
        resultado = colecao.get(**argumentos)
    except Exception as erro:
        raise ErroConsulta(f"Falha na busca por palavras-chave: {erro}") from erro
    documentos = resultado.get("documents") or []
    metadados = resultado.get("metadatas") or []
    ids = resultado.get("ids") or []
    consulta = termos_consulta(pergunta)
    if not consulta:
        return []
    frequencia_documental = Counter()
    tokens_documentos: list[list[str]] = []
    for documento in documentos:
        tokens = normalizar_termos(str(documento or ""))
        tokens_documentos.append(tokens)
        frequencia_documental.update(set(tokens) & consulta)
    total = max(1, len(documentos))
    candidatos: list[TrechoRecuperado] = []
    for posicao, (documento, metadata, tokens) in enumerate(
        zip(documentos, metadados, tokens_documentos)
    ):
        contagens = Counter(tokens)
        pontuacao = 0.0
        for termo in consulta:
            if termo not in contagens:
                continue
            idf = math.log(
                1.0 + (total - frequencia_documental[termo] + 0.5)
                / (frequencia_documental[termo] + 0.5)
            )
            pontuacao += idf * (1.0 + math.log(contagens[termo]))
        if pontuacao <= 0:
            continue
        candidatos.append(
            TrechoRecuperado(
                texto=str(documento),
                arquivo=str(
                    metadata.get("arquivo") or metadata.get("caminho_relativo")
                    or metadata.get("nome_arquivo") or "desconhecido"
                ),
                pagina=int(metadata.get("pagina", 0)),
                indice=int(metadata.get("indice_trecho", 0)),
                disciplina=str(metadata.get("disciplina") or SEM_DISCIPLINA),
                id=str(ids[posicao]) if posicao < len(ids) else "",
                pontuacao_palavras=pontuacao,
            )
        )
    candidatos.sort(key=lambda item: item.pontuacao_palavras, reverse=True)
    return candidatos[:quantidade]


def fundir_resultados(
    vetoriais: Sequence[TrechoRecuperado],
    lexicais: Sequence[TrechoRecuperado],
    constante_rrf: int = 60,
) -> list[TrechoRecuperado]:
    """Fusão por Reciprocal Rank Fusion sem misturar escalas de pontuação."""
    itens: dict[str, TrechoRecuperado] = {}
    pontos: Counter[str] = Counter()
    for ranking in (vetoriais, lexicais):
        for posicao, trecho in enumerate(ranking, start=1):
            chave = _chave(trecho)
            pontos[chave] += 1.0 / (constante_rrf + posicao)
            anterior = itens.get(chave)
            if anterior is None:
                itens[chave] = trecho
            else:
                itens[chave] = replace(
                    anterior,
                    distancia=anterior.distancia if anterior.distancia is not None else trecho.distancia,
                    pontuacao_palavras=max(anterior.pontuacao_palavras, trecho.pontuacao_palavras),
                )
    if not pontos:
        return []
    maior_lexical = max(
        (item.pontuacao_palavras for item in itens.values()), default=0.0
    )
    if maior_lexical:
        for chave, item in itens.items():
            # O RRF preserva os rankings; este bônus retém a intensidade de uma
            # correspondência lexical muito mais específica que as demais.
            pontos[chave] += 0.5 * item.pontuacao_palavras / maior_lexical
    maximo = max(pontos.values())
    fundidos = [
        replace(item, pontuacao_fusao=pontos[chave] / maximo)
        for chave, item in itens.items()
    ]
    return sorted(fundidos, key=lambda item: item.pontuacao_fusao, reverse=True)


def selecionar_contexto(
    candidatos: Sequence[TrechoRecuperado],
    top_k: int,
    *,
    diversificar_arquivos: bool = True,
    tolerancia: float = 0.08,
) -> list[TrechoRecuperado]:
    restantes = list(candidatos)
    selecionados: list[TrechoRecuperado] = []
    arquivos: set[str] = set()
    while restantes and len(selecionados) < top_k:
        melhor = restantes[0]
        escolhido = melhor
        if diversificar_arquivos and melhor.arquivo in arquivos:
            limite = melhor.pontuacao_fusao * (1.0 - tolerancia)
            alternativo = next(
                (
                    item for item in restantes[1:]
                    if item.arquivo not in arquivos and item.pontuacao_fusao >= limite
                ),
                None,
            )
            if alternativo is not None:
                escolhido = alternativo
        selecionados.append(escolhido)
        arquivos.add(escolhido.arquivo)
        restantes.remove(escolhido)
    return selecionados


def adicionar_paginas_vizinhas(
    colecao: object,
    selecionados: Sequence[TrechoRecuperado],
    pergunta: str,
    disciplina: str | None = None,
    arquivo: str | None = None,
) -> list[TrechoRecuperado]:
    argumentos: dict = {"include": ["documents", "metadatas"]}
    if filtro := filtro_chroma(disciplina, arquivo):
        argumentos["where"] = filtro
    dados = colecao.get(**argumentos)
    ids = dados.get("ids") or []
    consulta = termos_consulta(pergunta)
    mapa: dict[tuple[str, int], list[TrechoRecuperado]] = {}
    for posicao, (texto, metadata) in enumerate(
        zip(dados.get("documents") or [], dados.get("metadatas") or [])
    ):
        trecho = TrechoRecuperado(
            texto=str(texto),
            arquivo=str(metadata.get("arquivo") or metadata.get("caminho_relativo") or "desconhecido"),
            pagina=int(metadata.get("pagina", 0)),
            indice=int(metadata.get("indice_trecho", 0)),
            disciplina=str(metadata.get("disciplina") or SEM_DISCIPLINA),
            id=str(ids[posicao]) if posicao < len(ids) else "",
            pagina_vizinha=True,
        )
        mapa.setdefault((trecho.arquivo, trecho.pagina), []).append(trecho)
    resposta = list(selecionados)
    vistos = {_chave(item) for item in resposta}
    for origem in selecionados:
        for pagina in (origem.pagina - 1, origem.pagina + 1):
            opcoes = mapa.get((origem.arquivo, pagina), [])
            if not opcoes:
                continue
            melhor = max(
                opcoes,
                key=lambda item: len(set(normalizar_termos(item.texto)) & consulta),
            )
            if _chave(melhor) not in vistos:
                resposta.append(melhor)
                vistos.add(_chave(melhor))
    return resposta


def recuperar_trechos(
    colecao: object,
    pergunta: str,
    embedding: Sequence[float],
    top_k: int = 4,
    *,
    candidatos: int = MINIMO_CANDIDATOS,
    disciplina: str | None = None,
    busca_hibrida: bool = True,
    incluir_vizinhas: bool = False,
    diversificar_arquivos: bool = True,
    arquivo: str | None = None,
) -> tuple[list[TrechoRecuperado], list[TrechoRecuperado]]:
    quantidade = max(MINIMO_CANDIDATOS, candidatos, top_k)
    vetoriais = (
        buscar_trechos(colecao, embedding, quantidade, disciplina, arquivo)
        if arquivo
        else buscar_trechos(colecao, embedding, quantidade, disciplina)
    )
    if busca_hibrida:
        lexicais = (
            buscar_por_palavras_chave(
                colecao, pergunta, quantidade, disciplina, arquivo
            )
            if arquivo
            else buscar_por_palavras_chave(
                colecao, pergunta, quantidade, disciplina
            )
        )
        ranking = fundir_resultados(vetoriais, lexicais)
    else:
        ranking = [
            replace(item, pontuacao_fusao=1.0 / posicao)
            for posicao, item in enumerate(vetoriais, start=1)
        ]
    selecionados = selecionar_contexto(
        ranking, top_k, diversificar_arquivos=diversificar_arquivos
    )
    if incluir_vizinhas:
        selecionados = adicionar_paginas_vizinhas(
            colecao, selecionados, pergunta, disciplina, arquivo
        )
    return selecionados, ranking


def filtrar_por_relevancia(
    trechos: Sequence[TrechoRecuperado], min_relevancia: float
) -> list[TrechoRecuperado]:
    if not 0.0 <= min_relevancia <= 1.0:
        raise ErroConsulta("--min-relevancia deve estar entre 0 e 1.")
    if min_relevancia == 0.0:
        return list(trechos)
    aceitos = [
        trecho for trecho in trechos
        if trecho.relevancia is not None and trecho.relevancia >= min_relevancia
    ]
    if not aceitos:
        melhor = max((item.relevancia or 0.0 for item in trechos), default=0.0)
        raise ErroConsulta(
            "Não encontrei trechos com relevância suficiente para responder. "
            f"Melhor relevância: {melhor:.3f}; mínimo exigido: {min_relevancia:.3f}."
        )
    return aceitos


def fontes_unicas(trechos: Sequence[TrechoRecuperado]) -> list[tuple[str, int]]:
    fontes: list[tuple[str, int]] = []
    vistos: set[tuple[str, int]] = set()
    for trecho in trechos:
        chave = (trecho.arquivo, trecho.pagina)
        if chave not in vistos:
            vistos.add(chave)
            fontes.append(chave)
    return fontes


def remover_secao_fontes(resposta: str) -> str:
    return re.split(
        r"(?im)^\s*(?:#+\s*)?fontes\s*:?\s*$", resposta, maxsplit=1
    )[0].strip()


def anexar_fontes(resposta: str, trechos: Sequence[TrechoRecuperado]) -> str:
    corpo = remover_secao_fontes(resposta)
    referencias = "\n".join(
        f"- [{arquivo}, página do PDF {pagina}]"
        for arquivo, pagina in fontes_unicas(trechos)
    )
    return f"{corpo}\n\nFontes\n{referencias}"


def mensagem_sistema(idioma_resposta: str = "Português") -> str:
    if idioma_resposta not in IDIOMAS_RESPOSTA:
        raise ErroConsulta(f"Idioma de resposta inválido: {idioma_resposta}.")
    destino = "PORTUGUÊS DO BRASIL" if idioma_resposta == "Português" else "ENGLISH"
    return (
        f"REGRA DE MAIOR PRIORIDADE: escreva TODA a resposta em {destino}. "
        "O idioma da pergunta e do contexto não altera essa regra; traduza mentalmente o "
        f"material antes de responder. Responda exclusivamente em {idioma_resposta}. "
        "Use somente o contexto fornecido e não use conhecimento externo."
    )


def resposta_no_idioma(texto: str, idioma_resposta: str) -> bool:
    """Heurística conservadora para detectar uma resposta claramente no idioma oposto."""
    termos = set(normalizar_termos(texto))
    marcadores_pt = {"uma", "sao", "sinal", "periodico", "periodicos", "caracterizado", "repete", "apos"}
    marcadores_en = {"the", "are", "signal", "signals", "periodic", "characterized", "repeats", "after"}
    pontos_pt = len(termos & marcadores_pt)
    pontos_en = len(termos & marcadores_en)
    if idioma_resposta == "Português":
        return pontos_en == 0 or pontos_pt >= pontos_en
    return pontos_pt == 0 or pontos_en >= pontos_pt


def montar_prompt(
    pergunta: str,
    trechos: Sequence[TrechoRecuperado],
    idioma_resposta: str = "Português",
) -> str:
    contexto = "\n\n".join(
        f"[Trecho {numero}]\nArquivo: {trecho.arquivo}\n"
        f"Disciplina: {trecho.disciplina}\nPágina do PDF: {trecho.pagina}\n"
        f"Conteúdo: {trecho.texto}"
        for numero, trecho in enumerate(trechos, start=1)
    )
    insuficiente = (
        "Não encontrei a resposta no material indexado."
        if idioma_resposta == "Português"
        else "I could not find the answer in the indexed material."
    )
    return f"""Regras obrigatórias:
1. Responda somente com base nos trechos fornecidos abaixo.
2. Não invente, complete ou suponha informações ausentes.
3. Se o contexto não for suficiente, responda exatamente: \"{insuficiente}\"
4. Cite afirmações no formato [arquivo, página do PDF N].
5. Não crie uma seção Fontes; referências verificadas serão adicionadas pelo programa.
6. Não atribua nomes a teoremas, autores ou métodos se não aparecerem nos trechos.
7. Preserve símbolos, fórmulas, unidades e desigualdades do material.
8. Idioma obrigatório da resposta: {idioma_resposta}.

Pergunta:
{pergunta}

Trechos recuperados:
{contexto}
"""


def gerar_resposta(
    cliente: Client,
    pergunta: str,
    trechos: Sequence[TrechoRecuperado],
    idioma_resposta: str = "Português",
) -> str:
    mensagens = [
        {"role": "system", "content": mensagem_sistema(idioma_resposta)},
        {"role": "user", "content": montar_prompt(pergunta, trechos, idioma_resposta)},
    ]

    def chamar(mensagens_chat: list[dict]) -> object:
        return cliente.chat(
            model=MODELO_CONVERSA,
            messages=mensagens_chat,
            stream=False,
            options={"temperature": 0, "num_predict": 512},
        )

    try:
        resposta = chamar(mensagens)
    except ResponseError as erro:
        if erro.status_code == 404:
            raise ErroConsulta(f"Modelo de conversa ausente: {MODELO_CONVERSA}.") from erro
        raise ErroConsulta(f"O Ollama recusou a geração da resposta: {erro.error}") from erro
    except Exception as erro:
        raise ErroConsulta(
            f"Ollama indisponível em {OLLAMA_HOST} durante a geração da resposta."
        ) from erro
    mensagem = getattr(resposta, "message", None)
    texto = (
        getattr(mensagem, "content", None)
        or (mensagem.get("content") if isinstance(mensagem, dict) else None)
        or ""
    ).strip()
    if not texto:
        raise ErroConsulta("O modelo de conversa retornou uma resposta vazia.")
    if not resposta_no_idioma(texto, idioma_resposta):
        destino = "PORTUGUÊS DO BRASIL" if idioma_resposta == "Português" else "ENGLISH"
        mensagens_correcao = [
            {
                "role": "system",
                "content": (
                    f"Você é um tradutor técnico. Produza exclusivamente a tradução em {destino}, "
                    "sem comentários, sem responder novamente à pergunta e sem adicionar ou remover fatos."
                ),
            },
            {
                "role": "user",
                "content": f"Traduza o texto a seguir para {destino}:\n\n{texto}",
            },
        ]
        try:
            resposta = chamar(mensagens_correcao)
            mensagem = getattr(resposta, "message", None)
            texto = (
                getattr(mensagem, "content", None)
                or (mensagem.get("content") if isinstance(mensagem, dict) else None)
                or ""
            ).strip()
        except Exception as erro:
            raise ErroConsulta("Não foi possível corrigir o idioma da resposta.") from erro
        if not texto or not resposta_no_idioma(texto, idioma_resposta):
            raise ErroConsulta(
                f"O modelo não respeitou o idioma selecionado ({idioma_resposta}). Tente novamente."
            )
    return anexar_fontes(texto, trechos)


def consultar(
    pergunta: str,
    top_k: int = 4,
    *,
    cliente_ollama: Client | None = None,
    colecao: object | None = None,
    min_relevancia: float = 0.0,
    disciplina: str | None = None,
    candidatos: int = MINIMO_CANDIDATOS,
    busca_hibrida: bool = True,
    incluir_vizinhas: bool = False,
    diversificar_arquivos: bool = True,
    idioma_resposta: str = "Português",
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> tuple[list[TrechoRecuperado], str]:
    pergunta = pergunta.strip()
    if not pergunta:
        raise ErroConsulta("A pergunta não pode estar vazia.")
    if top_k <= 0:
        raise ErroConsulta("--top-k deve ser um número inteiro maior que zero.")
    colecao = colecao or abrir_colecao()
    manifesto = manifesto_compativel(modelo_embeddings)
    cliente_ollama = cliente_ollama or Client(host=OLLAMA_HOST)
    verificar_ollama_e_modelos(cliente_ollama, modelo_embeddings)
    vetor = gerar_embedding_pergunta(cliente_ollama, pergunta, modelo_embeddings)
    if len(vetor) != manifesto.dimensao:
        raise ErroConsulta(
            f"Dimensão incompatível: consulta={len(vetor)}, índice={manifesto.dimensao}. "
            "Reindexe toda a biblioteca com o modelo configurado."
        )
    trechos, _ = recuperar_trechos(
        colecao,
        pergunta,
        vetor,
        top_k,
        candidatos=candidatos,
        disciplina=disciplina,
        busca_hibrida=busca_hibrida,
        incluir_vizinhas=incluir_vizinhas,
        diversificar_arquivos=diversificar_arquivos,
    )
    trechos = filtrar_por_relevancia(trechos, min_relevancia)
    return trechos, gerar_resposta(cliente_ollama, pergunta, trechos, idioma_resposta)


def mostrar_contexto(trechos: Sequence[TrechoRecuperado]) -> None:
    print("Trechos recuperados")
    for numero, trecho in enumerate(trechos, start=1):
        relevancia = trecho.relevancia or 0.0
        tipo = " | página vizinha" if trecho.pagina_vizinha else ""
        print(
            f"\n[{numero}] [{trecho.arquivo}, página do PDF {trecho.pagina}] "
            f"| disciplina: {trecho.disciplina} | relevância: {relevancia:.4f} "
            f"| fusão: {trecho.pontuacao_fusao:.4f}{tipo}"
        )
        print(trecho.texto)


def _inteiro_positivo(valor: str) -> int:
    try:
        numero = int(valor)
    except ValueError as erro:
        raise argparse.ArgumentTypeError("deve ser um número inteiro") from erro
    if numero <= 0:
        raise argparse.ArgumentTypeError("deve ser maior que zero")
    return numero


def _relevancia(valor: str) -> float:
    try:
        numero = float(valor.replace(",", "."))
    except ValueError as erro:
        raise argparse.ArgumentTypeError("deve ser um número entre 0 e 1") from erro
    if not 0.0 <= numero <= 1.0:
        raise argparse.ArgumentTypeError("deve estar entre 0 e 1")
    return numero


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Responde perguntas usando somente os PDFs indexados no RAG local."
    )
    parser.add_argument("pergunta")
    parser.add_argument("--top-k", type=_inteiro_positivo, default=4)
    parser.add_argument("--candidatos", type=_inteiro_positivo, default=MINIMO_CANDIDATOS)
    parser.add_argument("--min-relevancia", type=_relevancia, default=0.0)
    parser.add_argument("--mostrar-contexto", action="store_true")
    parser.add_argument("--disciplina")
    parser.add_argument("--sem-busca-hibrida", action="store_true")
    parser.add_argument("--paginas-vizinhas", action="store_true")
    parser.add_argument("--sem-diversificacao", action="store_true")
    parser.add_argument("--idioma", choices=IDIOMAS_RESPOSTA, default="Português")
    parser.add_argument("--modelo-embeddings", default=MODELO_EMBEDDINGS)
    parser.add_argument(
        "--modo",
        choices=("fundamentado", "compatibilidade"),
        default="fundamentado",
        help="fundamentado escolhe um único PDF e audita afirmações; compatibilidade mantém o fluxo anterior",
    )
    parser.add_argument(
        "--arquivo",
        help="restringe a consulta fundamentada a um caminho de PDF indexado",
    )
    parser.add_argument(
        "--nivel-detalhe",
        choices=("Curto", "Explicado", "Passo a passo"),
        default="Explicado",
    )
    parser.add_argument("--mostrar-evidencias", action="store_true")
    args = parser.parse_args()
    disciplina = (
        None if not args.disciplina or args.disciplina == TODAS_DISCIPLINAS
        else args.disciplina
    )
    try:
        if args.modo == "fundamentado":
            from .grounded import consultar_fundamentado

            resultado = consultar_fundamentado(
                args.pergunta,
                disciplina=disciplina,
                arquivo=args.arquivo,
                idioma=args.idioma,
                nivel_detalhe=args.nivel_detalhe,
                candidatos=args.candidatos,
                incluir_vizinhas=args.paginas_vizinhas,
                modelo_embeddings=args.modelo_embeddings,
            )
            trechos = list(resultado.trechos)
            resposta = resultado.resposta
            print(f"Documento escolhido: {resultado.documento}")
            print(f"Motivo: {resultado.motivo_documento}")
            if args.mostrar_evidencias:
                print("Evidências organizadas")
                for evidencia in resultado.evidencias:
                    print(
                        f"- {evidencia.tipo} | página do PDF {evidencia.pagina} "
                        f"| {evidencia.natureza}: {evidencia.conteudo}"
                    )
        else:
            trechos, resposta = consultar(
                args.pergunta,
                args.top_k,
                min_relevancia=args.min_relevancia,
                disciplina=disciplina,
                candidatos=args.candidatos,
                busca_hibrida=not args.sem_busca_hibrida,
                incluir_vizinhas=args.paginas_vizinhas,
                diversificar_arquivos=not args.sem_diversificacao,
                idioma_resposta=args.idioma,
                modelo_embeddings=args.modelo_embeddings,
            )
    except ErroConsulta as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    print(f"Escopo: {disciplina or TODAS_DISCIPLINAS} | idioma: {args.idioma}")
    if args.mostrar_contexto:
        mostrar_contexto(trechos)
        print()
    print("Resposta")
    print(resposta)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
