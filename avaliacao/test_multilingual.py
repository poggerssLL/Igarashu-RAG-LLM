from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.chat import (
    TrechoRecuperado,
    fundir_resultados,
    gerar_resposta,
    recuperar_trechos,
    selecionar_contexto,
    termos_consulta,
)
from src.index_manifest import (
    ErroManifesto,
    criar_manifesto,
    validar_compatibilidade,
)


LIVRO = "Sinais e Sistemas/Signals_and_Systems_2nd_Edition_by_Oppen.pdf"


def test_termos_portugueses_recebem_expansao_inglesa():
    termos = termos_consulta("O que caracteriza os sinais periódicos?")

    assert {"signals", "periodic", "class"}.issubset(termos)


def test_fusao_bilingue_promove_pagina_42():
    vetoriais = [
        TrechoRecuperado("outro", LIVRO, 200 + i, 0, 0.3 + i / 100, id=f"v{i}")
        for i in range(14)
    ]
    pagina_42 = TrechoRecuperado(
        "A class of signals with important properties are periodic signals.",
        LIVRO,
        42,
        0,
        0.8,
        id="alvo",
    )
    vetoriais.append(pagina_42)
    lexicais = [pagina_42]

    ranking = fundir_resultados(vetoriais, lexicais)

    assert ranking[0].pagina == 42
    assert ranking[0].arquivo.endswith("Signals_and_Systems_2nd_Edition_by_Oppen.pdf")


def test_recuperacao_solicita_no_minimo_vinte_candidatos(monkeypatch):
    chamada = {}

    def busca(colecao, embedding, top_k, disciplina=None):
        chamada["top_k"] = top_k
        return [TrechoRecuperado("periodic signals", LIVRO, 42, 0, 0.2)]

    monkeypatch.setattr("src.chat.buscar_trechos", busca)
    monkeypatch.setattr("src.chat.buscar_por_palavras_chave", lambda *args: [])

    recuperar_trechos("colecao", "periodic signals", [1.0], top_k=4)

    assert chamada["top_k"] >= 20


def test_diversificacao_so_troca_quando_relevancia_e_semelhante():
    candidatos = [
        TrechoRecuperado("A1", "a.pdf", 1, 0, pontuacao_fusao=1.0),
        TrechoRecuperado("A2", "a.pdf", 2, 0, pontuacao_fusao=0.95),
        TrechoRecuperado("B fraco", "b.pdf", 1, 0, pontuacao_fusao=0.40),
    ]
    escolhidos = selecionar_contexto(candidatos, 2, diversificar_arquivos=True)
    assert [item.texto for item in escolhidos] == ["A1", "A2"]

    semelhantes = [
        candidatos[0],
        candidatos[1],
        TrechoRecuperado("B similar", "b.pdf", 1, 0, pontuacao_fusao=0.92),
    ]
    escolhidos = selecionar_contexto(semelhantes, 2, diversificar_arquivos=True)
    assert [item.texto for item in escolhidos] == ["A1", "B similar"]


def test_pergunta_inglesa_contexto_ingles_resposta_obrigatoria_em_portugues():
    cliente = Mock()
    cliente.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="Um sinal periódico se repete após um período.")
    )
    trechos = [TrechoRecuperado("A periodic signal repeats after T.", LIVRO, 42, 0)]

    resposta = gerar_resposta(
        cliente,
        "What characterizes periodic signals?",
        trechos,
        idioma_resposta="Português",
    )

    mensagens = cliente.chat.call_args.kwargs["messages"]
    assert mensagens[0]["role"] == "system"
    assert "exclusivamente em Português" in mensagens[0]["content"]
    assert "What characterizes periodic signals?" in mensagens[1]["content"]
    assert "A periodic signal" in mensagens[1]["content"]
    assert resposta.startswith("Um sinal periódico")
    assert "página do PDF 42" in resposta


def test_modelo_diferente_do_manifesto_e_bloqueado():
    manifesto = criar_manifesto("nomic-embed-text", 768, 100)

    with pytest.raises(ErroManifesto, match="incompatível"):
        validar_compatibilidade("embeddinggemma", manifesto)
