from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.chat import (
    ErroConsulta,
    TrechoRecuperado,
    anexar_fontes,
    buscar_trechos,
    filtrar_por_relevancia,
    fontes_unicas,
    gerar_resposta,
    montar_prompt,
)
from src.config import MODELO_CONVERSA


def test_busca_converte_resultado_mockado_do_chroma():
    colecao = Mock()
    colecao.count.return_value = 10
    colecao.query.return_value = {
        "documents": [["Primeiro trecho", "Segundo trecho"]],
        "metadatas": [[
            {"arquivo": "aula.pdf", "pagina": 2, "indice_trecho": 0},
            {"arquivo": "aula.pdf", "pagina": 4, "indice_trecho": 1},
        ]],
        "distances": [[0.12, 0.25]],
    }

    trechos = buscar_trechos(colecao, [0.1, 0.2], top_k=2)

    assert [trecho.texto for trecho in trechos] == ["Primeiro trecho", "Segundo trecho"]
    assert trechos[0].arquivo == "aula.pdf"
    assert trechos[1].pagina == 4
    colecao.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


def test_busca_limita_top_k_ao_total_de_documentos():
    colecao = Mock()
    colecao.count.return_value = 1
    colecao.query.return_value = {
        "documents": [["Trecho"]],
        "metadatas": [[{"arquivo": "fonte.pdf", "pagina": 1}]],
        "distances": [[0.1]],
    }

    buscar_trechos(colecao, [1.0], top_k=4)

    assert colecao.query.call_args.kwargs["n_results"] == 1


def test_prompt_impoe_resposta_fundamentada_e_fontes():
    prompt = montar_prompt(
        "O que é amostragem?",
        [TrechoRecuperado("Conteúdo da aula", "sinais.pdf", 8, 0)],
    )

    assert "somente com base nos trechos" in prompt
    assert "Não encontrei a resposta no material indexado." in prompt
    assert "Não atribua nomes a teoremas" in prompt
    assert "referências verificadas" in prompt
    assert "sinais.pdf" in prompt
    assert "Página do PDF: 8" in prompt


def test_resposta_do_modelo_e_gerada_com_mock():
    cliente = Mock()
    cliente.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content="A amostragem converte um sinal contínuo.\n\nFontes\n- sinais.pdf"
        )
    )
    trechos = [TrechoRecuperado("Conteúdo", "sinais.pdf", 8, 0)]

    resposta = gerar_resposta(cliente, "O que é amostragem?", trechos)

    assert "Fontes" in resposta
    cliente.chat.assert_called_once()
    assert cliente.chat.call_args.kwargs["model"] == MODELO_CONVERSA
    mensagens = cliente.chat.call_args.kwargs["messages"]
    assert mensagens[0]["role"] == "system"
    assert "exclusivamente em Português" in mensagens[0]["content"]
    assert "O que é amostragem?" in mensagens[1]["content"]
    assert "Conteúdo" in mensagens[1]["content"]


def test_busca_vazia_produz_erro_claro():
    colecao = Mock()
    colecao.count.return_value = 1
    colecao.query.return_value = {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    with pytest.raises(ErroConsulta, match="nenhum trecho"):
        buscar_trechos(colecao, [1.0], top_k=1)


def test_fontes_duplicadas_sao_removidas_e_formatadas():
    trechos = [
        TrechoRecuperado("A", "aula.pdf", 7, 0),
        TrechoRecuperado("B", "aula.pdf", 7, 1),
        TrechoRecuperado("C", "aula.pdf", 8, 0),
    ]

    assert fontes_unicas(trechos) == [("aula.pdf", 7), ("aula.pdf", 8)]
    resposta = anexar_fontes("Resposta\n\nFontes:\n- fonte inventada", trechos)
    assert resposta.count("[aula.pdf, página do PDF 7]") == 1
    assert resposta.count("[aula.pdf, página do PDF 8]") == 1
    assert "fonte inventada" not in resposta


def test_relevancia_minima_bloqueia_trechos_fracos():
    trechos = [TrechoRecuperado("A", "aula.pdf", 1, 0, distancia=1.0)]

    with pytest.raises(ErroConsulta, match="relevância suficiente"):
        filtrar_por_relevancia(trechos, min_relevancia=0.6)


def test_busca_aplica_filtro_de_disciplina_no_chroma():
    colecao = Mock()
    colecao.get.return_value = {"ids": ["id-1"]}
    colecao.query.return_value = {
        "documents": [["Trecho de sinais"]],
        "metadatas": [[{
            "arquivo": "Sinais/aula.pdf",
            "pagina": 2,
            "disciplina": "Sinais e Sistemas",
        }]],
        "distances": [[0.2]],
    }

    trechos = buscar_trechos(
        colecao, [0.1], top_k=4, disciplina="Sinais e Sistemas"
    )

    colecao.get.assert_called_once_with(
        where={"disciplina": "Sinais e Sistemas"}, limit=4, include=[]
    )
    assert colecao.query.call_args.kwargs["where"] == {
        "disciplina": "Sinais e Sistemas"
    }
    assert trechos[0].disciplina == "Sinais e Sistemas"
