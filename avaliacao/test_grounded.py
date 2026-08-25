import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.chat import TrechoRecuperado, filtro_chroma
from src.grounded import (
    AfirmacaoVerificada,
    montar_resposta_verificada,
    organizar_evidencias,
    remover_quase_duplicados,
    selecionar_evidencias,
    selecionar_fonte,
    verificar_afirmacoes,
)


def trecho(texto, arquivo="livro.pdf", pagina=1, score=1.0, indice=0):
    return TrechoRecuperado(
        texto, arquivo, pagina, indice, pontuacao_fusao=score, id=f"{arquivo}-{pagina}-{indice}"
    )


def resposta_json(dados):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(dados, ensure_ascii=False)))


def test_filtro_combina_disciplina_e_arquivo():
    assert filtro_chroma("Sinais", "Sinais/livro.pdf") == {
        "$and": [
            {"disciplina": "Sinais"},
            {"arquivo": "Sinais/livro.pdf"},
        ]
    }


def test_selecao_automatica_mantem_documento_com_evidencia_mais_forte():
    candidatos = [
        trecho("definição principal", "forte.pdf", 42, 1.0),
        trecho("continuação", "forte.pdf", 43, 0.9),
        trecho("menção isolada", "outro.pdf", 5, 0.91),
    ]

    arquivo, motivo = selecionar_fonte(candidatos)

    assert arquivo == "forte.pdf"
    assert "segunda busca" in motivo


def test_pdf_manual_e_respeitado():
    candidatos = [
        trecho("forte", "a.pdf", 1, 1.0),
        trecho("manual", "b.pdf", 2, 0.5),
    ]

    arquivo, motivo = selecionar_fonte(candidatos, "b.pdf")

    assert arquivo == "b.pdf"
    assert "manualmente" in motivo


def test_remove_trechos_quase_identicos_e_nao_preenche_com_fracos():
    candidatos = [
        trecho("Um sinal periódico repete seus valores após o período T.", pagina=42, score=1.0),
        trecho("Um sinal periódico repete seus valores depois do período T.", pagina=42, score=0.98, indice=1),
        trecho("O menor período positivo é o período fundamental.", pagina=43, score=0.85),
        trecho("texto sem relação", pagina=500, score=0.1),
    ]

    unicos = remover_quase_duplicados(candidatos, limite=0.75)
    selecionados = selecionar_evidencias(unicos, minimo=2, maximo=6)

    assert len([item for item in unicos if item.pagina == 42]) == 1
    assert all(item.pagina != 500 for item in selecionados)


def test_organizador_descarta_pagina_que_nao_foi_recuperada():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "informacao_faltante": "",
            "evidencias": [
                {"tipo": "definicao", "conteudo": "Sinal se repete.", "pagina": 42, "natureza": "texto_explicito"},
                {"tipo": "fato", "conteudo": "Inventado.", "pagina": 999, "natureza": "texto_explicito"},
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente, "O que é periódico?", [trecho("Periodic signal", pagina=42)]
    )

    assert suficiente
    assert [item.pagina for item in evidencias] == [42]


def test_organizador_assume_texto_explicito_quando_natureza_omitida():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {"tipo": "fato", "conteudo": "Sinal se repete.", "pagina": 42}
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente, "Pergunta", [trecho("Periodic signal", pagina=42)]
    )

    assert suficiente
    assert evidencias[0].natureza == "texto_explicito"


def test_verificador_remove_afirmacao_sem_suporte_e_citacao_invalida():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {"afirmacoes": [
            {
                "id": "A1",
                "texto_original": "Sinais periódicos se repetem.",
                "texto_final": "Sinais periódicos se repetem.",
                "classificacao": "sustentada",
                "paginas": [42],
                "natureza": "texto_explicito",
                "secao": "resposta_direta",
                "justificativa": "Está explícito.",
            },
            {
                "id": "A2",
                "texto_original": "Todo sistema é estável.",
                "texto_final": "Todo sistema é estável.",
                "classificacao": "sustentada",
                "paginas": [999],
                "natureza": "texto_explicito",
                "secao": "explicacao",
                "justificativa": "Página não autorizada.",
            },
        ]}
    )
    verificadas = verificar_afirmacoes(
        cliente,
        [
            {"texto": "Sinais periódicos se repetem."},
            {"texto": "Todo sistema é estável."},
        ],
        [trecho("Periodic signals repeat.", pagina=42)],
        "Português",
    )
    resposta, insuficiente = montar_resposta_verificada(
        "livro.pdf", verificadas, "Explicado"
    )

    assert not insuficiente
    assert "Sinais periódicos se repetem" in resposta
    assert "Todo sistema é estável" not in resposta
    assert "página do PDF 42" in resposta
    assert "999" not in resposta
    assert verificadas[1].classificacao == "não sustentada"


def test_resposta_distingue_deducao_simples():
    afirmacao = AfirmacaoVerificada(
        texto_original="Logo, T é positivo.",
        texto_final="T deve ser positivo.",
        classificacao="sustentada",
        paginas=(42,),
        natureza="deducao_simples",
        secao="explicacao",
    )

    resposta, _ = montar_resposta_verificada(
        "livro.pdf", [afirmacao], "Curto"
    )

    assert "Dedução simples" in resposta
    assert "página do PDF 42" in resposta


def test_sem_afirmacao_aprovada_recusa_e_informa_o_que_falta():
    resposta, insuficiente = montar_resposta_verificada(
        "livro.pdf", [], "Curto", "falta a especificação do motor"
    )

    assert insuficiente
    assert "Não encontrei evidência suficiente" in resposta
    assert "falta a especificação do motor" in resposta
