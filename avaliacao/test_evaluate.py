from src.chat import TrechoRecuperado
from src.evaluate import avaliar_caso, normalizar_texto, taxa_acerto_recuperacao
import src.evaluate as evaluate


def _caso():
    return {
        "pergunta": "O que é aliasing?",
        "paginas_esperadas": [17],
        "termos_esperados": ["frequência", "anti-aliasing"],
        "observacao": "caso unitário",
    }


def test_normalizacao_corrige_artefatos_comuns_de_pdf():
    assert normalizar_texto("Frequˆência e quantiza¸c˜ao") == "frequencia e quantizacao"
    assert normalizar_texto("E∗(s), s´ıncrona e −π") == "e*(s), sincrona e -π"


def test_avaliacao_detecta_pagina_e_termos():
    trechos = [
        TrechoRecuperado(
            "A frequˆencia pode exigir um filtro anti-aliasing.",
            "aula.pdf",
            17,
            0,
            0.2,
        )
    ]

    resultado = avaliar_caso(_caso(), trechos)

    assert resultado.acertou_pagina
    assert resultado.paginas_retornadas == [17]
    assert resultado.termos_ausentes == []


def test_avaliacao_registra_falha_e_termo_ausente():
    trechos = [TrechoRecuperado("Conteúdo diferente", "aula.pdf", 3, 0, 0.8)]

    resultado = avaliar_caso(_caso(), trechos)

    assert not resultado.acertou_pagina
    assert resultado.termos_encontrados == []
    assert resultado.termos_ausentes == ["frequência", "anti-aliasing"]


def test_taxa_de_acerto():
    sucesso = avaliar_caso(_caso(), [TrechoRecuperado("frequência anti-aliasing", "a.pdf", 17, 0)])
    falha = avaliar_caso(_caso(), [TrechoRecuperado("outro", "a.pdf", 2, 0)])

    assert taxa_acerto_recuperacao([sucesso, falha]) == 0.5


def test_avaliacao_encaminha_filtro_opcional_de_disciplina(monkeypatch):
    caso = _caso()
    monkeypatch.setattr(evaluate, "carregar_casos", lambda: [caso])
    monkeypatch.setattr(evaluate, "abrir_colecao", lambda: "colecao")
    monkeypatch.setattr(evaluate, "Client", lambda host: "cliente")
    monkeypatch.setattr(
        evaluate,
        "manifesto_compativel",
        lambda modelo: type("Manifesto", (), {"dimensao": 1})(),
    )
    monkeypatch.setattr(
        evaluate,
        "gerar_embedding_pergunta",
        lambda cliente, pergunta, modelo: [1.0],
    )
    chamada = {}

    def busca(colecao, pergunta, vetor, top_k, **opcoes):
        chamada["disciplina"] = opcoes["disciplina"]
        return [TrechoRecuperado("frequência anti-aliasing", "a.pdf", 17, 0)], []

    monkeypatch.setattr(evaluate, "recuperar_trechos", busca)

    resultados = evaluate.executar_avaliacao("Sinais e Sistemas")

    assert chamada["disciplina"] == "Sinais e Sistemas"
    assert resultados[0].acertou_pagina
