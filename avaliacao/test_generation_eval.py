import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.chat import TrechoRecuperado
from src.generation_eval import (
    auditar_resposta_publicada,
    avaliar_conceitos,
    avaliar_requisitos_semanticos,
    avaliar_rastreabilidade_estrutural,
    avaliar_idioma,
    avaliar_saida,
    carregar_relatorio_detalhado,
    metadados_auditoria,
    normalizar_formula,
    resultado_aprovado,
    resumo_metricas,
    serializar_resultado,
    salvar_resultados_detalhados,
)
from src.grounded import (
    AfirmacaoVerificada,
    DiagnosticoEstrutural,
    EvidenciaOrganizada,
    rotular_trechos,
)


ARQUIVO = "Sinais e Sistemas/Livro.pdf"


def trecho(arquivo: str = ARQUIVO, pagina: int = 42) -> TrechoRecuperado:
    return TrechoRecuperado(
        texto="A periodic signal repeats after a period T.",
        arquivo=arquivo,
        pagina=pagina,
        indice=0,
        id=f"{arquivo}:{pagina}",
    )


def afirmacao(
    classificacao: str = "sustentada",
    *,
    pagina: int = 42,
    arquivo: str = ARQUIVO,
) -> AfirmacaoVerificada:
    original = (
        "Um sinal periódico se repete após um período. "
        f"[{arquivo}, página do PDF {pagina}]"
    )
    final = (
        "Um sinal periódico se repete após um período."
        if classificacao != "não sustentada"
        else ""
    )
    return AfirmacaoVerificada(
        texto_original=original,
        texto_final=final,
        classificacao=classificacao,
        paginas=(pagina,),
        natureza="texto_explicito",
        secao="resposta_publicada",
        justificativa="Classificação simulada para o teste.",
    )


def evidencia_estruturada() -> EvidenciaOrganizada:
    return EvidenciaOrganizada(
        id="E1",
        tipo="definicao",
        conteudo="Um sinal periódico se repete após um período.",
        natureza="texto_explicito",
        trecho_ids=("T1",),
        ids_chroma=(f"{ARQUIVO}:42",),
        arquivo=ARQUIVO,
        paginas=(42,),
    )


def afirmacao_estruturada() -> AfirmacaoVerificada:
    return AfirmacaoVerificada(
        texto_original=(
            "Um sinal periódico se repete após um período. "
            f"[{ARQUIVO}, página do PDF 42]"
        ),
        texto_final="Um sinal periódico se repete após um período.",
        classificacao="sustentada",
        paginas=(42,),
        natureza="texto_explicito",
        secao="resposta_direta",
        evidencia_ids=("E1",),
        fontes=((ARQUIVO, 42),),
        origem_vinculo="geracao_validada",
    )


def caso_resposta() -> dict:
    return {
        "tipo": "resposta direta",
        "pergunta": "O que é um sinal periódico?",
        "arquivo": ARQUIVO,
        "paginas_esperadas": [42],
        "conceitos_esperados": ["periódico"],
        "idioma": "Português",
        "espera_recusa": False,
        "observacao": "caso sintético",
    }


def resposta_citada(arquivo: str = ARQUIVO, pagina: int = 42) -> str:
    return (
        "Um sinal periódico se repete após um período. "
        f"[{arquivo}, página do PDF {pagina}]\n\n"
        f"Fontes\n- [{arquivo}, página do PDF {pagina}]"
    )


def caso_periodo_semantico() -> dict:
    return caso_resposta() | {
        "pergunta": "Como se define o período fundamental discreto?",
        "paginas_esperadas": [43, 242],
        "afirmacoes_obrigatorias": [
            {
                "id": "definicao_periodo_fundamental",
                "qualquer_de": [
                    {
                        "termos_todos": [
                            "período fundamental",
                            "menor",
                            "positivo",
                            "N",
                        ]
                    }
                ],
                "fontes_aceitaveis": [
                    {"arquivo": ARQUIVO, "paginas": [43, 242]}
                ],
            }
        ],
        "omissoes_criticas": [
            {
                "id": "periodo_deve_ser_inteiro",
                "afirmacao_id": "definicao_periodo_fundamental",
                "termos_todos": ["inteiro"],
            }
        ],
        "afirmacoes_proibidas": [
            {
                "id": "generalizacao_exponencial",
                "qualquer_de": [
                    {
                        "termos_todos": [
                            "todo sinal periódico",
                            "exponencial complexo",
                        ]
                    },
                    "sinais periódicos são complexos exponenciais",
                ],
            }
        ],
        "misturas_proibidas": ["tempo shift", "periodic sinal"],
    }


def resposta_periodo(pagina: int = 43, *, inteiro: bool = True) -> str:
    modificador = "inteiro positivo" if inteiro else "valor positivo"
    return (
        f"O período fundamental é o menor {modificador} de N para o qual "
        f"x[n] = x[n + N]. [{ARQUIVO}, página do PDF {pagina}]"
    )


def caso_formula_zoh() -> dict:
    return caso_resposta() | {
        "pergunta": "Qual é a função do ZOH?",
        "paginas_esperadas": [21],
        "afirmacoes_obrigatorias": [],
        "afirmacoes_proibidas": [],
        "omissoes_criticas": [],
        "misturas_proibidas": [],
        "formulas_esperadas": [
            {
                "id": "funcao_zoh",
                "qualquer_de": [
                    "(1-e^(-Ts))/s",
                    r"\frac{1-e^{-Ts}}{s}",
                ],
                "variaveis_obrigatorias": ["T", "s"],
            }
        ],
    }


def avaliar(
    *,
    modo: str = "fundamentado",
    trechos=None,
    resposta: str | None = None,
    afirmacoes=None,
    caso=None,
    insuficiente: bool = False,
    evidencias_geracao=None,
    afirmacoes_geracao=None,
    trechos_rotulados=None,
    diagnostico_estrutural=None,
):
    return avaliar_saida(
        caso or caso_resposta(),
        modo,
        trechos or [trecho()],
        resposta or resposta_citada(),
        ARQUIVO,
        afirmacoes if afirmacoes is not None else [afirmacao()],
        insuficiente,
        evidencias_geracao=evidencias_geracao,
        afirmacoes_geracao=afirmacoes_geracao,
        trechos_rotulados=trechos_rotulados,
        diagnostico_estrutural=diagnostico_estrutural,
    )


def avaliar_estruturado(**kwargs):
    trechos = kwargs.pop("trechos", [trecho()])
    afirmacoes_geracao = kwargs.pop(
        "afirmacoes_geracao", [afirmacao_estruturada()]
    )
    return avaliar(
        trechos=trechos,
        afirmacoes=kwargs.pop("afirmacoes", [afirmacao_estruturada()]),
        evidencias_geracao=kwargs.pop(
            "evidencias_geracao", [evidencia_estruturada()]
        ),
        afirmacoes_geracao=afirmacoes_geracao,
        trechos_rotulados=kwargs.pop(
            "trechos_rotulados", rotular_trechos(trechos)
        ),
        diagnostico_estrutural=kwargs.pop(
            "diagnostico_estrutural", DiagnosticoEstrutural()
        ),
        **kwargs,
    )


def test_fundamentado_conta_afirmacao_insegura_publicada():
    resultado = avaliar(afirmacoes=[afirmacao("não sustentada")])
    assert resultado.nao_sustentadas_publicadas == 1


def test_compatibilidade_conta_afirmacao_insegura_publicada():
    resultado = avaliar(
        modo="compatibilidade", afirmacoes=[afirmacao("não sustentada")]
    )
    assert resultado.nao_sustentadas_publicadas == 1


def test_afirmacao_parcial_publicada_continua_insegura():
    resultado = avaliar(afirmacoes=[afirmacao("parcialmente sustentada")])
    assert resultado.parcialmente_sustentadas_detectadas == 1
    assert resultado.nao_sustentadas_publicadas == 1


def test_resposta_final_sem_afirmacoes_inseguras():
    resultado = avaliar(afirmacoes=[afirmacao("sustentada")])
    assert resultado.nao_sustentadas_publicadas == 0


def test_pagina_certa_no_arquivo_errado_nao_e_fonte_correta():
    resultado = avaliar(
        trechos=[trecho("Outro.pdf", 42)],
        resposta=resposta_citada("Outro.pdf", 42),
        afirmacoes=[afirmacao(arquivo="Outro.pdf")],
    )
    assert resultado.pagina_correta is True
    assert resultado.arquivo_correto is False
    assert resultado.fonte_correta is False
    assert resultado.citacao_pagina_esperada is True
    assert resultado.citacao_fonte_esperada is False


def test_pagina_recuperada_nao_aprova_citacao_publicada_em_pagina_errada():
    resultado = avaliar(
        trechos=[trecho(ARQUIVO, 42), trecho(ARQUIVO, 43)],
        resposta=resposta_citada(ARQUIVO, 43),
        afirmacoes=[afirmacao(pagina=43)],
    )

    assert resultado.pagina_recuperada is True
    assert resultado.fonte_recuperada is True
    assert resultado.citacao_pagina_esperada is False
    assert resultado.citacao_fonte_esperada is False


def test_pagina_recuperada_e_corretamente_citada_aprova_as_duas_metricas():
    resultado = avaliar()

    assert resultado.pagina_recuperada is True
    assert resultado.citacao_pagina_esperada is True
    assert resultado.citacao_fonte_esperada is True


def test_arquivo_certo_na_pagina_errada_nao_e_fonte_correta():
    resultado = avaliar(
        trechos=[trecho(ARQUIVO, 41)],
        resposta=resposta_citada(ARQUIVO, 41),
        afirmacoes=[afirmacao(pagina=41)],
    )
    assert resultado.arquivo_correto is True
    assert resultado.pagina_correta is False
    assert resultado.fonte_correta is False


def test_arquivo_e_pagina_corretos_formam_fonte_correta():
    resultado = avaliar()
    assert resultado.arquivo_correto is True
    assert resultado.pagina_correta is True
    assert resultado.fonte_correta is True


def test_recusa_torna_paginas_conceitos_e_citacoes_nao_aplicaveis():
    caso = caso_resposta() | {
        "paginas_esperadas": [],
        "conceitos_esperados": [],
        "espera_recusa": True,
    }
    resultado = avaliar(
        caso=caso,
        resposta="Não encontrei evidência suficiente no material para responder.",
        afirmacoes=[],
        insuficiente=True,
    )
    assert resultado.arquivo_correto is None
    assert resultado.pagina_correta is None
    assert resultado.fonte_correta is None
    assert resultado.conceitos_presentes is None
    assert resultado.citacao_formal_valida is None
    assert resultado.citacao_recuperada is None
    assert resultado.citacao_pagina_esperada is None
    assert resultado.citacao_fonte_esperada is None
    assert resultado.quantidade_citacoes_unicas is None
    assert resultado.recusa_correta is True


def test_agregado_ignora_nao_aplicavel_no_denominador():
    resposta = avaliar()
    caso_recusa = caso_resposta() | {
        "paginas_esperadas": [],
        "conceitos_esperados": [],
        "espera_recusa": True,
    }
    recusa = avaliar(
        caso=caso_recusa,
        resposta="Não encontrei evidência suficiente para responder.",
        afirmacoes=[],
        insuficiente=True,
    )
    metrica = resumo_metricas([resposta, recusa])["metricas_deterministicas"][
        "pagina_correta"
    ]
    assert metrica == {"acertos": 1, "aplicaveis": 1, "taxa": 1.0}


def test_citacao_formalmente_invalida():
    resultado = avaliar(
        resposta="Um sinal periódico se repete. [Sinais e Sistemas/Livro.pdf, p. 42]",
        afirmacoes=[afirmacao()],
    )
    assert resultado.citacao_formal_valida is False


def test_citacao_valida_mas_nao_recuperada():
    resultado = avaliar(
        trechos=[trecho(ARQUIVO, 42)],
        resposta=resposta_citada(ARQUIVO, 43),
        afirmacoes=[afirmacao(pagina=43)],
    )
    assert resultado.citacao_formal_valida is True
    assert resultado.citacao_recuperada is False


def test_citacao_inline_repetida_em_fontes_e_contada_uma_vez():
    resultado = avaliar()

    assert resultado.citacoes == ((ARQUIVO, 42),)
    assert resultado.citacoes_inline == ((ARQUIVO, 42),)
    assert resultado.citacoes_bibliografia == ((ARQUIVO, 42),)
    assert resultado.quantidade_citacoes_unicas == 1
    assert resultado.citacoes_duplicadas_removidas == 1


def test_recusa_sem_citacao_nao_falha_metrica_de_citacao():
    caso = caso_resposta() | {
        "paginas_esperadas": [],
        "conceitos_esperados": [],
        "espera_recusa": True,
    }
    resultado = avaliar(
        caso=caso,
        resposta="Não encontrei a resposta no material indexado.",
        afirmacoes=[],
        insuficiente=True,
    )
    assert resultado.citacoes == ()
    assert resultado.citacao_formal_valida is None


def test_recusa_fundamentada_torna_metricas_estruturais_nao_aplicaveis():
    caso = caso_resposta() | {
        "paginas_esperadas": [],
        "conceitos_esperados": [],
        "espera_recusa": True,
    }
    diagnostico = DiagnosticoEstrutural(
        trecho_ids_invalidos_rejeitados=["T99"],
        evidencia_ids_invalidos_rejeitados=["E99"],
    )

    resultado = avaliar_estruturado(
        caso=caso,
        resposta="Não encontrei evidência suficiente no material para responder.",
        afirmacoes=[],
        afirmacoes_geracao=[],
        insuficiente=True,
        diagnostico_estrutural=diagnostico,
    )

    assert resultado.recusa_correta is True
    assert resultado.afirmacoes_com_evidencia_valida is None
    assert resultado.evidencias_com_trechos_validos is None
    assert resultado.citacoes_derivadas_evidencias is None
    assert resultado.tentativas_evidencia_inexistente is None
    assert resultado.tentativas_trecho_inexistente is None
    assert resultado.trechos_suporte_por_afirmacao is None


def test_idioma_portugues_curto():
    assert avaliar_idioma("Sinal periódico.", "Português") is True


def test_idioma_ingles_curto():
    assert avaliar_idioma("Periodic signal.", "English") is True


def test_recusa_em_portugues_e_identificada():
    texto = "Não encontrei evidência suficiente para responder."
    assert avaliar_idioma(texto, "Português") is True


def test_recusa_em_ingles_e_identificada():
    texto = "I could not find sufficient evidence to answer."
    assert avaliar_idioma(texto, "English") is True


def test_idioma_de_formula_isolada_e_indeterminado():
    assert avaliar_idioma("x(t) = x(t + T)", "Português") is None


def test_conceito_aceita_alternativa_numerica_declarada_com_virgula():
    conceitos = [
        {
            "id": "frequencia_minima",
            "qualquer_de": [
                "14 rad/s",
                {"valor": 2.2282, "unidade": "Hz", "tolerancia": 0.0005},
            ],
        }
    ]

    assert avaliar_conceitos("A frequência mínima é 14 rad/s.", conceitos) is True
    assert avaliar_conceitos("A frequência mínima é 2,2282 Hz.", conceitos) is True


@pytest.mark.parametrize(
    "resposta",
    ("A frequência é 2,2282 rad/s.", "A frequência é 2,5 Hz."),
)
def test_conceito_rejeita_unidade_incompativel_ou_valor_fora_da_tolerancia(resposta):
    conceitos = [
        {
            "id": "frequencia_minima",
            "qualquer_de": [
                {"valor": 2.2282, "unidade": "Hz", "tolerancia": 0.0005}
            ],
        }
    ]

    assert avaliar_conceitos(resposta, conceitos) is False


def test_metricas_deterministicas_cobrem_afirmacao_evidencia_trecho_e_citacao():
    resultado = avaliar_estruturado()

    assert resultado.afirmacoes_com_evidencia_valida is True
    assert resultado.evidencias_com_trechos_validos is True
    assert resultado.citacoes_derivadas_evidencias is True
    assert resultado.cobertura_evidencias_afirmacoes == 1.0
    assert resultado.afirmacoes_publicadas_sem_evidencia == 0
    agregadas = resumo_metricas([resultado])[
        "metricas_rastreabilidade_deterministicas"
    ]
    assert agregadas["cobertura_media_evidencias_afirmacoes"] == 1.0
    assert agregadas["afirmacoes_publicadas_sem_evidencia"] == 0


def test_rastreabilidade_conta_somente_suporte_e_nao_contexto():
    trechos = [trecho(pagina=42), trecho(pagina=43)]
    evidencia = EvidenciaOrganizada(
        id="E1",
        tipo="definicao",
        conteudo="Um sinal periódico se repete após um período.",
        natureza="texto_explicito",
        trecho_ids=("T1",),
        ids_chroma=(f"{ARQUIVO}:42",),
        arquivo=ARQUIVO,
        paginas=(42,),
        trecho_ids_contexto=("T2",),
        ids_chroma_contexto=(f"{ARQUIVO}:43",),
        paginas_contexto=(43,),
    )

    resultado = avaliar_estruturado(
        trechos=trechos,
        evidencias_geracao=[evidencia],
    )

    assert resultado.citacoes_derivadas_evidencias is True
    assert resultado.trechos_suporte_por_afirmacao == (1,)
    assert resultado.paginas_citadas_por_afirmacao == (1,)
    assert resultado.citacoes == ((ARQUIVO, 42),)


def test_afirmacao_publicada_sem_evidencia_reduz_cobertura():
    sem_vinculo = AfirmacaoVerificada(
        texto_original="Um sinal periódico se repete após um período.",
        texto_final="Um sinal periódico se repete após um período.",
        classificacao="sustentada",
        paginas=(),
        natureza="texto_explicito",
        secao="resposta_direta",
    )

    resultado = avaliar_estruturado(afirmacoes_geracao=[sem_vinculo])

    assert resultado.afirmacoes_com_evidencia_valida is False
    assert resultado.cobertura_evidencias_afirmacoes == 0.0
    assert resultado.afirmacoes_publicadas_sem_evidencia == 1


def test_tentativas_estruturais_rejeitadas_sao_contabilizadas():
    diagnostico = DiagnosticoEstrutural(
        trecho_ids_invalidos_rejeitados=["T99"],
        evidencia_ids_invalidos_rejeitados=["E99"],
        tentativas_mistura_arquivos=1,
    )

    resultado = avaliar_estruturado(diagnostico_estrutural=diagnostico)

    assert resultado.tentativas_trecho_inexistente == 1
    assert resultado.tentativas_evidencia_inexistente == 1
    assert resultado.tentativas_mistura_arquivos == 1


def test_modo_compatibilidade_nao_inventa_ids_de_evidencia():
    resultado = avaliar(modo="compatibilidade")

    assert resultado.origem_vinculos_evidencia == "reconstruidos_auxiliares"
    assert resultado.afirmacoes_com_evidencia_valida is None
    assert resultado.evidencias_com_trechos_validos is None
    assert resultado.citacoes_derivadas_evidencias is None
    assert resultado.tentativas_evidencia_inexistente is None
    assert resultado.trechos_suporte_por_afirmacao is None
    assert resultado.paginas_citadas_por_afirmacao is None
    rastreabilidade = resumo_metricas([resultado])[
        "metricas_rastreabilidade_deterministicas"
    ]
    assert rastreabilidade["trechos_suporte_por_afirmacao"] is None
    assert rastreabilidade["paginas_citadas_por_afirmacao"] is None


def test_serializacao_do_relatorio_detalhado(tmp_path):
    destino = salvar_resultados_detalhados(
        [avaliar_estruturado()],
        modo="fundamentado",
        diretorio=tmp_path,
        data_utc=datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc),
    )
    dados = json.loads(destino.read_text(encoding="utf-8"))
    assert dados["versao_esquema"] == "2.2"
    assert dados["ambiente"]["sistema_operacional"]
    assert dados["modelos"]["avaliacao_independente"] is False
    assert dados["casos"][0]["resposta_final"] == resposta_citada()
    caso = dados["casos"][0]
    assert caso["trechos_rotulados"][0]["rotulo"] == "T1"
    assert caso["evidencias_geracao"][0]["id"] == "E1"
    assert caso["evidencias_geracao"][0]["ids_chroma"] == [f"{ARQUIVO}:42"]
    assert caso["rastreabilidade"]["afirmacao_para_evidencias"][0][
        "evidencia_ids"
    ] == ["E1"]
    assert caso["rastreabilidade"]["evidencia_para_trechos"][0][
        "trecho_ids_suporte"
    ] == ["T1"]
    assert caso["citacoes_publicadas"]["quantidade_unicas"] == 1
    assert caso["citacoes_publicadas"]["duplicadas_removidas"] == 1
    assert str(Path.home()) not in destino.read_text(encoding="utf-8")


def test_relatorio_antigo_carrega_sem_inventar_metricas_novas(tmp_path):
    antigo = tmp_path / "relatorio-2.0.json"
    antigo.write_text(
        json.dumps(
            {
                "versao_esquema": "2.0",
                "metricas": {
                    "metricas_deterministicas": {"pagina_correta": True}
                },
            }
        ),
        encoding="utf-8",
    )

    carregado = carregar_relatorio_detalhado(antigo)

    assert carregado["versao_esquema"] == "2.0"
    assert "pagina_recuperada" not in carregado["metricas"]["metricas_deterministicas"]
    assert json.loads(json.dumps(carregado)) == carregado


def test_salvamento_preserva_resultado_anterior(tmp_path):
    momento = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    primeiro = salvar_resultados_detalhados(
        [avaliar()], modo="fundamentado", diretorio=tmp_path, data_utc=momento
    )
    conteudo_original = primeiro.read_bytes()
    segundo = salvar_resultados_detalhados(
        [avaliar()], modo="fundamentado", diretorio=tmp_path, data_utc=momento
    )
    assert segundo != primeiro
    assert primeiro.read_bytes() == conteudo_original


def test_gerador_e_auditor_iguais_nao_sao_avaliacao_independente():
    metadados = metadados_auditoria("qwen2.5:3b", "qwen2.5:3b")
    assert metadados["gerador_e_auditor_iguais"] is True
    assert metadados["avaliacao_independente"] is False


def test_auditoria_comum_usa_resposta_publicada_sem_secao_fontes(monkeypatch):
    capturado = {}

    def auditor_simulado(cliente, rascunho, trechos, idioma):
        capturado["rascunho"] = rascunho
        return [afirmacao()]

    monkeypatch.setattr(
        "src.generation_eval.verificar_afirmacoes", auditor_simulado
    )
    resultado = auditar_resposta_publicada(
        object(), resposta_citada(), [trecho()], "Português"
    )
    assert len(resultado) == 1
    assert len(capturado["rascunho"]) == 1
    assert "Fontes" not in capturado["rascunho"][0]["texto"]


def test_auditoria_publicada_fundamentada_reusa_ids_da_geracao(monkeypatch):
    capturado = {}

    def auditor_simulado(cliente, rascunho, trechos, idioma, **kwargs):
        capturado["rascunho"] = rascunho
        capturado["evidencias"] = kwargs["evidencias"]
        return [afirmacao_estruturada()]

    monkeypatch.setattr(
        "src.generation_eval.verificar_afirmacoes", auditor_simulado
    )
    auditadas = auditar_resposta_publicada(
        object(),
        resposta_citada(),
        [trecho()],
        "Português",
        evidencias=[evidencia_estruturada()],
        afirmacoes_origem=[afirmacao_estruturada()],
        trechos_rotulados=rotular_trechos([trecho()]),
    )

    assert auditadas[0].evidencia_ids == ("E1",)
    assert capturado["rascunho"][0]["evidencia_ids"] == ["E1"]
    assert capturado["evidencias"][0].trecho_ids == ("T1",)


def test_recusa_publicada_e_registrada_mas_nao_auditada_como_fato(monkeypatch):
    chamado = False

    def auditor_simulado(*args, **kwargs):
        nonlocal chamado
        chamado = True
        return []

    monkeypatch.setattr(
        "src.generation_eval.verificar_afirmacoes", auditor_simulado
    )
    resposta = "Não encontrei evidência suficiente no PDF para responder."
    auditadas = auditar_resposta_publicada(
        object(), resposta, [trecho()], "Português"
    )
    resultado = avaliar(
        caso=caso_resposta() | {
            "paginas_esperadas": [],
            "conceitos_esperados": [],
            "espera_recusa": True,
        },
        resposta=resposta,
        afirmacoes=auditadas,
        insuficiente=True,
    )
    assert chamado is False
    assert resultado.afirmacoes_publicadas == (resposta,)
    assert resultado.nao_sustentadas_publicadas == 0


def test_pagina_242_e_aceita_como_fonte_alternativa_legitima():
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=242)],
        resposta=resposta_periodo(242),
        afirmacoes=[afirmacao(pagina=242)],
    )

    assert resultado.afirmacoes_obrigatorias_presentes is True
    assert resultado.fontes_aceitaveis_citadas is True
    assert resultado.omissoes_criticas == 0


def test_pagina_recuperada_mas_nao_citada_nao_aprova_requisito():
    resposta = (
        "O período fundamental é o menor inteiro positivo de N para o qual "
        "x[n] = x[n + N]."
    )
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=242)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_obrigatorias_presentes is True
    assert resultado.fontes_aceitaveis_citadas is False


def test_fonte_citada_na_afirmacao_errada_nao_aprova_requisito():
    resposta = (
        "O período fundamental é o menor inteiro positivo de N. "
        f"Outro conceito usa N como índice. [{ARQUIVO}, página do PDF 242]"
    )
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=242)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_obrigatorias_presentes is True
    assert resultado.citacao_fonte_esperada is True
    assert resultado.fontes_aceitaveis_citadas is False


def test_menor_valor_positivo_falha_sem_modificador_inteiro():
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta_periodo(43, inteiro=False),
    )

    assert resultado.afirmacoes_obrigatorias_presentes is False
    assert resultado.fontes_aceitaveis_citadas is True
    assert resultado.omissoes_criticas == 1
    diagnostico = next(
        item
        for item in resultado.diagnosticos_semanticos
        if item.id == "periodo_deve_ser_inteiro"
    )
    assert diagnostico.termos_ausentes == ("inteiro",)


def test_menor_inteiro_positivo_aprova_modificador_critico():
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta_periodo(43),
    )

    assert resultado.afirmacoes_obrigatorias_presentes is True
    assert resultado.omissoes_criticas == 0


def test_generalizacao_de_sinal_periodico_e_reprovada():
    resposta = (
        resposta_periodo(43)
        + " Todo sinal periódico é exponencial complexo."
    )
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_proibidas_ausentes is False
    assert any(
        item.id == "generalizacao_exponencial" and item.estado == "reprovado"
        for item in resultado.diagnosticos_semanticos
    )


def test_generalizacao_com_ordem_complexos_exponenciais_e_reprovada():
    resposta = (
        resposta_periodo(43)
        + " Os sinais periódicos são complexos exponenciais."
    )
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_proibidas_ausentes is False


def test_flexao_plural_e_variacao_iguais_apos_aprovam_definicao():
    caso = json.loads(
        (Path(__file__).parent / "casos_geracao.json").read_text(encoding="utf-8")
    )[0]
    resposta = (
        "Os sinais periódicos são iguais após um deslocamento de período T. "
        f"[{caso['arquivo']}, página do PDF 43]"
    )
    resultado = avaliar(
        caso=caso,
        trechos=[trecho(arquivo=caso["arquivo"], pagina=43)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_obrigatorias_presentes is True
    assert resultado.fontes_aceitaveis_citadas is True
    assert resultado.omissoes_criticas == 0


def test_explicacao_correta_sobre_exponencial_periodico_nao_e_proibida():
    resposta = (
        resposta_periodo(43)
        + " Alguns exponenciais complexos de tempo discreto são periódicos "
        "quando sua frequência satisfaz a condição apropriada."
    )
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_proibidas_ausentes is True


@pytest.mark.parametrize("mistura", ("tempo shift", "periodic sinal"))
def test_mistura_terminologica_declarada_e_reprovada(mistura):
    resposta = resposta_periodo(43) + f" O conceito usa {mistura}."
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta,
    )

    assert resultado.misturas_proibidas_ausentes is False


def test_formula_zoh_linear_e_integra():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = (1 - e^(-T s)) / s.",
    )

    assert resultado.formulas_integras is True


def test_formula_zoh_latex_e_integra():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta=r"$G_{zoh}(s) = \frac{1-e^{-Ts}}{s}$.",
    )

    assert resultado.formulas_integras is True
    assert normalizar_formula(r"\frac{1-e^{-Ts}}{s}") == normalizar_formula(
        "(1-e^(-Ts))/s"
    )


def test_formula_zoh_sem_denominador_e_reprovada():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = 1 - e^(-Ts).",
    )

    assert resultado.formulas_integras is False


@pytest.mark.parametrize(
    "formula",
    ("(1-e^(-T))/s", "(1-e^(Ts))/s", "1-e-Ts/s"),
)
def test_formula_zoh_com_expoente_ou_sinal_incorreto_e_reprovada(formula):
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta=f"Gzoh(s) = {formula}.",
    )

    assert resultado.formulas_integras is False


def test_formula_zoh_corrompida_pelo_ocr_e_reprovada():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = 1 − ϵ − T s s.",
    )

    assert resultado.formulas_integras is False


def test_formula_zoh_com_termo_extra_nao_e_equivalente():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = (1-e^(-Ts))/s + 1.",
    )

    assert resultado.formulas_integras is False


def test_formula_zoh_aceita_parenteses_redundantes_e_multiplicacao_unicode():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = (((1-e^(-T×s))/s)).",
    )

    assert resultado.formulas_integras is True


def test_termos_espalhados_em_afirmacoes_nao_aprovam_requisito():
    caso = caso_periodo_semantico()
    resposta = (
        "O período fundamental é o menor valor. "
        f"N deve ser inteiro positivo. [{ARQUIVO}, página do PDF 43]"
    )
    resultado = avaliar(
        caso=caso,
        trechos=[trecho(pagina=43)],
        resposta=resposta,
    )

    assert resultado.afirmacoes_obrigatorias_presentes is False


def test_recusa_mantem_metricas_semanticas_nao_aplicaveis():
    caso = caso_periodo_semantico() | {
        "paginas_esperadas": [],
        "conceitos_esperados": [],
        "espera_recusa": True,
    }
    resultado = avaliar(
        caso=caso,
        resposta="Não encontrei evidência suficiente no material para responder.",
        afirmacoes=[],
        insuficiente=True,
    )

    assert resultado.afirmacoes_obrigatorias_presentes is None
    assert resultado.fontes_aceitaveis_citadas is None
    assert resultado.formulas_integras is None
    assert resultado.omissoes_criticas is None
    assert resultado.requisitos_semanticos_aplicaveis is None
    assert {item.estado for item in resultado.diagnosticos_semanticos} == {"N/A"}


def test_caso_antigo_sem_esquema_semantico_continua_funcionando():
    resultado = avaliar(caso=caso_resposta())

    assert resultado.conceitos_presentes is True
    assert resultado.afirmacoes_obrigatorias_presentes is None
    assert resultado.requisitos_semanticos_aplicaveis is None


def test_auditoria_qwen_auxiliar_nao_decide_aprovacao_deterministica():
    resultado = avaliar(afirmacoes=[afirmacao("não sustentada")])

    assert resultado.citacao_sustenta_afirmacao is False
    assert resultado_aprovado(resultado) is True


def test_resumo_agrega_contagens_semanticas_sem_trata_las_como_booleanos():
    resultado = avaliar(
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta_periodo(43, inteiro=False),
    )

    metricas = resumo_metricas([resultado])["metricas_deterministicas"]
    assert metricas["omissoes_criticas"] == {
        "total": 1,
        "casos_aplicaveis": 1,
    }
    assert metricas["requisitos_semanticos_aplicaveis"]["total"] > 0


@pytest.mark.parametrize("versao", ("2.0", "2.1"))
def test_relatorios_historicos_carregam_e_serializam_sem_novos_campos(
    tmp_path, versao
):
    caminho = tmp_path / f"relatorio-{versao}.json"
    original = {
        "versao_esquema": versao,
        "metricas": {"metricas_deterministicas": {"pagina_correta": True}},
    }
    caminho.write_text(json.dumps(original), encoding="utf-8")

    carregado = carregar_relatorio_detalhado(caminho)

    assert carregado == original
    assert json.loads(json.dumps(carregado)) == original


def test_modo_compatibilidade_preserva_metricas_estruturais_indisponiveis():
    resultado = avaliar(
        modo="compatibilidade",
        caso=caso_periodo_semantico(),
        trechos=[trecho(pagina=43)],
        resposta=resposta_periodo(43),
    )

    assert resultado.afirmacoes_com_evidencia_valida is None
    assert resultado.evidencias_com_trechos_validos is None
    assert resultado.citacoes_derivadas_evidencias is None
    assert resultado.afirmacoes_obrigatorias_presentes is True


def test_relatorio_expoe_motivo_deterministico_da_reprovacao():
    resultado = avaliar(
        caso=caso_formula_zoh(),
        trechos=[trecho(pagina=21)],
        resposta="Gzoh(s) = 1 - e^(-Ts).",
    )

    serializado = serializar_resultado(resultado)
    diagnostico = serializado["diagnostico_semantico"][0]
    assert diagnostico["estado"] == "reprovado"
    assert "estrutura matemática íntegra" in diagnostico["motivo_deterministico"]


def test_expansao_oppenheim_tem_oito_casos_com_gabarito_completo():
    casos = json.loads(
        (Path(__file__).parent / "casos_geracao.json").read_text(encoding="utf-8")
    )
    perguntas = {
        "Por que um sinal constante não possui período fundamental definido?",
        "Por que o sinal definido como cos(t) para t < 0 e sin(t) para t >= 0 não é periódico?",
        "Um sistema pode ser linear e variante no tempo ou não linear e invariante no tempo?",
        "Qual é a condição necessária e suficiente sobre h[n] para estabilidade BIBO de um sistema LTI discreto?",
        "Por que exponenciais complexos são autofunções de sistemas LTI?",
        "Qual diferença de periodicidade existe entre DTFT e CTFT?",
        "Se uma senoide com frequência 5ωs/6 for amostrada com frequência ωs, qual frequência aparece após reconstrução passa-baixas?",
        "Para H(s) = b/(s-a), com a > 0 e realimentação proporcional G(s)=K, qual condição estabiliza o sistema?",
    }
    expandidos = [caso for caso in casos if caso["pergunta"] in perguntas]

    assert len(expandidos) == 8
    for caso in expandidos:
        assert caso["afirmacoes_obrigatorias"]
        assert caso["omissoes_criticas"]
        assert caso["afirmacoes_proibidas"]
        assert caso["misturas_proibidas"]
        assert "resposta_esperada" in caso or "valor_esperado" in caso


def test_caso_periodo_discreto_declara_paginas_43_e_242():
    casos = json.loads(
        (Path(__file__).parent / "casos_geracao.json").read_text(encoding="utf-8")
    )
    caso = next(
        item
        for item in casos
        if item["pergunta"].startswith("Como o livro define o período fundamental")
    )

    assert caso["paginas_esperadas"] == [43, 242]
    assert caso["afirmacoes_obrigatorias"][0]["fontes_aceitaveis"][0][
        "paginas"
    ] == [43, 242]
