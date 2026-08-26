import json
from types import SimpleNamespace
from unittest.mock import Mock

from src.chat import TrechoRecuperado, filtro_chroma
from src.grounded import (
    AfirmacaoVerificada,
    DiagnosticoEstrutural,
    EvidenciaOrganizada,
    consultar_fundamentado,
    montar_resposta_verificada,
    organizar_evidencias,
    redigir_rascunho,
    remover_quase_duplicados,
    rotular_trechos,
    resposta_verificada_no_idioma,
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


def evidencia(
    evidencia_id="E1",
    *,
    trecho_ids=("T1",),
    ids_chroma=("livro.pdf-42-0",),
    arquivo="livro.pdf",
    paginas=(42,),
    tipo="definicao",
    conteudo="Um sinal periódico se repete após T.",
    natureza="texto_explicito",
    trecho_ids_contexto=(),
    ids_chroma_contexto=(),
    paginas_contexto=(),
):
    return EvidenciaOrganizada(
        id=evidencia_id,
        tipo=tipo,
        conteudo=conteudo,
        natureza=natureza,
        trecho_ids=trecho_ids,
        ids_chroma=ids_chroma,
        arquivo=arquivo,
        paginas=paginas,
        trecho_ids_contexto=trecho_ids_contexto,
        ids_chroma_contexto=ids_chroma_contexto,
        paginas_contexto=paginas_contexto,
    )


def resposta_auditor(
    *,
    afirmacao_id="A1",
    texto="Um sinal periódico se repete após T.",
    classificacao="sustentada",
    evidencia_ids=("E1",),
):
    return resposta_json(
        {
            "afirmacoes": [
                {
                    "id": afirmacao_id,
                    "texto_final": texto,
                    "classificacao": classificacao,
                    "evidencia_ids": list(evidencia_ids),
                    "justificativa": "A evidência contém a afirmação.",
                }
            ]
        }
    )


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


def test_ids_tn_apontam_para_ids_reais_e_metadados_dos_trechos():
    trechos = [
        trecho("definição", pagina=42, indice=3),
        trecho("continuação", pagina=43, indice=0),
    ]

    rotulados = rotular_trechos(trechos)

    assert [(item.rotulo, item.id_chroma) for item in rotulados] == [
        ("T1", "livro.pdf-42-3"),
        ("T2", "livro.pdf-43-0"),
    ]
    assert (rotulados[0].arquivo, rotulados[0].pagina, rotulados[0].indice) == (
        "livro.pdf",
        42,
        3,
    )


def test_validacao_de_idioma_ignora_nome_ingles_do_pdf_nas_citacoes():
    resposta = (
        "x(t) = x(t + T) "
        "[Signals_and_Systems.pdf, página do PDF 42]\n\n"
        "Fontes\n- [Signals_and_Systems.pdf, página do PDF 42]"
    )

    assert resposta_verificada_no_idioma(resposta, "Português") is True


def test_dois_trechos_na_mesma_pagina_recebem_rotulos_distintos():
    rotulados = rotular_trechos(
        [
            trecho("primeiro", pagina=42, indice=0),
            trecho("segundo", pagina=42, indice=1),
        ]
    )

    assert [item.rotulo for item in rotulados] == ["T1", "T2"]
    assert [item.id_chroma for item in rotulados] == [
        "livro.pdf-42-0",
        "livro.pdf-42-1",
    ]


def test_evidencia_valida_usa_t1_e_recebe_e1_do_programa():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {
                    "id": "E99",
                    "tipo": "definicao",
                    "conteudo": "Sinal se repete.",
                    "trecho_ids": ["T1"],
                }
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente, "Pergunta", [trecho("Periodic signal", pagina=42)]
    )

    assert suficiente
    assert evidencias[0].id == "E1"
    assert evidencias[0].trecho_ids == ("T1",)
    assert evidencias[0].ids_chroma == ("livro.pdf-42-0",)


def test_evidencia_sem_id_de_trecho_e_descartada():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {"tipo": "fato", "conteudo": "Sem vínculo.", "trecho_ids": []}
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente,
        "Pergunta",
        [trecho("texto")],
        diagnostico=diagnostico,
    )

    assert evidencias == []
    assert suficiente is False
    assert diagnostico.evidencias_sem_trecho_rejeitadas == 1


def test_evidencia_pode_usar_varios_trechos_do_mesmo_pdf():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {
                    "tipo": "condicao",
                    "conteudo": "A definição continua na página seguinte.",
                    "trecho_ids": ["T1", "T2"],
                }
            ],
        }
    )
    trechos = [
        trecho("início", pagina=42, indice=0),
        trecho("continuação", pagina=43, indice=0),
    ]

    evidencias, suficiente, _ = organizar_evidencias(cliente, "Pergunta", trechos)

    assert suficiente
    assert evidencias[0].trecho_ids == ("T1", "T2")
    assert evidencias[0].paginas == (42, 43)


def test_evidencia_separa_suporte_de_contexto_e_cita_somente_suporte():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {
                    "tipo": "fato",
                    "conteudo": "O resultado está no primeiro trecho.",
                    "trecho_ids_suporte": ["T1"],
                    "trecho_ids_contexto": ["T2"],
                }
            ],
        }
    )
    trechos = [
        trecho("resultado", pagina=42),
        trecho("explicação contextual", pagina=43),
    ]

    evidencias, suficiente, _ = organizar_evidencias(cliente, "Pergunta", trechos)
    afirmacao = AfirmacaoVerificada(
        texto_original="O resultado está no primeiro trecho.",
        texto_final="O resultado está no primeiro trecho.",
        classificacao="sustentada",
        paginas=(42,),
        natureza="texto_explicito",
        secao="resposta_direta",
        evidencia_ids=("E1",),
    )
    resposta, insuficiente = montar_resposta_verificada(
        "livro.pdf", [afirmacao], "Curto", evidencias=evidencias
    )

    assert suficiente is True
    assert insuficiente is False
    assert evidencias[0].trecho_ids_suporte == ("T1",)
    assert evidencias[0].trecho_ids_contexto == ("T2",)
    assert evidencias[0].paginas == (42,)
    assert evidencias[0].paginas_contexto == (43,)
    assert "página do PDF 42" in resposta
    assert "página do PDF 43" not in resposta


def test_afirmacao_que_depende_de_duas_paginas_preserva_ambas_as_citacoes():
    evidencias = [
        evidencia(
            trecho_ids=("T1", "T2"),
            ids_chroma=("livro.pdf-42-0", "livro.pdf-43-0"),
            paginas=(42, 43),
            conteudo="A condição começa em 42 e termina em 43.",
        )
    ]
    afirmacao = AfirmacaoVerificada(
        texto_original="A condição ocupa duas páginas.",
        texto_final="A condição ocupa duas páginas.",
        classificacao="sustentada",
        paginas=(42, 43),
        natureza="texto_explicito",
        secao="resposta_direta",
        evidencia_ids=("E1",),
    )

    resposta, insuficiente = montar_resposta_verificada(
        "livro.pdf", [afirmacao], "Curto", evidencias=evidencias
    )

    assert insuficiente is False
    assert "página do PDF 42" in resposta
    assert "página do PDF 43" in resposta


def test_evidencia_rejeita_id_de_suporte_nao_recuperado():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {
                    "tipo": "fato",
                    "conteudo": "Não recuperado.",
                    "trecho_ids_suporte": ["T99"],
                    "trecho_ids_contexto": ["T1"],
                }
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente,
        "Pergunta",
        [trecho("contexto", pagina=42)],
        diagnostico=diagnostico,
    )

    assert evidencias == []
    assert suficiente is False
    assert diagnostico.trecho_ids_invalidos_rejeitados == ["T99"]


def test_evidencia_que_mistura_pdfs_e_rejeitada():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {
                    "tipo": "fato",
                    "conteudo": "Mistura indevida.",
                    "trecho_ids": ["T1", "T2"],
                }
            ],
        }
    )
    trechos = [
        trecho("a", arquivo="a.pdf", pagina=1),
        trecho("b", arquivo="b.pdf", pagina=2),
    ]

    evidencias, suficiente, _ = organizar_evidencias(
        cliente, "Pergunta", trechos, diagnostico=diagnostico
    )

    assert evidencias == []
    assert suficiente is False
    assert diagnostico.tentativas_mistura_arquivos == 1


def test_ids_e1_e2_sao_criados_depois_da_validacao_e_deduplicacao():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {"tipo": "fato", "conteudo": "Primeiro conceito.", "trecho_ids": ["T1"]},
                {"tipo": "fato", "conteudo": "Primeiro conceito.", "trecho_ids": ["T1"]},
                {"tipo": "formula", "conteudo": "x(t)=x(t+T).", "trecho_ids": ["T2"]},
            ],
        }
    )
    trechos = [trecho("conceito", pagina=42), trecho("fórmula", pagina=43)]

    evidencias, suficiente, _ = organizar_evidencias(cliente, "Pergunta", trechos)

    assert suficiente
    assert [item.id for item in evidencias] == ["E1", "E2"]


def test_organizador_rejeita_trecho_inexistente_e_ignora_pagina_inventada():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "informacao_faltante": "",
            "evidencias": [
                {"tipo": "definicao", "conteudo": "Sinal se repete.", "trecho_ids": ["T1"], "pagina": 999, "natureza": "texto_explicito"},
                {"tipo": "fato", "conteudo": "Inventado.", "trecho_ids": ["T99"], "natureza": "texto_explicito"},
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente,
        "O que é periódico?",
        [trecho("Periodic signal", pagina=42)],
        diagnostico=diagnostico,
    )

    assert suficiente
    assert [item.pagina for item in evidencias] == [42]
    assert evidencias[0].trecho_ids == ("T1",)
    assert diagnostico.trecho_ids_invalidos_rejeitados == ["T99"]


def test_organizador_assume_texto_explicito_quando_natureza_omitida():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "suficiente": True,
            "evidencias": [
                {"tipo": "fato", "conteudo": "Sinal se repete.", "trecho_ids": ["T1"]}
            ],
        }
    )

    evidencias, suficiente, _ = organizar_evidencias(
        cliente, "Pergunta", [trecho("Periodic signal", pagina=42)]
    )

    assert suficiente
    assert evidencias[0].natureza == "texto_explicito"


def test_afirmacao_com_evidencia_valida_preserva_vinculo_e_paginas_derivadas():
    cliente = Mock()
    cliente.chat.return_value = resposta_auditor()
    trechos = [trecho("Um sinal periódico se repete após T.", pagina=42)]
    evidencias = [evidencia()]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=evidencias,
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "sustentada"
    assert verificadas[0].evidencia_ids == ("E1",)
    assert verificadas[0].paginas == (42,)
    assert verificadas[0].fontes == (("livro.pdf", 42),)


def test_auditor_recebe_somente_trechos_de_suporte_da_afirmacao():
    cliente = Mock()
    cliente.chat.return_value = resposta_auditor()
    trechos = [
        trecho("suporte explícito", pagina=42),
        trecho("contexto amplo", pagina=43),
    ]
    evidencias = [
        evidencia(
            trecho_ids_contexto=("T2",),
            ids_chroma_contexto=("livro.pdf-43-0",),
            paginas_contexto=(43,),
        )
    ]

    verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=evidencias,
        trechos_rotulados=rotular_trechos(trechos),
    )

    prompt = cliente.chat.call_args.kwargs["messages"][1]["content"]
    assert "[T1]" in prompt
    assert "[T2]" not in prompt
    assert "contexto amplo" not in prompt


def test_afirmacao_com_evidencia_inexistente_e_rejeitada_sem_chamar_auditor():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    trechos = [trecho("texto", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Afirmação inventada.", "evidencia_ids": ["E99"]}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
        diagnostico=diagnostico,
    )

    cliente.chat.assert_not_called()
    assert verificadas[0].classificacao == "não sustentada"
    assert verificadas[0].ids_evidencia_invalidos == ("E99",)
    assert diagnostico.evidencia_ids_invalidos_rejeitados == ["E99"]


def test_afirmacao_sem_evidencia_e_rejeitada_sem_preenchimento_automatico():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    trechos = [trecho("texto", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Afirmação sem vínculo."}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
        diagnostico=diagnostico,
    )

    assert verificadas[0].evidencia_ids == ()
    assert verificadas[0].classificacao == "não sustentada"
    assert diagnostico.afirmacoes_sem_evidencia == 1


def test_redacao_nao_preenche_evidencia_omitida_pelo_modelo():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_json(
        {
            "afirmacoes": [
                {
                    "texto": "Afirmação sem vínculo.",
                    "secao": "explicacao",
                    "paginas": [42],
                }
            ]
        }
    )

    rascunho = redigir_rascunho(
        cliente,
        "Pergunta",
        [evidencia()],
        "Português",
        "Curto",
        diagnostico=diagnostico,
    )

    assert rascunho[0]["evidencia_ids"] == []
    assert diagnostico.afirmacoes_sem_evidencia == 1


def test_redacao_aceita_chave_afirmacoes_localizada_sem_alterar_ids():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "afirmações": [
                {
                    "texto": "Um sinal periódico se repete após T.",
                    "secao": "resposta_direta",
                    "evidencia_ids": ["E1"],
                    "natureza": "texto_explicito",
                }
            ]
        }
    )

    rascunho = redigir_rascunho(
        cliente,
        "Pergunta",
        [evidencia()],
        "Português",
        "Curto",
    )

    assert rascunho[0]["evidencia_ids"] == ["E1"]
    assert rascunho[0]["texto"] == "Um sinal periódico se repete após T."


def test_deducao_simples_pode_usar_duas_evidencias():
    cliente = Mock()
    cliente.chat.return_value = resposta_auditor(
        texto="Logo, o menor T positivo é o período fundamental.",
        evidencia_ids=("E1", "E2"),
    )
    trechos = [
        trecho("O sinal se repete após T.", pagina=42),
        trecho("O menor T positivo é fundamental.", pagina=43),
    ]
    evidencias = [
        evidencia("E1"),
        evidencia(
            "E2",
            trecho_ids=("T2",),
            ids_chroma=("livro.pdf-43-0",),
            paginas=(43,),
            conteudo="O menor T positivo é o período fundamental.",
        ),
    ]

    verificadas = verificar_afirmacoes(
        cliente,
        [{
            "texto": "Logo, o menor T positivo é o período fundamental.",
            "evidencia_ids": ["E1", "E2"],
            "natureza": "deducao_simples",
        }],
        trechos,
        "Português",
        evidencias=evidencias,
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].evidencia_ids == ("E1", "E2")
    assert verificadas[0].paginas == (42, 43)
    assert verificadas[0].natureza == "deducao_simples"


def test_formula_permanece_ligada_ao_trecho_que_a_contem():
    cliente = Mock()
    cliente.chat.return_value = resposta_auditor(
        texto="x(t) = x(t + T)", evidencia_ids=("E1",)
    )
    trechos = [trecho("x(t) = x(t + T)", pagina=43, indice=2)]
    evidencias = [
        evidencia(
            tipo="formula",
            conteudo="x(t) = x(t + T)",
            ids_chroma=("livro.pdf-43-2",),
            paginas=(43,),
        )
    ]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "x(t) = x(t + T)", "evidencia_ids": ["E1"], "secao": "formula"}],
        trechos,
        "Português",
        evidencias=evidencias,
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "sustentada"
    assert verificadas[0].paginas == (43,)
    assert verificadas[0].evidencia_ids == ("E1",)


def test_citacao_e_derivada_da_evidencia_e_pagina_do_modelo_e_ignorada():
    afirmacao = AfirmacaoVerificada(
        texto_original="Definição.",
        texto_final="Um sinal periódico se repete após T.",
        classificacao="sustentada",
        paginas=(999,),
        natureza="texto_explicito",
        secao="resposta_direta",
        evidencia_ids=("E1",),
    )

    resposta, insuficiente = montar_resposta_verificada(
        "livro.pdf",
        [afirmacao],
        "Curto",
        evidencias=[evidencia()],
    )

    assert not insuficiente
    assert "página do PDF 42" in resposta
    assert "999" not in resposta


def test_id_inexistente_adicionado_pelo_auditor_e_rejeitado():
    cliente = Mock()
    diagnostico = DiagnosticoEstrutural()
    cliente.chat.return_value = resposta_auditor(evidencia_ids=("E1", "E99"))
    trechos = [trecho("Um sinal periódico se repete após T.", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
        diagnostico=diagnostico,
    )

    assert verificadas[0].evidencia_ids == ("E1",)
    assert diagnostico.ids_adicionados_auditor_rejeitados == ["E99"]


def test_auditor_pode_omitir_envelope_sem_alterar_ids_da_afirmacao():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "id": "A1",
            "texto_final": "Um sinal periódico se repete após T.",
            "classificacao": "sustentada",
            "evidencia_ids": ["E1"],
            "justificativa": "A evidência declara explicitamente a repetição.",
        }
    )
    trechos = [trecho("Um sinal periódico se repete após T.", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "sustentada"
    assert verificadas[0].evidencia_ids == ("E1",)
    assert verificadas[0].paginas == (42,)


def test_auditor_sustentado_pode_reter_texto_original_se_omitir_texto_final():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "id": "A1",
            "classificacao": "sustentada",
            "evidencia_ids": ["E1"],
            "justificativa": "A evidência sustenta integralmente a frase.",
        }
    )
    trechos = [trecho("Um sinal periódico se repete após T.", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "sustentada"
    assert verificadas[0].texto_final == "Um sinal periódico se repete após T."


def test_auditor_que_remove_id_da_afirmacao_e_rejeitado():
    cliente = Mock()
    cliente.chat.return_value = resposta_json(
        {
            "id": "A1",
            "texto_final": "Conclusão baseada em duas evidências.",
            "classificacao": "sustentada",
            "evidencia_ids": ["E1"],
            "justificativa": "Aprovada.",
        }
    )
    trechos = [
        trecho("Primeira premissa.", pagina=42),
        trecho("Segunda premissa.", pagina=43),
    ]
    evidencias = [
        evidencia("E1"),
        evidencia(
            "E2",
            trecho_ids=("T2",),
            ids_chroma=("livro.pdf-43-0",),
            paginas=(43,),
            conteudo="Segunda premissa.",
        ),
    ]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Conclusão baseada em duas evidências.", "evidencia_ids": ["E1", "E2"]}],
        trechos,
        "Português",
        evidencias=evidencias,
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "não sustentada"
    assert verificadas[0].texto_final == ""
    assert verificadas[0].evidencia_ids == ("E1", "E2")


def test_json_invalido_do_auditor_estruturado_reprova_sem_perder_ids():
    cliente = Mock()
    cliente.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="JSON inválido")
    )
    trechos = [trecho("Um sinal periódico se repete após T.", pagina=42)]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Um sinal periódico se repete após T.", "evidencia_ids": ["E1"]}],
        trechos,
        "Português",
        evidencias=[evidencia()],
        trechos_rotulados=rotular_trechos(trechos),
    )

    assert verificadas[0].classificacao == "não sustentada"
    assert verificadas[0].evidencia_ids == ("E1",)
    assert verificadas[0].paginas == (42,)


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


def test_json_invalido_no_fallback_auxiliar_reprova_sem_abortar():
    cliente = Mock()
    invalida = SimpleNamespace(message=SimpleNamespace(content="não é JSON"))
    cliente.chat.side_effect = [
        resposta_json({"afirmacoes": []}),
        invalida,
        invalida,
    ]

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Afirmação sem retorno do auditor."}],
        [trecho("texto autorizado", pagina=42)],
        "Português",
    )

    assert verificadas[0].classificacao == "não sustentada"
    assert verificadas[0].texto_final == ""


def test_json_invalido_no_lote_auxiliar_reprova_sem_abortar():
    cliente = Mock()
    cliente.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content="saída inválida")
    )

    verificadas = verificar_afirmacoes(
        cliente,
        [{"texto": "Afirmação que não pôde ser auditada."}],
        [trecho("texto autorizado", pagina=42)],
        "Português",
    )

    assert verificadas[0].classificacao == "não sustentada"
    assert verificadas[0].texto_final == ""


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
        "livro.pdf",
        [],
        "Curto",
        "falta a especificação do motor",
        evidencias=[],
    )

    assert insuficiente
    assert "Não encontrei evidência suficiente" in resposta
    assert "falta a especificação do motor" in resposta
    assert "Fontes" not in resposta
    assert "E1" not in resposta


def test_consulta_fundamentada_completa_preserva_encadeamento_de_ids(monkeypatch):
    candidatos = [
        trecho(
            "Um sinal periódico se repete após um período T.",
            pagina=42,
            score=1.0,
            indice=0,
        ),
        trecho(
            "O período T caracteriza a repetição do sinal periódico.",
            pagina=43,
            score=0.85,
            indice=0,
        ),
        trecho(
            "O menor período positivo é chamado período fundamental.",
            pagina=43,
            score=0.75,
            indice=1,
        ),
    ]
    cliente = Mock()
    cliente.chat.side_effect = [
        resposta_json(
            {
                "suficiente": True,
                "informacao_faltante": "",
                "evidencias": [
                    {
                        "tipo": "definicao",
                        "conteudo": "Um sinal periódico se repete após T.",
                        "trecho_ids": ["T1"],
                        "natureza": "texto_explicito",
                    }
                ],
            }
        ),
        resposta_json(
            {
                "afirmacoes": [
                    {
                        "texto": "Um sinal periódico se repete após T.",
                        "secao": "resposta_direta",
                        "evidencia_ids": ["E1"],
                        "natureza": "texto_explicito",
                    }
                ]
            }
        ),
        resposta_auditor(),
    ]

    monkeypatch.setattr(
        "src.grounded.manifesto_compativel",
        lambda *args, **kwargs: SimpleNamespace(dimensao=2),
    )
    monkeypatch.setattr(
        "src.grounded.verificar_ollama_e_modelos", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.grounded.gerar_embedding_pergunta", lambda *args, **kwargs: [0.1, 0.2]
    )
    monkeypatch.setattr(
        "src.grounded.recuperar_trechos",
        lambda *args, **kwargs: (candidatos, candidatos),
    )

    resultado = consultar_fundamentado(
        "O que caracteriza um sinal periódico?",
        arquivo="livro.pdf",
        incluir_vizinhas=False,
        cliente=cliente,
        colecao=object(),
    )

    assert not resultado.insuficiente
    assert resultado.trechos_rotulados[0].rotulo == "T1"
    assert resultado.evidencias[0].id == "E1"
    assert resultado.evidencias[0].trecho_ids == ("T1",)
    assert resultado.afirmacoes[0].evidencia_ids == ("E1",)
    assert "[livro.pdf, página do PDF 42]" in resultado.resposta
