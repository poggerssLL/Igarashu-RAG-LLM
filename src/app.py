"""Interface Streamlit local para o RAG organizado por matérias."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from src.chat import (  # noqa: E402
    ErroConsulta,
    IDIOMAS_RESPOSTA,
    abrir_colecao,
    consultar,
    listar_arquivos_indexados,
    remover_secao_fontes,
)
from src.config import (  # noqa: E402
    MINIMO_CANDIDATOS,
    MODELO_EMBEDDINGS,
    MODELOS_EMBEDDINGS_SUPORTADOS,
    PASTA_DOCUMENTOS,
)
from src.disciplinas import (  # noqa: E402
    ErroDisciplina,
    Materia,
    SEM_DISCIPLINA,
    TODAS_DISCIPLINAS,
    carregar_materias,
    criar_materia,
    disciplinas_com_trechos,
    editar_materia,
    estatisticas_materias,
    excluir_materia,
    garantir_materias_padrao,
    mover_pdf,
    resumo_biblioteca,
)
from src.evaluate import executar_avaliacao, taxa_acerto_recuperacao  # noqa: E402
from src.ingest import ErroIngestao, executar_ingestao, encontrar_pdfs  # noqa: E402
from src.index_manifest import carregar_manifesto  # noqa: E402
from src.grounded import (  # noqa: E402
    MODO_AUTOMATICO,
    NIVEIS_DETALHE,
    consultar_fundamentado,
)
from src.ui_utils import salvar_pdf  # noqa: E402


st.set_page_config(page_title="Engenharia em Foco", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 1.2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background: #171d2b; border: 1px solid #2b354a;
        border-radius: 12px; padding: .7rem 1rem;}
      [data-testid="stChatMessage"] {border: 1px solid #2b354a; border-radius: 14px;
        padding: .35rem .65rem; margin-bottom: .7rem;}
      .study-header {background: linear-gradient(120deg,#172033,#202943); border:1px solid #33405b;
        border-radius:16px; padding:1rem 1.25rem; margin-bottom:1rem;}
      .study-header h1 {font-size:1.55rem; margin:0 0 .25rem 0;}
      .study-header p {margin:0; color:#bdc7dc;}
      .scope {color:#9fb3d9; font-size:.88rem; margin-bottom:.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def colecao_opcional() -> object | None:
    try:
        return abrir_colecao()
    except ErroConsulta:
        return None


def materias_atuais() -> list[Materia]:
    try:
        return garantir_materias_padrao()
    except ErroDisciplina as erro:
        st.error(str(erro))
        return []


def marcar_reindexacao(mensagem: str) -> None:
    st.session_state["reindexacao_pendente"] = mensagem


def modelo_do_indice() -> str:
    manifesto = carregar_manifesto()
    return manifesto.modelo_embeddings if manifesto else MODELO_EMBEDDINGS


def cabecalho(materias: list[Materia], colecao: object | None) -> dict[str, int]:
    resumo = resumo_biblioteca(materias, colecao=colecao)
    estado = "Biblioteca pronta" if resumo["trechos"] else "Biblioteca ainda não indexada"
    st.markdown(
        f"""
        <div class="study-header">
          <h1>Engenharia em Foco</h1>
          <p>Assistente local de estudos com RAG · {estado}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if aviso := st.session_state.get("reindexacao_pendente"):
        st.warning(f"{aviso} Reindexe toda a biblioteca para sincronizar a busca.")
    return resumo


def fontes_detalhadas(trechos: list) -> list:
    unicos = []
    vistos = set()
    for trecho in trechos:
        chave = (trecho.arquivo, trecho.pagina)
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(trecho)
    return unicos


def exibir_fontes(trechos: list) -> None:
    fontes = fontes_detalhadas(trechos)
    with st.expander(f"Fontes consultadas ({len(fontes)})", expanded=False):
        for trecho in fontes:
            relevancia = trecho.relevancia or 0.0
            st.markdown(
                f"- **{trecho.arquivo}**  \n"
                f"  Matéria: `{trecho.disciplina}` · página do PDF **{trecho.pagina}** · "
                f"relevância `{relevancia:.3f}`"
            )


def exibir_contexto(trechos: list) -> None:
    with st.expander("Contexto recuperado", expanded=False):
        for numero, trecho in enumerate(trechos, start=1):
            st.markdown(
                f"**{numero}. [{trecho.arquivo}, página do PDF {trecho.pagina}]** "
                f"— {trecho.disciplina} — relevância `{trecho.relevancia or 0.0:.3f}`"
            )
            st.write(trecho.texto)


def exibir_evidencias(evidencias: list) -> None:
    with st.expander("Evidências organizadas e auditoria", expanded=False):
        if not evidencias:
            st.caption("Nenhuma evidência foi considerada suficiente.")
            return
        for evidencia in evidencias:
            natureza = (
                "dedução simples"
                if evidencia.natureza == "deducao_simples"
                else "texto explícito"
            )
            paginas = ", ".join(str(pagina) for pagina in evidencia.paginas)
            trechos = ", ".join(evidencia.trecho_ids)
            st.markdown(
                f"- **{evidencia.id} · {evidencia.tipo}** · `{natureza}`  \n"
                f"  Arquivo: **{evidencia.arquivo}** · página do PDF **{paginas}** · "
                f"trechos `{trechos}`  \n  {evidencia.conteudo}"
            )


def opcoes_disciplinas_busca(materias: list[Materia], colecao: object | None) -> list[str]:
    indexadas = set(disciplinas_com_trechos(colecao))
    cadastradas = [materia.nome for materia in materias if materia.nome in indexadas]
    return [TODAS_DISCIPLINAS, SEM_DISCIPLINA, *cadastradas]


def pagina_conversar(
    materias: list[Materia], colecao: object | None, resumo: dict[str, int]
) -> None:
    with st.sidebar:
        st.subheader("Escopo e busca")
        escopo = st.selectbox(
            "Disciplina", opcoes_disciplinas_busca(materias, colecao), index=0
        )
        modo = st.selectbox(
            "Modo de resposta",
            ("Fundamentado", "Compatibilidade"),
            index=0,
            help="Fundamentado usa um único PDF e audita as afirmações. Compatibilidade preserva o fluxo anterior.",
        )
        disciplina_filtro = None if escopo == TODAS_DISCIPLINAS else escopo
        arquivos = (
            listar_arquivos_indexados(colecao, disciplina_filtro)
            if colecao is not None
            else []
        )
        arquivo_escolhido = st.selectbox(
            "Fonte",
            (MODO_AUTOMATICO, *arquivos),
            index=0,
            help="Automático escolhe o PDF com as evidências mais fortes e refaz a busca dentro dele.",
        )
        with st.expander("Controles de recuperação", expanded=True):
            top_k = st.number_input("Trechos (top-k)", 1, 20, 4, 1)
            candidatos = st.slider(
                "Candidatos iniciais", MINIMO_CANDIDATOS, 100, MINIMO_CANDIDATOS, 5,
                help="A busca reúne ao menos 20 candidatos antes de escolher o contexto.",
            )
            min_relevancia = st.slider(
                "Relevância mínima", 0.0, 1.0, 0.0, 0.05,
                help="Use 0 para desativar. Para começar, experimente 0,55.",
            )
            busca_hibrida = st.checkbox("Combinar vetores e palavras-chave", value=True)
            incluir_vizinhas = st.checkbox("Incluir páginas vizinhas", value=False)
            diversificar = st.checkbox(
                "Diversificar arquivos quando forem semelhantes", value=True
            )
            idioma = st.selectbox("Idioma da resposta", IDIOMAS_RESPOSTA, index=0)
            nivel_detalhe = st.selectbox(
                "Nível de detalhe", NIVEIS_DETALHE, index=1
            )
            st.caption(f"Embeddings do índice: `{modelo_do_indice()}`")
            mostrar = st.checkbox("Mostrar contexto recuperado")
            mostrar_evidencias = st.checkbox(
                "Mostrar evidências utilizadas", value=False
            )

    st.subheader("Converse com seus materiais")
    historico = st.session_state.setdefault("historico", [])
    pergunta_sugerida = None
    if not historico:
        st.info(
            "Bem-vindo! Faça perguntas sobre conceitos, fórmulas e exemplos presentes "
            "nos PDFs indexados."
        )
        colunas = st.columns(3)
        colunas[0].metric("Matérias", resumo["materias"])
        colunas[1].metric("PDFs", resumo["pdfs"])
        colunas[2].metric("Trechos indexados", resumo["trechos"])
        st.markdown("**Sugestões para começar**")
        sugestoes = (
            "Resuma os principais conceitos do material.",
            "Quais fórmulas importantes aparecem nas aulas?",
            "Explique um conceito central com as fontes.",
        )
        botoes = st.columns(3)
        for indice, sugestao in enumerate(sugestoes):
            if botoes[indice].button(sugestao, width="stretch"):
                pergunta_sugerida = sugestao

    for mensagem in historico:
        with st.chat_message("user"):
            st.write(mensagem["pergunta"])
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="scope">Escopo pesquisado: {mensagem["escopo"]}</div>',
                unsafe_allow_html=True,
            )
            if mensagem.get("documento"):
                st.info(f"Documento escolhido: {mensagem['documento']}")
                st.caption(mensagem.get("motivo_documento", ""))
            st.markdown(mensagem["resposta"])
            exibir_fontes(mensagem["trechos"])
            if mensagem["mostrar_contexto"]:
                exibir_contexto(mensagem["trechos"])
            if mensagem.get("mostrar_evidencias"):
                exibir_evidencias(mensagem.get("evidencias", []))

    pergunta = st.chat_input("Pergunte algo sobre os materiais") or pergunta_sugerida
    if pergunta:
        disciplina = None if escopo == TODAS_DISCIPLINAS else escopo
        with st.chat_message("user"):
            st.write(pergunta)
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="scope">Escopo pesquisado: {escopo}</div>',
                unsafe_allow_html=True,
            )
            try:
                with st.spinner("Consultando a biblioteca local..."):
                    if modo == "Fundamentado":
                        resultado = consultar_fundamentado(
                            pergunta,
                            disciplina=disciplina,
                            arquivo=(
                                None
                                if arquivo_escolhido == MODO_AUTOMATICO
                                else arquivo_escolhido
                            ),
                            idioma=idioma,
                            nivel_detalhe=nivel_detalhe,
                            candidatos=int(candidatos),
                            incluir_vizinhas=incluir_vizinhas,
                            modelo_embeddings=modelo_do_indice(),
                        )
                        trechos = list(resultado.trechos)
                        resposta_com_fontes = resultado.resposta
                        documento = resultado.documento
                        motivo_documento = resultado.motivo_documento
                        evidencias = list(resultado.evidencias)
                    else:
                        trechos, resposta_com_fontes = consultar(
                            pergunta,
                            int(top_k),
                            min_relevancia=float(min_relevancia),
                            disciplina=disciplina,
                            candidatos=int(candidatos),
                            busca_hibrida=busca_hibrida,
                            incluir_vizinhas=incluir_vizinhas,
                            diversificar_arquivos=diversificar,
                            idioma_resposta=idioma,
                            modelo_embeddings=modelo_do_indice(),
                        )
                        documento = ""
                        motivo_documento = ""
                        evidencias = []
                resposta = remover_secao_fontes(resposta_com_fontes)
                if documento:
                    st.info(f"Documento escolhido: {documento}")
                    st.caption(motivo_documento)
                st.markdown(resposta)
                exibir_fontes(trechos)
                if mostrar:
                    exibir_contexto(trechos)
                if mostrar_evidencias:
                    exibir_evidencias(evidencias)
                historico.append(
                    {
                        "pergunta": pergunta,
                        "resposta": resposta,
                        "trechos": trechos,
                        "escopo": escopo,
                        "mostrar_contexto": mostrar,
                        "idioma": idioma,
                        "modo": modo,
                        "documento": documento,
                        "motivo_documento": motivo_documento,
                        "evidencias": evidencias,
                        "mostrar_evidencias": mostrar_evidencias,
                    }
                )
            except ErroConsulta as erro:
                st.error(str(erro))
                st.info(
                    "Confirme que o Ollama está aberto, os modelos estão instalados e "
                    "a biblioteca foi reindexada após organizar as matérias."
                )

    with st.sidebar:
        st.divider()
        if st.button("Limpar histórico da sessão", width="stretch"):
            st.session_state["historico"] = []
            st.rerun()


def pasta_e_pdfs_da_selecao(selecao: str, materias: list[Materia]) -> tuple[Path, list[Path]]:
    if selecao == SEM_DISCIPLINA:
        pasta = PASTA_DOCUMENTOS
        pdfs = sorted(
            arquivo for arquivo in pasta.glob("*.pdf") if arquivo.is_file()
        )
    else:
        materia = next(materia for materia in materias if materia.nome == selecao)
        pasta = PASTA_DOCUMENTOS / materia.pasta
        pasta.mkdir(parents=True, exist_ok=True)
        pdfs = encontrar_pdfs(pasta)
    return pasta, pdfs


def mostrar_relatorio_indexacao(relatorio) -> None:
    st.success("Indexação concluída.")
    colunas = st.columns(4)
    colunas[0].metric("PDFs", relatorio.pdfs_encontrados)
    colunas[1].metric("Páginas", relatorio.paginas_lidas)
    colunas[2].metric("Trechos", relatorio.trechos_indexados)
    colunas[3].metric("Ignorados", relatorio.trechos_ignorados)
    st.caption(
        f"Modelo: {relatorio.modelo_embeddings} · dimensão: {relatorio.dimensao_embeddings}"
    )


def pagina_documentos(materias: list[Materia]) -> None:
    st.subheader("Documentos da biblioteca")
    opcoes = [SEM_DISCIPLINA, *[materia.nome for materia in materias]]
    selecao = st.selectbox("Matéria para receber e listar PDFs", opcoes)
    pasta, pdfs = pasta_e_pdfs_da_selecao(selecao, materias)
    st.caption(f"Destino: Documentos/{'' if selecao == SEM_DISCIPLINA else selecao + '/'}")

    if pdfs:
        st.markdown(f"**{len(pdfs)} PDF(s) em {selecao}**")
        for pdf in pdfs:
            st.markdown(f"- `{pdf.relative_to(PASTA_DOCUMENTOS).as_posix()}`")
    else:
        st.info("Nenhum PDF nesta matéria. Selecione arquivos abaixo para começar.")

    enviados = st.file_uploader(
        "Enviar PDFs para a matéria selecionada",
        type=["pdf"],
        accept_multiple_files=True,
    )
    sobrescrever = st.checkbox(
        "Confirmo a substituição de arquivos com o mesmo nome", value=False
    )
    if st.button("Salvar PDFs selecionados", disabled=not enviados):
        salvos = 0
        for enviado in enviados or []:
            try:
                destino = salvar_pdf(
                    enviado.name,
                    bytes(enviado.getbuffer()),
                    pasta,
                    sobrescrever=sobrescrever,
                )
                st.success(f"Salvo em {destino.relative_to(PASTA_DOCUMENTOS)}")
                salvos += 1
            except (ValueError, FileExistsError, OSError) as erro:
                st.warning(str(erro))
        if salvos:
            marcar_reindexacao("Novos documentos foram adicionados.")
            st.warning("Indexe ou reindexe a biblioteca antes de pesquisar os novos PDFs.")

    st.divider()
    manifesto = carregar_manifesto()
    atual = manifesto.modelo_embeddings.split(":", 1)[0] if manifesto else MODELO_EMBEDDINGS
    opcoes_modelo = list(MODELOS_EMBEDDINGS_SUPORTADOS)
    indice_modelo = opcoes_modelo.index(atual) if atual in opcoes_modelo else 0
    modelo_escolhido = st.selectbox(
        "Modelo de embeddings para a biblioteca",
        opcoes_modelo,
        index=indice_modelo,
        help="Trocar o modelo exige recriar todo o índice; PDFs não são apagados.",
    )
    if manifesto:
        st.caption(
            f"Índice atual: {manifesto.modelo_embeddings} · {manifesto.dimensao} dimensões · "
            f"trechos de {manifesto.tamanho_trecho} caracteres · {manifesto.data_indexacao}"
        )
    esquerda, direita = st.columns(2)
    if esquerda.button("Indexar documentos", type="primary", width="stretch"):
        try:
            with st.spinner("Gerando embeddings localmente..."):
                relatorio = executar_ingestao(modelo_embeddings=modelo_escolhido)
            mostrar_relatorio_indexacao(relatorio)
        except ErroIngestao as erro:
            st.error(str(erro))

    with direita:
        confirmar = st.checkbox(
            "Confirmo recriar todo o índice vetorial",
            help="PDFs nunca são apagados. Somente a coleção do ChromaDB é recriada.",
        )
        if st.button(
            "Reindexar toda a biblioteca",
            disabled=not confirmar,
            width="stretch",
        ):
            try:
                with st.spinner("Recriando o índice e processando todos os PDFs..."):
                    relatorio = executar_ingestao(
                        recriar_indice=True,
                        confirmar_reindexacao=True,
                        modelo_embeddings=modelo_escolhido,
                    )
                st.session_state.pop("reindexacao_pendente", None)
                mostrar_relatorio_indexacao(relatorio)
            except ErroIngestao as erro:
                st.error(str(erro))

    st.info(
        "A interface não exclui PDFs. A reindexação remove apenas o índice vetorial "
        "anterior e o recria a partir dos arquivos existentes."
    )


def pagina_materias(materias: list[Materia], colecao: object | None) -> None:
    st.subheader("Gerenciamento de matérias")
    estatisticas = estatisticas_materias(materias, colecao=colecao)
    if estatisticas:
        st.dataframe(
            [
                {
                    "Matéria": item.materia.nome,
                    "Descrição": item.materia.descricao or "—",
                    "PDFs": item.pdfs,
                    "Trechos indexados": item.trechos,
                    "Criada em": item.materia.criada_em[:10],
                }
                for item in estatisticas
            ],
            hide_index=True,
            width="stretch",
        )

    with st.expander("Criar nova matéria"):
        with st.form("criar_materia"):
            nome = st.text_input("Nome")
            descricao = st.text_area("Descrição opcional")
            criar = st.form_submit_button("Criar matéria")
        if criar:
            try:
                criada = criar_materia(nome, descricao)
                marcar_reindexacao(f"A matéria '{criada.nome}' foi criada.")
                st.success(f"Pasta criada: Documentos/{criada.pasta}/")
                st.rerun()
            except ErroDisciplina as erro:
                st.error(str(erro))

    if not materias:
        return
    nomes_ids = {materia.nome: materia.id for materia in materias}

    with st.expander("Editar ou renomear matéria"):
        selecionada = st.selectbox("Matéria", list(nomes_ids), key="materia_editar")
        atual = next(materia for materia in materias if materia.nome == selecionada)
        with st.form("editar_materia"):
            novo_nome = st.text_input("Novo nome", value=atual.nome)
            nova_descricao = st.text_area("Descrição", value=atual.descricao)
            confirmar_nome = st.checkbox(
                "Confirmo que a pasta será renomeada e que precisarei reindexar"
            )
            salvar = st.form_submit_button("Salvar alterações")
        if salvar:
            try:
                renomeou = novo_nome.strip() != atual.nome
                atualizada = editar_materia(
                    atual.id,
                    novo_nome,
                    nova_descricao,
                    confirmar_renomeacao=confirmar_nome,
                )
                if renomeou:
                    marcar_reindexacao(
                        f"A matéria foi renomeada para '{atualizada.nome}' e sua pasta foi movida."
                    )
                st.success("Matéria atualizada.")
                st.rerun()
            except ErroDisciplina as erro:
                st.error(str(erro))

    with st.expander("Mover PDF entre matérias"):
        todos_pdfs = encontrar_pdfs(PASTA_DOCUMENTOS)
        if not todos_pdfs:
            st.info("Não há PDFs para mover.")
        else:
            relativos = [pdf.relative_to(PASTA_DOCUMENTOS).as_posix() for pdf in todos_pdfs]
            origem = st.selectbox("PDF", relativos)
            destinos = [SEM_DISCIPLINA, *list(nomes_ids)]
            destino_nome = st.selectbox("Matéria de destino", destinos)
            confirmar_movimento = st.checkbox(
                "Confirmo a movimentação do arquivo e a necessidade de reindexar"
            )
            if st.button("Mover PDF", disabled=not confirmar_movimento):
                try:
                    destino_id = None if destino_nome == SEM_DISCIPLINA else nomes_ids[destino_nome]
                    novo_caminho = mover_pdf(
                        origem, destino_id, confirmar=True
                    )
                    marcar_reindexacao(
                        f"O PDF foi movido para '{novo_caminho.as_posix()}'."
                    )
                    st.success("PDF movido sem alterar seu conteúdo.")
                    st.rerun()
                except ErroDisciplina as erro:
                    st.error(str(erro))

    with st.expander("Excluir matéria vazia"):
        excluir_nome = st.selectbox("Matéria", list(nomes_ids), key="materia_excluir")
        estatistica = next(
            item for item in estatisticas if item.materia.nome == excluir_nome
        )
        if estatistica.pdfs:
            st.warning(
                f"Esta matéria possui {estatistica.pdfs} PDF(s). Mova ou remova os arquivos "
                "manualmente antes de excluí-la."
            )
        confirmar_exclusao = st.checkbox(
            "Confirmo a exclusão da matéria vazia e de sua pasta vazia"
        )
        if st.button(
            "Excluir matéria",
            disabled=bool(estatistica.pdfs) or not confirmar_exclusao,
        ):
            try:
                excluir_materia(nomes_ids[excluir_nome], confirmar=True)
                st.success("Matéria vazia excluída.")
                st.rerun()
            except ErroDisciplina as erro:
                st.error(str(erro))


def pagina_avaliacao(materias: list[Materia], colecao: object | None) -> None:
    st.subheader("Qualidade e confiabilidade")
    aba_recuperacao, aba_geracao = st.tabs(("Recuperação", "Geração fundamentada"))

    with aba_recuperacao:
        opcoes = opcoes_disciplinas_busca(materias, colecao)
        escopo = st.selectbox("Disciplina avaliada", opcoes, key="disciplina_avaliacao")
        with st.expander("Controles da avaliação", expanded=False):
            candidatos = st.slider(
                "Candidatos", MINIMO_CANDIDATOS, 100, MINIMO_CANDIDATOS, 5,
                key="candidatos_avaliacao",
            )
            busca_hibrida = st.checkbox("Busca híbrida", value=True, key="hibrida_avaliacao")
            incluir_vizinhas = st.checkbox("Páginas vizinhas", value=False, key="vizinhas_avaliacao")
            diversificar = st.checkbox("Diversificação condicional", value=True, key="diversifica_avaliacao")
        if st.button("Executar avaliação da recuperação", type="primary"):
            disciplina = None if escopo == TODAS_DISCIPLINAS else escopo
            try:
                with st.spinner("Executando casos com embeddings locais..."):
                    st.session_state["ultima_avaliacao"] = (
                        escopo,
                        executar_avaliacao(
                            disciplina,
                            candidatos=int(candidatos),
                            busca_hibrida=busca_hibrida,
                            incluir_vizinhas=incluir_vizinhas,
                            diversificar_arquivos=diversificar,
                            modelo_embeddings=modelo_do_indice(),
                        ),
                    )
            except ErroConsulta as erro:
                st.error(str(erro))

        dados = st.session_state.get("ultima_avaliacao")
        if not dados:
            st.info("Execute a avaliação para medir a recuperação da biblioteca.")
        else:
            escopo_executado, resultados = dados
            falhas = [resultado for resultado in resultados if not resultado.acertou_pagina]
            termos_ok = sum(not resultado.termos_ausentes for resultado in resultados)
            colunas = st.columns(4)
            colunas[0].metric("Taxa de acerto", f"{taxa_acerto_recuperacao(resultados):.1%}")
            colunas[1].metric("Casos", len(resultados))
            colunas[2].metric("Falhas", len(falhas))
            colunas[3].metric("Termos completos", f"{termos_ok}/{len(resultados)}")
            st.caption(f"Escopo avaliado: {escopo_executado}")
            if falhas:
                for falha in falhas:
                    with st.expander(f"Falha · {falha.pergunta}"):
                        st.write(f"Páginas esperadas: {falha.paginas_esperadas}")
                        st.write(f"Páginas recuperadas: {falha.paginas_retornadas}")
                        st.write("Termos ausentes: " + ", ".join(falha.termos_ausentes))
            else:
                st.success("Todos os casos recuperaram ao menos uma página esperada.")

    with aba_geracao:
        st.caption(
            "Executa sete casos reais e audita a resposta final publicada. A auditoria semântica "
            "usa o mesmo Qwen e é auxiliar, não uma validação independente."
        )
        comparar = st.checkbox(
            "Executar também o modo Compatibilidade e gerar nova linha de base corrigida",
            value=False,
            key="comparar_geracao",
        )
        if st.button("Executar avaliação da geração", type="primary"):
            try:
                from src.generation_eval import (
                    carregar_casos_geracao,
                    executar_avaliacao_geracao,
                    resultado_aprovado,
                    resumo_metricas,
                    salvar_linha_base,
                )

                with st.spinner("Gerando e auditando respostas localmente..."):
                    casos_geracao = carregar_casos_geracao()
                    anteriores = None
                    if comparar:
                        anteriores = executar_avaliacao_geracao(
                            "compatibilidade",
                            casos_geracao,
                            salvar_resultado=False,
                        )
                        salvar_linha_base(anteriores)
                    atuais = executar_avaliacao_geracao(
                        "fundamentado", casos_geracao
                    )
                    st.session_state["ultima_avaliacao_geracao"] = (
                        resumo_metricas(anteriores or []),
                        resumo_metricas(atuais),
                        atuais,
                    )
            except (ErroConsulta, ValueError, OSError) as erro:
                st.error(str(erro))

        dados_geracao = st.session_state.get("ultima_avaliacao_geracao")
        if not dados_geracao:
            st.info("Execute a avaliação para medir a confiabilidade da resposta final.")
        else:
            base, metricas, resultados = dados_geracao
            deterministicas = metricas["metricas_deterministicas"]
            auxiliares = metricas["metricas_auxiliares_qwen"]
            rastreabilidade = metricas[
                "metricas_rastreabilidade_deterministicas"
            ]

            def exibir_contagem(metrica: dict) -> str:
                aplicaveis = int(metrica.get("aplicaveis") or 0)
                if not aplicaveis:
                    return "N/A"
                acertos = int(metrica.get("acertos") or 0)
                return f"{acertos}/{aplicaveis} · {acertos / aplicaveis:.0%}"

            colunas = st.columns(3)
            colunas[0].metric(
                "Página correta", exibir_contagem(deterministicas["pagina_correta"])
            )
            colunas[1].metric(
                "Fonte correta", exibir_contagem(deterministicas["fonte_correta"])
            )
            colunas[2].metric(
                "Citação recuperada",
                exibir_contagem(deterministicas["citacao_recuperada"]),
            )
            colunas = st.columns(3)
            colunas[0].metric(
                "Conceitos", exibir_contagem(deterministicas["conceitos_presentes"])
            )
            colunas[1].metric(
                "Recusa correta", exibir_contagem(deterministicas["recusa_correta"])
            )
            colunas[2].metric(
                "Sem afirmação insegura",
                exibir_contagem(
                    auxiliares["casos_sem_afirmacao_publicada_insegura"]
                ),
            )
            if base.get("casos"):
                det_base = base["metricas_deterministicas"]
                aux_base = base["metricas_auxiliares_qwen"]
                st.caption(
                    "Linha de base corrigida: fonte "
                    f"{exibir_contagem(det_base['fonte_correta'])}, recusas "
                    f"{exibir_contagem(det_base['recusa_correta'])}, casos sem afirmação "
                    f"insegura {exibir_contagem(aux_base['casos_sem_afirmacao_publicada_insegura'])}."
                )
            st.caption(auxiliares["aviso"])
            cobertura = rastreabilidade[
                "cobertura_media_evidencias_afirmacoes"
            ]
            st.caption(
                "Rastreabilidade determinística: cobertura afirmação→evidência "
                f"{cobertura:.0%}; afirmações publicadas sem evidência "
                f"{rastreabilidade['afirmacoes_publicadas_sem_evidencia']}."
                if cobertura is not None
                else "Rastreabilidade por IDs: não aplicável ao modo Compatibilidade."
            )
            for item in resultados:
                falhou = not resultado_aprovado(item)
                if falhou:
                    with st.expander(f"Atenção · {item.pergunta}"):
                        st.write(f"Documento: {item.documento}")
                        st.write(f"Páginas: {list(item.paginas_retornadas)}")
                        st.write(item.resposta)


materias = materias_atuais()
colecao = colecao_opcional()
resumo = cabecalho(materias, colecao)

st.sidebar.title("Navegação")
area = st.sidebar.radio(
    "Área", ("Conversar", "Documentos", "Matérias", "Avaliação"), index=0
)

if area == "Conversar":
    pagina_conversar(materias, colecao, resumo)
elif area == "Documentos":
    pagina_documentos(materias)
elif area == "Matérias":
    pagina_materias(materias, colecao)
else:
    pagina_avaliacao(materias, colecao)
