"""Modo de resposta auditável, restrito a um único PDF por consulta."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Sequence

from ollama import Client, ResponseError

from .chat import (
    ErroConsulta,
    TrechoRecuperado,
    _chave,
    abrir_colecao,
    filtro_chroma,
    gerar_embedding_pergunta,
    manifesto_compativel,
    normalizar_termos,
    recuperar_trechos,
    resposta_no_idioma,
    termos_consulta,
    verificar_ollama_e_modelos,
)
from .config import MINIMO_CANDIDATOS, MODELO_CONVERSA, MODELO_EMBEDDINGS, OLLAMA_HOST


NIVEIS_DETALHE = ("Curto", "Explicado", "Passo a passo")
MODO_AUTOMATICO = "Automático"


@dataclass(frozen=True)
class TrechoRotulado:
    """Rótulo efêmero de um trecho recuperado durante uma única consulta."""

    rotulo: str
    id_chroma: str
    arquivo: str
    pagina: int
    indice: int
    texto: str


@dataclass(frozen=True)
class EvidenciaOrganizada:
    id: str
    tipo: str
    conteudo: str
    natureza: str
    trecho_ids: tuple[str, ...]
    ids_chroma: tuple[str, ...]
    arquivo: str
    paginas: tuple[int, ...]

    @property
    def pagina(self) -> int:
        """Compatibilidade com consumidores antigos que exibiam uma página."""
        return self.paginas[0] if self.paginas else 0


@dataclass(frozen=True)
class AfirmacaoVerificada:
    texto_original: str
    texto_final: str
    classificacao: str
    paginas: tuple[int, ...]
    natureza: str
    secao: str
    justificativa: str = ""
    evidencia_ids: tuple[str, ...] = ()
    fontes: tuple[tuple[str, int], ...] = ()
    origem_vinculo: str = "ausente"
    ids_evidencia_invalidos: tuple[str, ...] = ()


@dataclass
class DiagnosticoEstrutural:
    """Tentativas rejeitadas deterministicamente no encadeamento de IDs."""

    trecho_ids_invalidos_rejeitados: list[str] = field(default_factory=list)
    evidencia_ids_invalidos_rejeitados: list[str] = field(default_factory=list)
    ids_adicionados_auditor_rejeitados: list[str] = field(default_factory=list)
    tentativas_mistura_arquivos: int = 0
    evidencias_sem_trecho_rejeitadas: int = 0
    afirmacoes_sem_evidencia: int = 0

    def como_dict(self) -> dict:
        return {
            "trecho_ids_invalidos_rejeitados": list(
                dict.fromkeys(self.trecho_ids_invalidos_rejeitados)
            ),
            "evidencia_ids_invalidos_rejeitados": list(
                dict.fromkeys(self.evidencia_ids_invalidos_rejeitados)
            ),
            "ids_adicionados_auditor_rejeitados": list(
                dict.fromkeys(self.ids_adicionados_auditor_rejeitados)
            ),
            "tentativas_mistura_arquivos": self.tentativas_mistura_arquivos,
            "evidencias_sem_trecho_rejeitadas": self.evidencias_sem_trecho_rejeitadas,
            "afirmacoes_sem_evidencia": self.afirmacoes_sem_evidencia,
        }


@dataclass(frozen=True)
class ResultadoFundamentado:
    documento: str
    motivo_documento: str
    trechos: tuple[TrechoRecuperado, ...]
    evidencias: tuple[EvidenciaOrganizada, ...]
    afirmacoes: tuple[AfirmacaoVerificada, ...]
    resposta: str
    insuficiente: bool
    informacao_faltante: str = ""
    trechos_rotulados: tuple[TrechoRotulado, ...] = ()
    diagnostico_estrutural: DiagnosticoEstrutural = field(
        default_factory=DiagnosticoEstrutural
    )

    @property
    def nao_sustentadas(self) -> int:
        return sum(item.classificacao == "não sustentada" for item in self.afirmacoes)


def _pontuacao(trecho: TrechoRecuperado) -> float:
    return trecho.pontuacao_fusao or trecho.relevancia or 0.0


def selecionar_fonte(
    candidatos: Sequence[TrechoRecuperado], arquivo: str | None = None
) -> tuple[str, str]:
    if arquivo:
        correspondentes = [item for item in candidatos if item.arquivo == arquivo]
        if not correspondentes:
            raise ErroConsulta(f"O PDF selecionado não possui candidatos: '{arquivo}'.")
        melhor = max(_pontuacao(item) for item in correspondentes)
        return arquivo, (
            f"PDF escolhido manualmente; melhor evidência inicial={melhor:.3f}."
        )
    por_arquivo: dict[str, list[TrechoRecuperado]] = {}
    for trecho in candidatos[:MINIMO_CANDIDATOS]:
        por_arquivo.setdefault(trecho.arquivo, []).append(trecho)
    if not por_arquivo:
        raise ErroConsulta("Nenhum documento apresentou evidência utilizável.")

    def nota(itens: list[TrechoRecuperado]) -> float:
        melhores = sorted((_pontuacao(item) for item in itens), reverse=True)[:3]
        pesos = (1.0, 0.5, 0.25)
        return sum(valor * pesos[posicao] for posicao, valor in enumerate(melhores))

    escolhido, itens = max(por_arquivo.items(), key=lambda par: nota(par[1]))
    return escolhido, (
        f"Escolha automática: nota agregada={nota(itens):.3f}, "
        f"{len(itens)} de {min(len(candidatos), MINIMO_CANDIDATOS)} candidatos fortes "
        "vieram deste PDF. A segunda busca foi restrita a ele."
    )


def similaridade_textual(a: str, b: str) -> float:
    termos_a, termos_b = set(normalizar_termos(a)), set(normalizar_termos(b))
    if not termos_a or not termos_b:
        return 0.0
    return len(termos_a & termos_b) / len(termos_a | termos_b)


def remover_quase_duplicados(
    candidatos: Sequence[TrechoRecuperado], limite: float = 0.86
) -> list[TrechoRecuperado]:
    unicos: list[TrechoRecuperado] = []
    for candidato in candidatos:
        if any(
            similaridade_textual(candidato.texto, existente.texto) >= limite
            for existente in unicos
        ):
            continue
        unicos.append(candidato)
    return unicos


def selecionar_evidencias(
    candidatos: Sequence[TrechoRecuperado], minimo: int = 4, maximo: int = 6
) -> list[TrechoRecuperado]:
    """Seleção MMR simples: relevância, novidade e corte de candidatos fracos."""
    unicos = remover_quase_duplicados(candidatos)
    if not unicos:
        return []
    melhor = max(_pontuacao(item) for item in unicos)
    elegiveis = [item for item in unicos if _pontuacao(item) >= melhor * 0.45]
    if not elegiveis:
        return []
    selecionados = [max(elegiveis, key=_pontuacao)]
    restantes = [item for item in elegiveis if item != selecionados[0]]
    while restantes and len(selecionados) < maximo:
        def mmr(item: TrechoRecuperado) -> float:
            redundancia = max(
                similaridade_textual(item.texto, atual.texto)
                for atual in selecionados
            )
            return 0.75 * _pontuacao(item) + 0.25 * (1.0 - redundancia)

        proximo = max(restantes, key=mmr)
        # Não preenche o limite com evidência claramente inferior.
        if len(selecionados) >= minimo and _pontuacao(proximo) < melhor * 0.60:
            break
        selecionados.append(proximo)
        restantes.remove(proximo)
    return selecionados


def incluir_vizinhas_relevantes(
    colecao: object,
    pergunta: str,
    evidencias: Sequence[TrechoRecuperado],
    disciplina: str | None,
    arquivo: str,
    maximo: int = 6,
) -> list[TrechoRecuperado]:
    if not evidencias or len(evidencias) >= maximo:
        return list(evidencias[:maximo])
    argumentos: dict = {"include": ["documents", "metadatas"]}
    if filtro := filtro_chroma(disciplina, arquivo):
        argumentos["where"] = filtro
    dados = colecao.get(**argumentos)
    ids = dados.get("ids") or []
    consulta = termos_consulta(pergunta)
    vizinhas: list[TrechoRecuperado] = []
    paginas_alvo = {
        pagina
        for origem in evidencias[:2]
        for pagina in (origem.pagina - 1, origem.pagina + 1)
    }
    for posicao, (texto, metadata) in enumerate(
        zip(dados.get("documents") or [], dados.get("metadatas") or [])
    ):
        pagina = int(metadata.get("pagina", 0))
        if pagina not in paginas_alvo:
            continue
        sobreposicao = len(set(normalizar_termos(str(texto))) & consulta)
        if sobreposicao == 0:
            continue
        vizinhas.append(
            TrechoRecuperado(
                texto=str(texto),
                arquivo=arquivo,
                pagina=pagina,
                indice=int(metadata.get("indice_trecho", 0)),
                disciplina=str(metadata.get("disciplina") or "Sem disciplina"),
                id=str(ids[posicao]) if posicao < len(ids) else "",
                pontuacao_fusao=max(0.01, sobreposicao / max(1, len(consulta))),
                pagina_vizinha=True,
            )
        )
    resultado = list(evidencias)
    vistos = {_chave(item) for item in resultado}
    for vizinha in sorted(vizinhas, key=_pontuacao, reverse=True):
        if len(resultado) >= maximo:
            break
        if _chave(vizinha) in vistos:
            continue
        if any(similaridade_textual(vizinha.texto, item.texto) >= 0.86 for item in resultado):
            continue
        resultado.append(vizinha)
        vistos.add(_chave(vizinha))
    return resultado


def rotular_trechos(
    trechos: Sequence[TrechoRecuperado],
) -> list[TrechoRotulado]:
    """Cria T1..Tn sem alterar nem inferir os IDs reais do ChromaDB."""
    return [
        TrechoRotulado(
            rotulo=f"T{numero}",
            id_chroma=item.id,
            arquivo=item.arquivo,
            pagina=item.pagina,
            indice=item.indice,
            texto=item.texto,
        )
        for numero, item in enumerate(trechos, start=1)
    ]


def _contexto(trechos: Sequence[TrechoRotulado]) -> str:
    return "\n\n".join(
        f"[{item.rotulo}] Arquivo: {item.arquivo} | Página do PDF: {item.pagina} | "
        f"Índice do trecho: {item.indice}\n{item.texto}"
        for item in trechos
    )


def _evidencia_para_modelo(item: EvidenciaOrganizada) -> dict:
    """Expõe ao modelo somente os IDs efêmeros já validados."""
    return {
        "id": item.id,
        "tipo": item.tipo,
        "conteudo": item.conteudo,
        "natureza": item.natureza,
        "trecho_ids": list(item.trecho_ids),
    }


def _fontes_evidencias(
    evidencia_ids: Sequence[str],
    evidencias: Sequence[EvidenciaOrganizada],
) -> tuple[tuple[str, int], ...]:
    por_id = {item.id: item for item in evidencias}
    return tuple(
        dict.fromkeys(
            (evidencia.arquivo, pagina)
            for evidencia_id in evidencia_ids
            if (evidencia := por_id.get(evidencia_id)) is not None
            for pagina in evidencia.paginas
        )
    )


def _conteudo(resposta: object) -> str:
    mensagem = getattr(resposta, "message", None)
    return str(
        getattr(mensagem, "content", None)
        or (mensagem.get("content") if isinstance(mensagem, dict) else "")
    ).strip()


def _json_modelo(
    cliente: Client, system: str, user: str, etapa: str
) -> dict:
    ultimo_erro: json.JSONDecodeError | None = None
    for tentativa in range(2):
        try:
            resposta = cliente.chat(
                model=MODELO_CONVERSA,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": user
                        + ("\n\nA saída anterior foi inválida. Retorne SOMENTE um objeto JSON completo." if tentativa else ""),
                    },
                ],
                format="json",
                stream=False,
                options={"temperature": 0, "num_predict": 800},
            )
        except ResponseError as erro:
            raise ErroConsulta(f"Falha local na etapa '{etapa}': {erro.error}") from erro
        except Exception as erro:
            raise ErroConsulta(f"Ollama indisponível durante a etapa '{etapa}'.") from erro
        texto = _conteudo(resposta).strip().removeprefix("```json").removesuffix("```").strip()
        candidatos = [texto]
        if "{" in texto and "}" in texto:
            candidatos.append(texto[texto.find("{") : texto.rfind("}") + 1])
        for candidato in candidatos:
            try:
                dados = json.loads(candidato)
                if isinstance(dados, dict):
                    return dados
            except json.JSONDecodeError as erro:
                ultimo_erro = erro
    raise ErroConsulta(
        f"O modelo retornou JSON inválido duas vezes na etapa '{etapa}'."
    ) from ultimo_erro


def organizar_evidencias(
    cliente: Client,
    pergunta: str,
    trechos: Sequence[TrechoRecuperado],
    *,
    trechos_rotulados: Sequence[TrechoRotulado] | None = None,
    diagnostico: DiagnosticoEstrutural | None = None,
) -> tuple[list[EvidenciaOrganizada], bool, str]:
    diagnostico = diagnostico or DiagnosticoEstrutural()
    rotulados = list(trechos_rotulados or rotular_trechos(trechos))
    por_rotulo = {item.rotulo: item for item in rotulados}
    dados = _json_modelo(
        cliente,
        "Você é um extrator de evidências. Use somente os trechos identificados por Tn. "
        "Não responda à pergunta, não crie IDs e não devolva arquivo ou página. "
        "Retorne JSON válido, sem markdown.",
        f"""Pergunta: {pergunta}

Organize fatos explícitos, definições, fórmulas, condições e limitações.
Marque natureza como texto_explicito ou deducao_simples. Uma dedução simples deve
decorrer diretamente das evidências, sem conhecimento externo.
Cada evidência deve listar em trecho_ids somente rótulos Tn apresentados abaixo.
Uma evidência pode usar vários trechos quando a informação continua em outra página.

Formato:
{{"suficiente": true, "informacao_faltante": "", "evidencias": [
  {{"tipo": "fato|definicao|formula|condicao|limitacao", "conteudo": "...",
    "trecho_ids": ["T1"], "natureza": "texto_explicito|deducao_simples"}}
]}}

Trechos:
{_contexto(rotulados)}""",
        "organização das evidências",
    )
    evidencias: list[EvidenciaOrganizada] = []
    for item in dados.get("evidencias", []):
        if not isinstance(item, dict):
            continue
        conteudo = str(item.get("conteudo") or "").strip()
        ids_brutos = item.get("trecho_ids")
        if not isinstance(ids_brutos, list):
            diagnostico.evidencias_sem_trecho_rejeitadas += 1
            continue
        trecho_ids = tuple(
            dict.fromkeys(
                str(rotulo).strip()
                for rotulo in ids_brutos
                if str(rotulo).strip()
            )
        )
        if not trecho_ids:
            diagnostico.evidencias_sem_trecho_rejeitadas += 1
            continue
        invalidos = [rotulo for rotulo in trecho_ids if rotulo not in por_rotulo]
        if invalidos:
            diagnostico.trecho_ids_invalidos_rejeitados.extend(invalidos)
            continue
        selecionados = [por_rotulo[rotulo] for rotulo in trecho_ids]
        sem_id_real = [item.rotulo for item in selecionados if not item.id_chroma]
        if sem_id_real:
            diagnostico.trecho_ids_invalidos_rejeitados.extend(sem_id_real)
            continue
        arquivos = {item.arquivo for item in selecionados}
        if len(arquivos) != 1:
            diagnostico.tentativas_mistura_arquivos += 1
            continue
        if not conteudo:
            continue
        natureza = str(item.get("natureza") or "texto_explicito")
        if natureza not in {"texto_explicito", "deducao_simples"}:
            natureza = "texto_explicito"
        nova = EvidenciaOrganizada(
            id=f"E{len(evidencias) + 1}",
            tipo=str(item.get("tipo") or "fato"),
            conteudo=conteudo,
            natureza=natureza,
            trecho_ids=trecho_ids,
            ids_chroma=tuple(item.id_chroma for item in selecionados),
            arquivo=selecionados[0].arquivo,
            paginas=tuple(dict.fromkeys(item.pagina for item in selecionados)),
        )
        if any(
            similaridade_textual(existente.conteudo, nova.conteudo) >= 0.80
            for existente in evidencias
        ):
            continue
        evidencias.append(nova)
    suficiente = bool(dados.get("suficiente")) and bool(evidencias)
    faltando = str(dados.get("informacao_faltante") or "").strip()
    return evidencias, suficiente, faltando


def redigir_rascunho(
    cliente: Client,
    pergunta: str,
    evidencias: Sequence[EvidenciaOrganizada],
    idioma: str,
    nivel: str,
    *,
    conservador: bool = False,
    diagnostico: DiagnosticoEstrutural | None = None,
) -> list[dict]:
    diagnostico = diagnostico or DiagnosticoEstrutural()
    instrucao_conservadora = (
        "Copie ou traduza fielmente a evidência mais direta, alterando apenas o necessário "
        "para formar uma frase clara. Não generalize e não introduza termos novos."
        if conservador
        else "A primeira afirmação deve responder diretamente à pergunta usando a evidência mais direta."
    )
    dados = _json_modelo(
        cliente,
        f"Escreva exclusivamente em {idioma}. Use somente as evidências estruturadas. "
        "Retorne JSON válido, não acrescente conhecimento externo e não crie IDs.",
        f"""Pergunta: {pergunta}
Nível de detalhe: {nivel}

Crie afirmações concisas. Fórmula ou exemplo somente se houver evidência desse tipo.
Para definição curta, não alongue. Toda afirmação factual deve declarar evidencia_ids.
Use somente IDs En apresentados abaixo. Não devolva arquivo nem página.
{instrucao_conservadora}

Formato:
{{"afirmacoes": [{{"texto": "...", "secao": "resposta_direta|explicacao|formula|exemplo|limitacao",
"evidencia_ids": ["E1"], "natureza": "texto_explicito|deducao_simples"}}]}}

Evidências:
{json.dumps([_evidencia_para_modelo(item) for item in evidencias], ensure_ascii=False)}""",
        "redação da resposta",
    )
    ids_evidencia = {item.id for item in evidencias}
    limite = {"Curto": 2, "Explicado": 4, "Passo a passo": 6}.get(nivel, 4)
    afirmacoes = []
    itens_afirmacoes = dados.get("afirmacoes")
    if itens_afirmacoes is None:
        # Modelos pequenos por vezes localizam também a chave do JSON. Isso não
        # altera nem completa os vínculos: cada item ainda precisa declarar seus
        # próprios evidencia_ids válidos.
        itens_afirmacoes = dados.get("afirmações", [])
    if not isinstance(itens_afirmacoes, list):
        itens_afirmacoes = []
    for item in itens_afirmacoes:
        if not isinstance(item, dict) or not str(item.get("texto") or "").strip():
            continue
        secao_bruta = str(item.get("secao") or "explicacao")
        ids_brutos = item.get("evidencia_ids")
        evidencia_ids = tuple(
            dict.fromkeys(
                str(evidencia_id).strip()
                for evidencia_id in ids_brutos
                if str(evidencia_id).strip()
            )
        ) if isinstance(ids_brutos, list) else ()
        invalidos = tuple(
            evidencia_id
            for evidencia_id in evidencia_ids
            if evidencia_id not in ids_evidencia
        )
        if invalidos:
            diagnostico.evidencia_ids_invalidos_rejeitados.extend(invalidos)
        if not evidencia_ids:
            diagnostico.afirmacoes_sem_evidencia += 1
        natureza = str(item.get("natureza") or "")
        if natureza not in {"texto_explicito", "deducao_simples"}:
            natureza = (
                "deducao_simples" if "deducao_simples" in secao_bruta
                else "texto_explicito"
            )
        afirmacoes.append(
            {
                "texto": str(item["texto"]).strip(),
                "secao": secao_bruta.split("|", 1)[0],
                "evidencia_ids": list(evidencia_ids),
                "ids_evidencia_invalidos": list(invalidos),
                "diagnostico_estrutural_registrado": True,
                "natureza": natureza,
            }
        )
    return afirmacoes[:limite]


def _verificar_afirmacoes_legado(
    cliente: Client,
    rascunho: Sequence[dict],
    trechos: Sequence[TrechoRecuperado],
    idioma: str,
) -> list[AfirmacaoVerificada]:
    permitidas = {item.pagina for item in trechos}
    originais = [dict(item, id=f"A{numero}") for numero, item in enumerate(rascunho, 1)]

    def auditar_lote(lote: Sequence[dict], etapa: str) -> list[dict]:
        dados = _json_modelo(
            cliente,
            f"Você é um auditor factual rigoroso. Escreva exclusivamente em {idioma}. "
            "Audite todas as afirmações recebidas usando somente os trechos. Retorne JSON válido.",
            f"""Para CADA id recebido, devolva exatamente um item com o mesmo id.
Classifique como sustentada, parcialmente sustentada ou não sustentada.
Em texto_final, remova qualquer parte sem suporte. Para não sustentada, use texto_final vazio.
Use apenas páginas presentes nos trechos. Não omita nenhum id.

Formato:
{{"afirmacoes": [{{"id": "A1", "texto_original": "...", "texto_final": "...",
"classificacao": "sustentada|parcialmente sustentada|não sustentada", "paginas": [1],
"natureza": "texto_explicito|deducao_simples", "secao": "resposta_direta|explicacao|formula|exemplo|limitacao",
"justificativa": "..."}}]}}

Afirmações:
{json.dumps(list(lote), ensure_ascii=False)}

 Trechos autorizados:
{_contexto(rotular_trechos(trechos))}""",
            etapa,
        )
        return [item for item in dados.get("afirmacoes", []) if isinstance(item, dict)]

    try:
        auditadas = auditar_lote(originais, "verificação factual")
    except ErroConsulta:
        # A auditoria por LLM é auxiliar. Saída inválida reprova de forma
        # conservadora, sem apagar as métricas determinísticas da execução.
        auditadas = []
    por_id = {str(item.get("id")): item for item in auditadas if item.get("id")}
    # Se o modelo omitir um id, audita apenas o item ausente em uma chamada curta.
    fallbacks = 0
    for original in originais:
        if original["id"] not in por_id:
            if fallbacks >= 2:
                continue
            try:
                recuperado = auditar_lote(
                    [original], f"verificação factual de {original['id']}"
                )
            except ErroConsulta:
                # Auditoria auxiliar inválida não pode aprovar a afirmação nem
                # impedir a conclusão das métricas determinísticas.
                recuperado = []
            fallbacks += 1
            if recuperado:
                por_id[original["id"]] = recuperado[0]

    verificadas: list[AfirmacaoVerificada] = []
    for original in originais:
        item = por_id.get(original["id"], {})
        classificacao = unicodedata.normalize(
            "NFC", str(item.get("classificacao") or "não sustentada").strip().casefold()
        )
        if classificacao not in {
            "sustentada", "parcialmente sustentada", "não sustentada"
        }:
            classificacao = "não sustentada"
        paginas = tuple(
            dict.fromkeys(
                int(pagina)
                for pagina in item.get("paginas", [])
                if str(pagina).isdigit() and int(pagina) in permitidas
            )
        )
        texto_final = str(item.get("texto_final") or "").strip()
        if not paginas or not texto_final:
            classificacao = "não sustentada"
            texto_final = ""
        natureza = str(
            item.get("natureza") or original.get("natureza") or "texto_explicito"
        )
        if natureza not in {"texto_explicito", "deducao_simples"}:
            natureza = "texto_explicito"
        verificadas.append(
            AfirmacaoVerificada(
                texto_original=str(
                    item.get("texto_original") or original.get("texto") or ""
                ).strip(),
                texto_final=texto_final,
                classificacao=classificacao,
                paginas=paginas,
                natureza=natureza,
                secao=str(item.get("secao") or original.get("secao") or "explicacao"),
                justificativa=str(item.get("justificativa") or "").strip(),
                fontes=tuple(
                    dict.fromkeys(
                        (trecho.arquivo, trecho.pagina)
                        for trecho in trechos
                        if trecho.pagina in paginas
                    )
                ),
                origem_vinculo="reconstruido_auxiliar",
            )
        )
    return verificadas


def verificar_afirmacoes(
    cliente: Client,
    rascunho: Sequence[dict],
    trechos: Sequence[TrechoRecuperado],
    idioma: str,
    *,
    evidencias: Sequence[EvidenciaOrganizada] | None = None,
    trechos_rotulados: Sequence[TrechoRotulado] | None = None,
    diagnostico: DiagnosticoEstrutural | None = None,
    origem_vinculo: str = "geracao_validada",
) -> list[AfirmacaoVerificada]:
    """Audita cada afirmação apenas contra os vínculos de evidência declarados."""
    if evidencias is None:
        return _verificar_afirmacoes_legado(cliente, rascunho, trechos, idioma)

    diagnostico = diagnostico or DiagnosticoEstrutural()
    rotulados = list(trechos_rotulados or rotular_trechos(trechos))
    por_rotulo = {item.rotulo: item for item in rotulados}
    por_evidencia = {item.id: item for item in evidencias}
    verificadas: list[AfirmacaoVerificada] = []

    for numero, bruto in enumerate(rascunho, start=1):
        original = dict(bruto)
        id_afirmacao = f"A{numero}"
        texto = str(original.get("texto") or "").strip()
        texto_publicado = str(
            original.get("texto_original_publicado") or texto
        ).strip()
        ids_brutos = original.get("evidencia_ids")
        evidencia_ids = tuple(
            dict.fromkeys(
                str(evidencia_id).strip()
                for evidencia_id in ids_brutos
                if str(evidencia_id).strip()
            )
        ) if isinstance(ids_brutos, list) else ()
        invalidos = tuple(
            dict.fromkeys(
                [
                    str(item).strip()
                    for item in original.get("ids_evidencia_invalidos", [])
                    if str(item).strip()
                ]
                + [
                    evidencia_id
                    for evidencia_id in evidencia_ids
                    if evidencia_id not in por_evidencia
                ]
            )
        )
        registrado = bool(original.get("diagnostico_estrutural_registrado"))
        if invalidos and not registrado:
            diagnostico.evidencia_ids_invalidos_rejeitados.extend(invalidos)
        if not evidencia_ids and not registrado:
            diagnostico.afirmacoes_sem_evidencia += 1

        selecionadas = [
            por_evidencia[evidencia_id]
            for evidencia_id in evidencia_ids
            if evidencia_id in por_evidencia
        ]
        rotulos_associados = tuple(
            dict.fromkeys(
                rotulo
                for evidencia in selecionadas
                for rotulo in evidencia.trecho_ids
            )
        )
        rotulos_invalidos = [
            rotulo for rotulo in rotulos_associados if rotulo not in por_rotulo
        ]
        if rotulos_invalidos:
            diagnostico.trecho_ids_invalidos_rejeitados.extend(rotulos_invalidos)

        trechos_associados = [
            por_rotulo[rotulo]
            for rotulo in rotulos_associados
            if rotulo in por_rotulo
        ]
        arquivos = {item.arquivo for item in trechos_associados}
        ids_reais_por_rotulo = {
            item.rotulo: item.id_chroma for item in trechos_associados
        }
        relacao_integra = all(
            tuple(ids_reais_por_rotulo.get(rotulo, "") for rotulo in evidencia.trecho_ids)
            == evidencia.ids_chroma
            and all(
                por_rotulo.get(rotulo) is not None
                and por_rotulo[rotulo].arquivo == evidencia.arquivo
                and por_rotulo[rotulo].pagina in evidencia.paginas
                for rotulo in evidencia.trecho_ids
            )
            for evidencia in selecionadas
        )
        mistura_arquivos = len(arquivos) > 1 or len(
            {item.arquivo for item in selecionadas}
        ) > 1
        if mistura_arquivos:
            diagnostico.tentativas_mistura_arquivos += 1

        estrutura_valida = bool(
            texto
            and evidencia_ids
            and not invalidos
            and len(selecionadas) == len(evidencia_ids)
            and not rotulos_invalidos
            and trechos_associados
            and relacao_integra
            and not mistura_arquivos
        )
        fontes = _fontes_evidencias(evidencia_ids, selecionadas)
        paginas = tuple(dict.fromkeys(pagina for _, pagina in fontes))
        natureza = str(original.get("natureza") or "texto_explicito")
        if natureza not in {"texto_explicito", "deducao_simples"}:
            natureza = "texto_explicito"
        secao = str(original.get("secao") or "explicacao")

        if not estrutura_valida:
            verificadas.append(
                AfirmacaoVerificada(
                    texto_original=texto_publicado,
                    texto_final="",
                    classificacao="não sustentada",
                    paginas=paginas,
                    natureza=natureza,
                    secao=secao,
                    justificativa=(
                        "A validação determinística rejeitou IDs ausentes, inexistentes "
                        "ou relações inconsistentes entre evidência e trecho."
                    ),
                    evidencia_ids=evidencia_ids,
                    fontes=fontes,
                    origem_vinculo=origem_vinculo,
                    ids_evidencia_invalidos=invalidos,
                )
            )
            continue

        try:
            dados = _json_modelo(
                cliente,
                f"Você é um auditor factual rigoroso. Escreva exclusivamente em {idioma}. "
                "Use somente as evidências e os trechos associados a esta afirmação. "
                "Não crie, remova ou substitua IDs. Retorne JSON válido.",
                f"""Classifique a afirmação como sustentada, parcialmente sustentada ou não sustentada.
Em texto_final, remova qualquer parte sem suporte. Para não sustentada, use texto_final vazio.
Devolva o mesmo id e exatamente os mesmos evidencia_ids. Não devolva páginas ou arquivos.

Formato:
{{"afirmacoes": [{{"id": "{id_afirmacao}", "texto_final": "...",
"classificacao": "sustentada|parcialmente sustentada|não sustentada",
"evidencia_ids": {json.dumps(list(evidencia_ids), ensure_ascii=False)},
"justificativa": "..."}}]}}

Afirmação:
{json.dumps({"id": id_afirmacao, "texto": texto, "evidencia_ids": list(evidencia_ids), "natureza": natureza, "secao": secao}, ensure_ascii=False)}

Evidências indicadas:
{json.dumps([_evidencia_para_modelo(item) for item in selecionadas], ensure_ascii=False)}

Trechos associados:
{_contexto(trechos_associados)}""",
                f"verificação factual de {id_afirmacao}",
            )
        except ErroConsulta:
            verificadas.append(
                AfirmacaoVerificada(
                    texto_original=texto_publicado,
                    texto_final="",
                    classificacao="não sustentada",
                    paginas=paginas,
                    natureza=natureza,
                    secao=secao,
                    justificativa=(
                        "O auditor auxiliar retornou estrutura inválida; a afirmação "
                        "foi reprovada conservadoramente."
                    ),
                    evidencia_ids=evidencia_ids,
                    fontes=fontes,
                    origem_vinculo=origem_vinculo,
                )
            )
            continue
        itens_brutos = dados.get("afirmacoes")
        if itens_brutos is None:
            itens_brutos = dados.get("afirmações")
        if itens_brutos is None and str(dados.get("id")) == id_afirmacao:
            # A auditoria trata uma única afirmação. O Qwen ocasionalmente omite
            # apenas o envelope da lista, preservando o id e evidencia_ids
            # explícitos; aceitar esse envelope curto não cria nem completa IDs.
            itens_brutos = [dados]
        elif isinstance(itens_brutos, dict):
            itens_brutos = [itens_brutos]
        elif not isinstance(itens_brutos, list):
            itens_brutos = []
        itens = [
            item
            for item in itens_brutos
            if isinstance(item, dict) and str(item.get("id")) == id_afirmacao
        ]
        auditada = itens[0] if itens else {}
        ids_auditor_brutos = auditada.get("evidencia_ids")
        ids_auditor = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in ids_auditor_brutos
                if str(item).strip()
            )
        ) if isinstance(ids_auditor_brutos, list) else ()
        adicionados = [
            evidencia_id
            for evidencia_id in ids_auditor
            if evidencia_id not in evidencia_ids
        ]
        diagnostico.ids_adicionados_auditor_rejeitados.extend(adicionados)
        ids_preservados = ids_auditor == evidencia_ids

        classificacao = unicodedata.normalize(
            "NFC",
            str(auditada.get("classificacao") or "não sustentada")
            .strip()
            .casefold(),
        )
        if classificacao not in {
            "sustentada", "parcialmente sustentada", "não sustentada"
        }:
            classificacao = "não sustentada"
        texto_final = str(auditada.get("texto_final") or "").strip()
        if not ids_preservados:
            classificacao = "não sustentada"
            texto_final = ""
        elif classificacao == "sustentada" and not texto_final:
            # Se o auditor aprovou integralmente a afirmação e preservou todos
            # os IDs, o texto original já é a versão final auditada. Reescrita
            # continua obrigatória para classificação parcial.
            texto_final = texto
        elif not texto_final:
            classificacao = "não sustentada"
        verificadas.append(
            AfirmacaoVerificada(
                texto_original=texto_publicado,
                texto_final=texto_final if classificacao != "não sustentada" else "",
                classificacao=classificacao,
                paginas=paginas,
                natureza=natureza,
                secao=secao,
                justificativa=(
                    str(auditada.get("justificativa") or "").strip()
                    if ids_preservados
                    else "O auditor não preservou exatamente os IDs de evidência."
                ),
                evidencia_ids=evidencia_ids,
                fontes=fontes,
                origem_vinculo=origem_vinculo,
                ids_evidencia_invalidos=invalidos,
            )
        )
    return verificadas


def montar_resposta_verificada(
    arquivo: str,
    afirmacoes: Sequence[AfirmacaoVerificada],
    nivel: str,
    informacao_faltante: str = "",
    *,
    evidencias: Sequence[EvidenciaOrganizada] | None = None,
) -> tuple[str, bool]:
    por_evidencia = {item.id: item for item in evidencias or []}

    def fontes_da_afirmacao(
        item: AfirmacaoVerificada,
    ) -> tuple[tuple[str, int], ...]:
        if evidencias is not None:
            if not item.evidencia_ids or any(
                evidencia_id not in por_evidencia
                for evidencia_id in item.evidencia_ids
            ):
                return ()
            fontes = _fontes_evidencias(item.evidencia_ids, evidencias)
            if {fonte_arquivo for fonte_arquivo, _ in fontes} != {arquivo}:
                return ()
            return fontes
        if item.fontes:
            return item.fontes
        return tuple((arquivo, pagina) for pagina in item.paginas)

    aceitas = [
        (item, fontes)
        for item in afirmacoes
        if item.classificacao == "sustentada" and item.texto_final
        if (fontes := fontes_da_afirmacao(item))
    ]
    limite = {"Curto": 2, "Explicado": 5, "Passo a passo": 6}.get(nivel, 5)
    aceitas = aceitas[:limite]
    if not aceitas:
        faltando = informacao_faltante or (
            "faltam trechos que respondam diretamente à pergunta ou sustentem uma conclusão."
        )
        return (
            "Não encontrei evidência suficiente no PDF selecionado para responder com segurança. "
            f"Informação faltante: {faltando}",
            True,
        )

    linhas = []
    for indice, (item, fontes) in enumerate(aceitas, start=1):
        rotulo = (
            "Dedução simples"
            if item.natureza == "deducao_simples"
            else "Texto explícito da fonte"
        )
        citacoes = " ".join(
            f"[{fonte_arquivo}, página do PDF {pagina}]"
            for fonte_arquivo, pagina in fontes
        )
        prefixo = f"{indice}. " if nivel == "Passo a passo" else "- "
        linhas.append(f"{prefixo}**{rotulo}:** {item.texto_final} {citacoes}")
    fontes = tuple(
        dict.fromkeys(
            fonte
            for _, fontes_item in aceitas
            for fonte in fontes_item
        )
    )
    referencias = "\n".join(
        f"- [{fonte_arquivo}, página do PDF {pagina}]"
        for fonte_arquivo, pagina in fontes
    )
    return "\n".join(linhas) + f"\n\nFontes\n{referencias}", False


def traduzir_afirmacao(cliente: Client, texto: str, idioma: str) -> str:
    destino = "PORTUGUÊS DO BRASIL" if idioma == "Português" else "ENGLISH"
    try:
        resposta = cliente.chat(
            model=MODELO_CONVERSA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Traduza exclusivamente para {destino}. Preserve exatamente o significado, "
                        "símbolos e fórmulas. Não acrescente nem remova fatos. Retorne somente a tradução."
                    ),
                },
                {"role": "user", "content": texto},
            ],
            stream=False,
            options={"temperature": 0, "num_predict": 256},
        )
    except Exception as erro:
        raise ErroConsulta("Não foi possível corrigir o idioma de uma afirmação auditada.") from erro
    traducao = _conteudo(resposta)
    return traducao if resposta_no_idioma(traducao, idioma) else ""


def consultar_fundamentado(
    pergunta: str,
    *,
    disciplina: str | None = None,
    arquivo: str | None = None,
    idioma: str = "Português",
    nivel_detalhe: str = "Explicado",
    candidatos: int = MINIMO_CANDIDATOS,
    incluir_vizinhas: bool = True,
    cliente: Client | None = None,
    colecao: object | None = None,
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> ResultadoFundamentado:
    diagnostico = DiagnosticoEstrutural()
    pergunta = pergunta.strip()
    if not pergunta:
        raise ErroConsulta("A pergunta não pode estar vazia.")
    if nivel_detalhe not in NIVEIS_DETALHE:
        raise ErroConsulta(f"Nível de detalhe inválido: {nivel_detalhe}.")
    colecao = colecao or abrir_colecao()
    manifesto = manifesto_compativel(modelo_embeddings)
    cliente = cliente or Client(host=OLLAMA_HOST)
    verificar_ollama_e_modelos(cliente, modelo_embeddings)
    vetor = gerar_embedding_pergunta(cliente, pergunta, modelo_embeddings)
    if len(vetor) != manifesto.dimensao:
        raise ErroConsulta("A dimensão da consulta não coincide com o manifesto do índice.")

    # Primeira recuperação: escolhe o documento, sem forçar diversidade entre PDFs.
    _, ranking_inicial = recuperar_trechos(
        colecao,
        pergunta,
        vetor,
        top_k=6,
        candidatos=min(20, max(MINIMO_CANDIDATOS, candidatos)),
        disciplina=disciplina,
        busca_hibrida=True,
        diversificar_arquivos=False,
        arquivo=arquivo,
    )
    documento, motivo = selecionar_fonte(ranking_inicial, arquivo)

    # Segunda recuperação: todo o contexto passa a vir do único PDF escolhido.
    _, ranking_documento = recuperar_trechos(
        colecao,
        pergunta,
        vetor,
        top_k=6,
        candidatos=max(MINIMO_CANDIDATOS, candidatos),
        disciplina=disciplina,
        busca_hibrida=True,
        diversificar_arquivos=False,
        arquivo=documento,
    )
    termos_originais = set(normalizar_termos(pergunta))
    termos_expandidos = termos_consulta(pergunta)
    cobertura = max(
        (
            len(set(normalizar_termos(item.texto)) & termos_expandidos)
            for item in ranking_documento[:MINIMO_CANDIDATOS]
        ),
        default=0,
    )
    cobertura_minima = 1 if len(termos_originais) <= 2 else 2
    if cobertura < cobertura_minima:
        trechos_fracos = selecionar_evidencias(
            ranking_documento, minimo=1, maximo=4
        )
        rotulados_fracos = rotular_trechos(trechos_fracos)
        faltando = (
            "faltam no PDF os conceitos centrais da pergunta; a semelhança vetorial "
            "isolada não é evidência suficiente."
        )
        resposta, _ = montar_resposta_verificada(
            documento, [], nivel_detalhe, faltando, evidencias=[]
        )
        return ResultadoFundamentado(
            documento=documento,
            motivo_documento=motivo + f" Cobertura lexical insuficiente ({cobertura}/{cobertura_minima}).",
            trechos=tuple(trechos_fracos),
            evidencias=(),
            afirmacoes=(),
            resposta=resposta,
            insuficiente=True,
            informacao_faltante=faltando,
            trechos_rotulados=tuple(rotulados_fracos),
            diagnostico_estrutural=diagnostico,
        )
    # Reserva duas vagas para continuações em páginas adjacentes.
    trechos = selecionar_evidencias(ranking_documento, minimo=3, maximo=4)
    if incluir_vizinhas:
        trechos = incluir_vizinhas_relevantes(
            colecao, pergunta, trechos, disciplina, documento, maximo=6
        )
    trechos = remover_quase_duplicados(trechos)[:6]
    if not trechos:
        raise ErroConsulta("Não encontrei evidências suficientemente fortes no PDF escolhido.")
    trechos_rotulados = rotular_trechos(trechos)

    evidencias, suficiente, faltando = organizar_evidencias(
        cliente,
        pergunta,
        trechos,
        trechos_rotulados=trechos_rotulados,
        diagnostico=diagnostico,
    )
    if not suficiente:
        resposta, _ = montar_resposta_verificada(
            documento, [], nivel_detalhe, faltando, evidencias=evidencias
        )
        return ResultadoFundamentado(
            documento=documento,
            motivo_documento=motivo,
            trechos=tuple(trechos),
            evidencias=tuple(evidencias),
            afirmacoes=(),
            resposta=resposta,
            insuficiente=True,
            informacao_faltante=faltando,
            trechos_rotulados=tuple(trechos_rotulados),
            diagnostico_estrutural=diagnostico,
        )
    rascunho = redigir_rascunho(
        cliente,
        pergunta,
        evidencias,
        idioma,
        nivel_detalhe,
        diagnostico=diagnostico,
    )
    afirmacoes = verificar_afirmacoes(
        cliente,
        rascunho,
        trechos,
        idioma,
        evidencias=evidencias,
        trechos_rotulados=trechos_rotulados,
        diagnostico=diagnostico,
    )
    if not any(
        item.classificacao in {"sustentada", "parcialmente sustentada"}
        and item.texto_final
        for item in afirmacoes
    ):
        # Uma redação ruim não deve transformar boa evidência em uma recusa falsa.
        # Repete somente a etapa de redação, de modo mais extrativo, e audita novamente.
        rascunho = redigir_rascunho(
            cliente,
            pergunta,
            evidencias[:2],
            idioma,
            "Curto",
            conservador=True,
            diagnostico=diagnostico,
        )
        afirmacoes = verificar_afirmacoes(
            cliente,
            rascunho,
            trechos,
            idioma,
            evidencias=evidencias,
            trechos_rotulados=trechos_rotulados,
            diagnostico=diagnostico,
        )
    afirmacoes = [
        replace(
            item,
            texto_final=traduzir_afirmacao(cliente, item.texto_final, idioma),
        )
        if item.texto_final and not resposta_no_idioma(item.texto_final, idioma)
        else item
        for item in afirmacoes
    ]
    resposta, insuficiente = montar_resposta_verificada(
        documento,
        afirmacoes,
        nivel_detalhe,
        faltando,
        evidencias=evidencias,
    )
    if not resposta_no_idioma(resposta, idioma):
        raise ErroConsulta(
            f"A resposta verificada não respeitou o idioma selecionado ({idioma})."
        )
    return ResultadoFundamentado(
        documento=documento,
        motivo_documento=motivo,
        trechos=tuple(trechos),
        evidencias=tuple(evidencias),
        afirmacoes=tuple(afirmacoes),
        resposta=resposta,
        insuficiente=insuficiente,
        informacao_faltante=faltando,
        trechos_rotulados=tuple(trechos_rotulados),
        diagnostico_estrutural=diagnostico,
    )
