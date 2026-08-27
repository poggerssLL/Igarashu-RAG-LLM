"""Avalia a recuperação híbrida do RAG sem gerar respostas de conversa."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ollama import Client

from .chat import (
    ErroConsulta,
    TrechoRecuperado,
    abrir_colecao,
    gerar_embedding_pergunta,
    manifesto_compativel,
    recuperar_trechos,
)
from .config import MINIMO_CANDIDATOS, MODELO_EMBEDDINGS, OLLAMA_HOST, RAIZ_PROJETO
from .disciplinas import TODAS_DISCIPLINAS


ARQUIVO_CASOS = RAIZ_PROJETO / "avaliacao" / "casos_rag.json"
TOP_K_AVALIACAO = 4


@dataclass(frozen=True)
class ResultadoCaso:
    pergunta: str
    paginas_esperadas: list[int]
    paginas_retornadas: list[int]
    termos_encontrados: list[str]
    termos_ausentes: list[str]
    acertou_pagina: bool
    trechos: list[TrechoRecuperado]
    observacao: str


def normalizar_texto(texto: str) -> str:
    """Normaliza acentos e artefatos comuns da extração de texto de PDFs."""
    texto = re.sub(r"[´`ˆ˜¸˛]", "", texto.casefold())
    texto = texto.translate(str.maketrans({"ı": "i", "∗": "*", "−": "-"}))
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    return " ".join(texto.split())


def carregar_casos(caminho: Path = ARQUIVO_CASOS) -> list[dict]:
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except FileNotFoundError as erro:
        raise ErroConsulta(f"Arquivo de casos não encontrado: '{caminho}'.") from erro
    except json.JSONDecodeError as erro:
        raise ErroConsulta(f"JSON inválido em '{caminho}': {erro}.") from erro
    if not isinstance(dados, list) or not dados:
        raise ErroConsulta("O arquivo de avaliação deve conter uma lista não vazia de casos.")
    campos = {"pergunta", "paginas_esperadas", "termos_esperados", "observacao"}
    for indice, caso in enumerate(dados, start=1):
        if not isinstance(caso, dict) or not campos.issubset(caso):
            raise ErroConsulta(f"Caso {indice} não contém todos os campos obrigatórios.")
        if not caso["pergunta"].strip() or not caso["paginas_esperadas"] or not caso["termos_esperados"]:
            raise ErroConsulta(f"Caso {indice} possui pergunta, páginas ou termos vazios.")
    return dados


def avaliar_caso(caso: dict, trechos: Sequence[TrechoRecuperado]) -> ResultadoCaso:
    paginas_retornadas = list(dict.fromkeys(trecho.pagina for trecho in trechos))
    paginas_esperadas = [int(pagina) for pagina in caso["paginas_esperadas"]]
    texto_recuperado = normalizar_texto("\n".join(trecho.texto for trecho in trechos))
    encontrados: list[str] = []
    ausentes: list[str] = []
    for termo in caso["termos_esperados"]:
        destino = encontrados if normalizar_texto(str(termo)) in texto_recuperado else ausentes
        destino.append(str(termo))
    return ResultadoCaso(
        pergunta=caso["pergunta"],
        paginas_esperadas=paginas_esperadas,
        paginas_retornadas=paginas_retornadas,
        termos_encontrados=encontrados,
        termos_ausentes=ausentes,
        acertou_pagina=bool(set(paginas_esperadas) & set(paginas_retornadas)),
        trechos=list(trechos),
        observacao=caso["observacao"],
    )


def taxa_acerto_recuperacao(resultados: Sequence[ResultadoCaso]) -> float:
    if not resultados:
        return 0.0
    return sum(resultado.acertou_pagina for resultado in resultados) / len(resultados)


def executar_avaliacao(
    disciplina: str | None = None,
    *,
    candidatos: int = MINIMO_CANDIDATOS,
    busca_hibrida: bool = True,
    incluir_vizinhas: bool = False,
    diversificar_arquivos: bool = True,
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> list[ResultadoCaso]:
    casos = carregar_casos()
    colecao = abrir_colecao()
    manifesto = manifesto_compativel(modelo_embeddings)
    cliente = Client(host=OLLAMA_HOST)
    resultados: list[ResultadoCaso] = []
    for caso in casos:
        vetor = gerar_embedding_pergunta(cliente, caso["pergunta"], modelo_embeddings)
        if len(vetor) != manifesto.dimensao:
            raise ErroConsulta(
                f"Dimensão incompatível: consulta={len(vetor)}, índice={manifesto.dimensao}."
            )
        trechos, _ = recuperar_trechos(
            colecao,
            caso["pergunta"],
            vetor,
            TOP_K_AVALIACAO,
            candidatos=candidatos,
            disciplina=disciplina,
            busca_hibrida=busca_hibrida,
            incluir_vizinhas=incluir_vizinhas,
            diversificar_arquivos=diversificar_arquivos,
        )
        resultados.append(avaliar_caso(caso, trechos))
    return resultados


def _resumo_trecho(texto: str, limite: int = 180) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= limite else texto[: limite - 1].rstrip() + "…"


def imprimir_relatorio(
    resultados: Sequence[ResultadoCaso], disciplina: str | None = None
) -> None:
    print("Avaliação da recuperação híbrida")
    print(
        f"Casos: {len(resultados)} | top-k: {TOP_K_AVALIACAO} "
        f"| escopo: {disciplina or TODAS_DISCIPLINAS}\n"
    )
    for indice, resultado in enumerate(resultados, start=1):
        estado = "ACERTO" if resultado.acertou_pagina else "FALHA"
        print(f"Caso {indice}: {estado}")
        print(f"Pergunta: {resultado.pergunta}")
        print(f"Páginas esperadas: {resultado.paginas_esperadas}")
        print(f"Páginas recuperadas: {resultado.paginas_retornadas}")
        print(
            "Termos no contexto: "
            + (", ".join(f"✓ {termo}" for termo in resultado.termos_encontrados) or "nenhum")
        )
        if resultado.termos_ausentes:
            print("Termos ausentes: " + ", ".join(f"✗ {termo}" for termo in resultado.termos_ausentes))
        print(f"Observação: {resultado.observacao}")
        print("Trechos retornados:")
        for trecho in resultado.trechos:
            relevancia = trecho.relevancia or 0.0
            print(
                f"  - [{trecho.arquivo}, página do PDF {trecho.pagina}] "
                f"relevância={relevancia:.3f}: {_resumo_trecho(trecho.texto)}"
            )
        print()

    acertos = sum(resultado.acertou_pagina for resultado in resultados)
    taxa = taxa_acerto_recuperacao(resultados)
    casos_com_todos_termos = sum(not resultado.termos_ausentes for resultado in resultados)
    print("Resumo")
    print(f"Acertos de página: {acertos}/{len(resultados)}")
    print(f"Taxa de acerto da recuperação: {taxa:.1%}")
    print(f"Casos com todos os termos no contexto: {casos_com_todos_termos}/{len(resultados)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia a recuperação híbrida dos materiais indexados."
    )
    parser.add_argument(
        "--disciplina", help="restringe a avaliação a uma disciplina"
    )
    parser.add_argument("--candidatos", type=int, default=MINIMO_CANDIDATOS)
    parser.add_argument("--sem-busca-hibrida", action="store_true")
    parser.add_argument("--paginas-vizinhas", action="store_true")
    parser.add_argument("--sem-diversificacao", action="store_true")
    parser.add_argument("--modelo-embeddings", default=MODELO_EMBEDDINGS)
    parser.add_argument(
        "--geracao",
        action="store_true",
        help="avalia resposta, citações, idioma, recusas e sustentação factual",
    )
    parser.add_argument(
        "--comparar-compatibilidade",
        action="store_true",
        help="registra o modo anterior como linha de base antes da avaliação fundamentada",
    )
    args = parser.parse_args()
    disciplina = (
        None
        if not args.disciplina or args.disciplina == TODAS_DISCIPLINAS
        else args.disciplina
    )
    try:
        if args.geracao:
            from .generation_eval import (
                carregar_casos_geracao,
                executar_avaliacao_geracao,
                formatar_metrica_agregada,
                resultado_aprovado,
                resumo_metricas,
                salvar_linha_base,
            )

            casos_geracao = carregar_casos_geracao()

            def imprimir_resumo_geracao(titulo: str, resultados: Sequence) -> None:
                metricas = resumo_metricas(resultados)
                print(titulo)
                print("  Métricas determinísticas:")
                for chave, metrica in metricas["metricas_deterministicas"].items():
                    if chave in {"arquivo_correto", "pagina_correta", "fonte_correta"}:
                        continue
                    print(f"    {chave}: {formatar_metrica_agregada(metrica)}")
                rastreabilidade = metricas[
                    "metricas_rastreabilidade_deterministicas"
                ]
                print("  Rastreabilidade determinística de evidências:")
                print(
                    "    cobertura média afirmação→evidência: "
                    + (
                        f"{rastreabilidade['cobertura_media_evidencias_afirmacoes']:.1%}"
                        if rastreabilidade[
                            "cobertura_media_evidencias_afirmacoes"
                        ] is not None
                        else "não aplicável"
                    )
                )
                for chave in (
                    "tentativas_evidencia_inexistente",
                    "tentativas_trecho_inexistente",
                    "tentativas_mistura_arquivos",
                    "afirmacoes_publicadas_sem_evidencia",
                ):
                    print(f"    {chave}: {rastreabilidade[chave]}")
                print(
                    "    citações únicas: "
                    f"{rastreabilidade['quantidade_citacoes_unicas']} | "
                    "duplicatas removidas: "
                    f"{rastreabilidade['citacoes_duplicadas_removidas']}"
                )
                auxiliares = metricas["metricas_auxiliares_qwen"]
                print("  Métricas auxiliares pelo Qwen (não independentes):")
                print(
                    "    citacao_sustenta_afirmacao: "
                    + formatar_metrica_agregada(
                        auxiliares["citacao_sustenta_afirmacao"]
                    )
                )
                print(
                    "    afirmações inseguras publicadas: "
                    f"{auxiliares['afirmacoes_inseguras_publicadas']}"
                )
                print(f"    aviso: {auxiliares['aviso']}")

            if args.comparar_compatibilidade:
                print("Executando nova linha de base (modo Compatibilidade)...")
                anteriores = executar_avaliacao_geracao(
                    "compatibilidade", casos_geracao, salvar_resultado=False
                )
                caminho_base = salvar_linha_base(anteriores)
                imprimir_resumo_geracao("Linha de base corrigida:", anteriores)
                print(f"  relatório: {caminho_base.relative_to(RAIZ_PROJETO)}")
                print()
            print("Executando modo Fundamentado...")
            novos = executar_avaliacao_geracao("fundamentado", casos_geracao)
            imprimir_resumo_geracao("Resultado fundamentado:", novos)
            if novos.relatorio:
                print(f"  relatório: {novos.relatorio.relative_to(RAIZ_PROJETO)}")
            print("\nDetalhes por caso:")
            for indice, item in enumerate(novos, start=1):
                estado = resultado_aprovado(item)
                print(
                    f"{indice}. {'OK' if estado else 'ATENÇÃO'} | {item.pergunta}\n"
                    f"   páginas={list(item.paginas_retornadas)} | documento={item.documento}\n"
                    f"   página_recuperada={item.pagina_recuperada} "
                    f"fonte_recuperada={item.fonte_recuperada} "
                    f"página_citada={item.citacao_pagina_esperada} "
                    f"fonte_citada={item.citacao_fonte_esperada} "
                    f"conceitos={item.conceitos_presentes} "
                    f"citação_formal={item.citacao_formal_valida} "
                    f"citação_recuperada={item.citacao_recuperada} idioma={item.idioma_correto} "
                    f"recusa={item.recusa_correta} "
                    f"afirmações_obrigatórias={item.afirmacoes_obrigatorias_presentes} "
                    f"fontes_por_afirmação={item.fontes_aceitaveis_citadas} "
                    f"fórmulas={item.formulas_integras} "
                    f"omissões_críticas={item.omissoes_criticas}"
                )
                for diagnostico in item.diagnosticos_semanticos:
                    if diagnostico.estado == "reprovado":
                        print(
                            f"   - {diagnostico.id}: "
                            f"{diagnostico.motivo_deterministico}"
                        )
            return 0
        resultados = executar_avaliacao(
            disciplina,
            candidatos=max(MINIMO_CANDIDATOS, args.candidatos),
            busca_hibrida=not args.sem_busca_hibrida,
            incluir_vizinhas=args.paginas_vizinhas,
            diversificar_arquivos=not args.sem_diversificacao,
            modelo_embeddings=args.modelo_embeddings,
        )
        imprimir_relatorio(resultados, disciplina)
    except ErroConsulta as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
