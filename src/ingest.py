"""Ingestão local de PDFs no ChromaDB usando embeddings do Ollama."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
import pymupdf
from ollama import Client, ResponseError

try:
    from .disciplinas import SEM_DISCIPLINA
except ImportError:
    from disciplinas import SEM_DISCIPLINA

try:
    from .config import (
        ARQUIVO_MANIFESTO_INDICE,
        MODELO_EMBEDDINGS,
        NOME_COLECAO,
        OLLAMA_HOST,
        PASTA_CHROMA,
        PASTA_DOCUMENTOS,
        SOBREPOSICAO_TRECHO,
        TAMANHO_LOTE_EMBEDDINGS,
        TAMANHO_TRECHO,
        preparar_pastas,
    )
except ImportError:  # Permite também executar: python src\ingest.py
    from config import (
        ARQUIVO_MANIFESTO_INDICE,
        MODELO_EMBEDDINGS,
        NOME_COLECAO,
        OLLAMA_HOST,
        PASTA_CHROMA,
        PASTA_DOCUMENTOS,
        SOBREPOSICAO_TRECHO,
        TAMANHO_LOTE_EMBEDDINGS,
        TAMANHO_TRECHO,
        preparar_pastas,
    )

try:
    from .index_manifest import (
        ErroManifesto,
        carregar_manifesto,
        criar_manifesto,
        salvar_manifesto,
        validar_compatibilidade,
    )
except ImportError:
    from index_manifest import (
        ErroManifesto,
        carregar_manifesto,
        criar_manifesto,
        salvar_manifesto,
        validar_compatibilidade,
    )


class ErroIngestao(RuntimeError):
    """Erro esperado, apresentado ao usuário sem traceback."""


@dataclass
class Relatorio:
    pdfs_encontrados: int = 0
    paginas_lidas: int = 0
    trechos_indexados: int = 0
    trechos_ignorados: int = 0
    indice_recriado: bool = False
    modelo_embeddings: str = MODELO_EMBEDDINGS
    dimensao_embeddings: int = 0


@dataclass(frozen=True)
class Trecho:
    id: str
    texto: str
    arquivo: str
    nome_arquivo: str
    pagina: int
    indice: int
    disciplina: str


def dividir_em_trechos(
    texto: str,
    tamanho: int = TAMANHO_TRECHO,
    sobreposicao: int = SOBREPOSICAO_TRECHO,
) -> list[str]:
    """Divide texto por caracteres, procurando terminar em espaço quando possível."""
    if tamanho <= 0:
        raise ValueError("O tamanho do trecho deve ser maior que zero.")
    if sobreposicao < 0 or sobreposicao >= tamanho:
        raise ValueError("A sobreposição deve ser maior ou igual a zero e menor que o tamanho.")

    texto_limpo = " ".join(texto.split())
    if not texto_limpo:
        return []

    trechos: list[str] = []
    inicio = 0
    while inicio < len(texto_limpo):
        fim = min(inicio + tamanho, len(texto_limpo))
        if fim < len(texto_limpo):
            quebra = texto_limpo.rfind(" ", inicio + tamanho // 2, fim)
            if quebra > inicio:
                fim = quebra

        trecho = texto_limpo[inicio:fim].strip()
        if trecho:
            trechos.append(trecho)

        if fim >= len(texto_limpo):
            break
        inicio = max(fim - sobreposicao, inicio + 1)

    return trechos


def gerar_id_estavel(caminho_relativo: str | Path, pagina: int, indice_trecho: int) -> str:
    """Gera o mesmo ID para a mesma posição lógica do trecho em todas as execuções."""
    caminho_normalizado = str(caminho_relativo).replace("\\", "/").casefold()
    origem = f"{caminho_normalizado}|pagina={pagina}|trecho={indice_trecho}"
    return hashlib.sha256(origem.encode("utf-8")).hexdigest()


def encontrar_pdfs(pasta: Path) -> list[Path]:
    return sorted(
        (arquivo for arquivo in pasta.rglob("*") if arquivo.is_file() and arquivo.suffix.lower() == ".pdf"),
        key=lambda arquivo: arquivo.relative_to(pasta).as_posix().casefold(),
    )


def inferir_disciplina(pdf: Path, raiz_documentos: Path = PASTA_DOCUMENTOS) -> str:
    """Usa a primeira subpasta após Documentos/ como nome da disciplina."""
    relativo = pdf.resolve().relative_to(raiz_documentos.resolve())
    return relativo.parts[0] if len(relativo.parts) > 1 else SEM_DISCIPLINA


def _nome_modelo(modelo: object) -> str:
    if isinstance(modelo, dict):
        return str(modelo.get("model") or modelo.get("name") or "")
    return str(getattr(modelo, "model", None) or getattr(modelo, "name", ""))


def verificar_ollama(cliente: Client, modelo_embeddings: str = MODELO_EMBEDDINGS) -> None:
    """Confirma que o serviço responde e que o modelo de embeddings está instalado."""
    try:
        resposta = cliente.list()
    except Exception as erro:
        raise ErroIngestao(
            f"Ollama indisponível em {OLLAMA_HOST}. Inicie o Ollama e tente novamente."
        ) from erro

    modelos = getattr(resposta, "models", None)
    if modelos is None and isinstance(resposta, dict):
        modelos = resposta.get("models", [])
    instalados = {_nome_modelo(modelo) for modelo in (modelos or [])}
    nome_base = modelo_embeddings.split(":", 1)[0]
    encontrado = any(nome == modelo_embeddings or nome.split(":", 1)[0] == nome_base for nome in instalados)
    if not encontrado:
        raise ErroIngestao(
            f"Modelo de embeddings ausente: {modelo_embeddings}. "
            f"Instale-o com: ollama pull {modelo_embeddings}"
        )


def extrair_trechos(pdf: Path, raiz_documentos: Path, relatorio: Relatorio) -> list[Trecho]:
    caminho_relativo = pdf.relative_to(raiz_documentos).as_posix()
    disciplina = inferir_disciplina(pdf, raiz_documentos)
    trechos_pdf: list[Trecho] = []
    try:
        documento = pymupdf.open(pdf)
    except Exception as erro:
        raise ErroIngestao(f"Não foi possível abrir o PDF '{caminho_relativo}': {erro}") from erro

    try:
        for numero_pagina, pagina in enumerate(documento, start=1):
            relatorio.paginas_lidas += 1
            partes = dividir_em_trechos(pagina.get_text("text"))
            if not partes:
                relatorio.trechos_ignorados += 1
                continue
            for indice, texto in enumerate(partes):
                trechos_pdf.append(
                    Trecho(
                        id=gerar_id_estavel(caminho_relativo, numero_pagina, indice),
                        texto=texto,
                        arquivo=caminho_relativo,
                        nome_arquivo=pdf.name,
                        pagina=numero_pagina,
                        indice=indice,
                        disciplina=disciplina,
                    )
                )
    finally:
        documento.close()

    if not trechos_pdf:
        print(
            f"Aviso: '{caminho_relativo}' não possui texto extraível; "
            "pode ser necessário aplicar OCR.",
            file=sys.stderr,
        )
    return trechos_pdf


def _lotes(itens: Sequence[Trecho], tamanho: int) -> list[Sequence[Trecho]]:
    return [itens[inicio : inicio + tamanho] for inicio in range(0, len(itens), tamanho)]


def indexar_trechos(
    cliente: Client,
    colecao: object,
    trechos: Sequence[Trecho],
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> tuple[int, int]:
    indexados = 0
    dimensao = 0
    for lote in _lotes(trechos, TAMANHO_LOTE_EMBEDDINGS):
        textos = [trecho.texto for trecho in lote]
        try:
            resposta = cliente.embed(model=modelo_embeddings, input=textos)
        except ResponseError as erro:
            if erro.status_code == 404:
                raise ErroIngestao(f"Modelo de embeddings ausente: {modelo_embeddings}.") from erro
            raise ErroIngestao(f"O Ollama recusou a geração de embeddings: {erro.error}") from erro
        except Exception as erro:
            raise ErroIngestao(
                f"Ollama ficou indisponível durante a geração de embeddings em {OLLAMA_HOST}."
            ) from erro

        embeddings = resposta.embeddings
        if len(embeddings) != len(lote):
            raise ErroIngestao("O Ollama retornou uma quantidade inesperada de embeddings.")
        dimensao_lote = len(embeddings[0]) if embeddings else 0
        if dimensao and dimensao_lote != dimensao:
            raise ErroIngestao("O modelo retornou embeddings com dimensões inconsistentes.")
        dimensao = dimensao_lote

        colecao.upsert(
            ids=[trecho.id for trecho in lote],
            documents=textos,
            embeddings=[list(vetor) for vetor in embeddings],
            metadatas=[
                {
                    "arquivo": trecho.arquivo,
                    "nome_arquivo": trecho.nome_arquivo,
                    "pagina": trecho.pagina,
                    "indice_trecho": trecho.indice,
                    "disciplina": trecho.disciplina,
                    "caminho_relativo": trecho.arquivo,
                }
                for trecho in lote
            ],
        )
        indexados += len(lote)
    return indexados, dimensao


def preparar_colecao(cliente_chroma: object, recriar_indice: bool = False) -> object:
    """Obtém a coleção e, quando solicitado, remove somente o índice anterior."""
    if recriar_indice:
        existentes = {colecao.name for colecao in cliente_chroma.list_collections()}
        if NOME_COLECAO in existentes:
            cliente_chroma.delete_collection(name=NOME_COLECAO)
    return cliente_chroma.get_or_create_collection(name=NOME_COLECAO)


def executar_ingestao(
    *,
    recriar_indice: bool = False,
    confirmar_reindexacao: bool = False,
    modelo_embeddings: str = MODELO_EMBEDDINGS,
) -> Relatorio:
    if recriar_indice and not confirmar_reindexacao:
        raise ErroIngestao(
            "A recriação do índice exige confirmação explícita. Nenhum PDF será apagado."
        )
    preparar_pastas()
    pdfs = encontrar_pdfs(PASTA_DOCUMENTOS)
    if not pdfs:
        raise ErroIngestao(
            f"Nenhum PDF encontrado em '{PASTA_DOCUMENTOS}'. "
            "Adicione ao menos um arquivo .pdf e tente novamente."
        )

    relatorio = Relatorio(
        pdfs_encontrados=len(pdfs),
        indice_recriado=recriar_indice,
        modelo_embeddings=modelo_embeddings,
    )
    cliente_ollama = Client(host=OLLAMA_HOST)
    verificar_ollama(cliente_ollama, modelo_embeddings)

    todos_trechos: list[Trecho] = []
    for pdf in pdfs:
        print(f"Lendo: {pdf.relative_to(PASTA_DOCUMENTOS).as_posix()}")
        todos_trechos.extend(extrair_trechos(pdf, PASTA_DOCUMENTOS, relatorio))

    if not todos_trechos:
        raise ErroIngestao(
            "Os PDFs foram lidos, mas nenhum texto extraível foi encontrado. "
            "Se forem documentos digitalizados, aplique OCR antes da ingestão."
        )

    cliente_chroma = chromadb.PersistentClient(path=PASTA_CHROMA)
    existentes = {item.name for item in cliente_chroma.list_collections()}
    indice_tem_dados = False
    if NOME_COLECAO in existentes:
        indice_tem_dados = cliente_chroma.get_collection(NOME_COLECAO).count() > 0
    if not recriar_indice:
        try:
            validar_compatibilidade(
                modelo_embeddings,
                carregar_manifesto(),
                indice_tem_dados=indice_tem_dados,
            )
        except ErroManifesto as erro:
            raise ErroIngestao(str(erro)) from erro
    colecao = preparar_colecao(cliente_chroma, recriar_indice)
    if recriar_indice and ARQUIVO_MANIFESTO_INDICE.exists():
        ARQUIVO_MANIFESTO_INDICE.unlink()
    quantidade, dimensao = indexar_trechos(
        cliente_ollama, colecao, todos_trechos, modelo_embeddings
    )
    relatorio.trechos_indexados = quantidade
    relatorio.dimensao_embeddings = dimensao
    salvar_manifesto(criar_manifesto(modelo_embeddings, dimensao, colecao.count()))
    return relatorio


def imprimir_relatorio(relatorio: Relatorio) -> None:
    print("\nIngestão concluída")
    print(f"PDFs encontrados: {relatorio.pdfs_encontrados}")
    print(f"Páginas lidas: {relatorio.paginas_lidas}")
    print(f"Trechos indexados: {relatorio.trechos_indexados}")
    print(f"Trechos ignorados: {relatorio.trechos_ignorados}")
    print(f"Modelo de embeddings: {relatorio.modelo_embeddings}")
    print(f"Dimensão dos embeddings: {relatorio.dimensao_embeddings}")
    print(f"Banco vetorial: {PASTA_CHROMA}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Indexa os PDFs de Documentos/ no ChromaDB usando o Ollama local."
    )
    parser.add_argument(
        "--reindexar-tudo",
        action="store_true",
        help="recria a coleção vetorial antes de indexar todos os PDFs",
    )
    parser.add_argument(
        "--modelo-embeddings",
        default=MODELO_EMBEDDINGS,
        help="modelo local de embeddings (ex.: nomic-embed-text ou embeddinggemma)",
    )
    parser.add_argument(
        "--confirmar",
        action="store_true",
        help="confirma explicitamente a recriação do índice vetorial",
    )
    args = parser.parse_args()
    try:
        imprimir_relatorio(
            executar_ingestao(
                recriar_indice=args.reindexar_tudo,
                confirmar_reindexacao=args.confirmar,
                modelo_embeddings=args.modelo_embeddings,
            )
        )
    except ErroIngestao as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
