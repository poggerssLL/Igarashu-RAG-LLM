"""RAG local e mínimo para PDFs, usando apenas a API local do Ollama."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTOS = ROOT / "Documentos"
INDICE = ROOT / "Dados" / "indice.json"
OLLAMA = "http://127.0.0.1:11434"
MODELO_CHAT = "qwen2.5:3b"
MODELO_EMBED = "nomic-embed-text"


def chamar_ollama(caminho: str, dados: dict) -> dict:
    """Envia uma requisição à API do Ollama que roda no próprio computador."""
    corpo = json.dumps(dados).encode("utf-8")
    requisicao = Request(
        f"{OLLAMA}{caminho}", corpo, {"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(requisicao, timeout=120) as resposta:
            return json.loads(resposta.read().decode("utf-8"))
    except URLError as erro:
        raise RuntimeError("Não foi possível acessar o Ollama em 127.0.0.1:11434.") from erro


def embedding(texto: str) -> list[float]:
    resposta = chamar_ollama("/api/embed", {"model": MODELO_EMBED, "input": texto})
    return resposta["embeddings"][0]


def dividir_texto(texto: str, tamanho: int = 900, sobreposicao: int = 150) -> list[str]:
    """Divide texto preservando uma pequena sobreposição entre trechos."""
    texto = " ".join(texto.split())
    if not texto:
        return []
    trechos = []
    inicio = 0
    while inicio < len(texto):
        fim = min(inicio + tamanho, len(texto))
        if fim < len(texto):
            quebra = texto.rfind(" ", inicio, fim)
            if quebra > inicio:
                fim = quebra
        trechos.append(texto[inicio:fim])
        if fim == len(texto):
            break
        inicio = fim - sobreposicao
    return trechos


def indexar() -> None:
    try:
        from pypdf import PdfReader
    except ImportError as erro:
        raise RuntimeError("Instale a dependência: python -m pip install -r requirements.txt") from erro

    pdfs = sorted(DOCUMENTOS.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError("Nenhum PDF encontrado em Documentos/. Coloque materiais lá e tente novamente.")

    registros = []
    for pdf in pdfs:
        print(f"Lendo: {pdf.relative_to(DOCUMENTOS)}")
        leitor = PdfReader(str(pdf))
        for numero, pagina in enumerate(leitor.pages, start=1):
            for trecho in dividir_texto(pagina.extract_text() or ""):
                registros.append(
                    {
                        "arquivo": str(pdf.relative_to(DOCUMENTOS)),
                        "pagina": numero,
                        "texto": trecho,
                        "vetor": embedding(trecho),
                    }
                )

    INDICE.parent.mkdir(exist_ok=True)
    INDICE.write_text(json.dumps(registros, ensure_ascii=False), encoding="utf-8")
    print(f"Índice criado: {len(registros)} trechos em {INDICE.relative_to(ROOT)}")


def similaridade(a: list[float], b: list[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    return produto / (norma_a * norma_b) if norma_a and norma_b else 0.0


def perguntar(pergunta: str, quantidade: int = 4) -> None:
    if not INDICE.exists():
        raise RuntimeError("Índice inexistente. Execute: python src/rag.py indexar")
    registros = json.loads(INDICE.read_text(encoding="utf-8"))
    if not registros:
        raise RuntimeError("O índice está vazio.")

    vetor_pergunta = embedding(pergunta)
    melhores = sorted(registros, key=lambda r: similaridade(vetor_pergunta, r["vetor"]), reverse=True)[:quantidade]
    contexto = "\n\n".join(
        f"Fonte [{r['arquivo']}, p. {r['pagina']}]:\n{r['texto']}" for r in melhores
    )
    prompt = f"""Responda em português, de forma didática, usando exclusivamente o contexto.
Se o contexto não bastar, diga isso claramente. Inclua citações no formato [arquivo, p. N]
para cada afirmação importante. Não invente fontes.

Contexto:
{contexto}

Pergunta: {pergunta}
"""
    resposta = chamar_ollama("/api/generate", {"model": MODELO_CHAT, "prompt": prompt, "stream": False})
    print("\n" + resposta["response"].strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG local de PDFs com Ollama")
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    subcomandos.add_parser("indexar", help="lê os PDFs e cria o índice local")
    consulta = subcomandos.add_parser("perguntar", help="responde com base no índice")
    consulta.add_argument("pergunta")
    consulta.add_argument("--trechos", type=int, default=4, help="quantidade de trechos recuperados")
    args = parser.parse_args()
    if args.comando == "indexar":
        indexar()
    else:
        perguntar(args.pergunta, args.trechos)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        raise SystemExit(1)
