"""Funções sem dependência do Streamlit usadas pela interface local."""

from __future__ import annotations

import re
from pathlib import Path


NOMES_RESERVADOS_WINDOWS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{numero}" for numero in range(1, 10)),
    *(f"lpt{numero}" for numero in range(1, 10)),
}


def nome_pdf_seguro(nome_original: str) -> str:
    """Remove caminhos e caracteres inválidos, mantendo somente um nome PDF seguro."""
    nome_base = nome_original.replace("\\", "/").rsplit("/", 1)[-1].strip()
    caminho = Path(nome_base)
    if caminho.suffix.casefold() != ".pdf":
        raise ValueError("O arquivo deve ter extensão .pdf.")

    radical = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", caminho.stem).strip(" .")
    if not radical:
        radical = "documento"
    if radical.casefold() in NOMES_RESERVADOS_WINDOWS:
        radical = f"_{radical}"
    radical = radical[:180].rstrip(" .") or "documento"
    return f"{radical}.pdf"


def salvar_pdf(
    nome_original: str,
    conteudo: bytes,
    pasta_documentos: Path,
    *,
    sobrescrever: bool = False,
) -> Path:
    """Valida e salva um PDF sem permitir fuga da pasta de documentos."""
    if b"%PDF-" not in conteudo[:1024]:
        raise ValueError("O conteúdo enviado não possui uma assinatura PDF válida.")
    pasta_documentos.mkdir(parents=True, exist_ok=True)
    destino = pasta_documentos / nome_pdf_seguro(nome_original)
    if destino.exists() and not sobrescrever:
        raise FileExistsError(f"O arquivo '{destino.name}' já existe.")
    destino.write_bytes(conteudo)
    return destino
