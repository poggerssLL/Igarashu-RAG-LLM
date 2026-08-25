"""Registro local e operações seguras de gerenciamento de matérias."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import PASTA_CHROMA, PASTA_DADOS, PASTA_DOCUMENTOS


ARQUIVO_DISCIPLINAS = PASTA_DADOS / "disciplinas.json"
SEM_DISCIPLINA = "Sem disciplina"
TODAS_DISCIPLINAS = "Todas as disciplinas"
MATERIAS_PADRAO = (
    "Sinais e Sistemas",
    "Controle Linear",
    "Eletrônica",
    "Automação Industrial",
)
NOMES_RESERVADOS_WINDOWS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{numero}" for numero in range(1, 10)),
    *(f"lpt{numero}" for numero in range(1, 10)),
}


class ErroDisciplina(RuntimeError):
    """Erro esperado em uma operação de gerenciamento de matérias."""


@dataclass(frozen=True)
class Materia:
    id: str
    nome: str
    pasta: str
    descricao: str
    criada_em: str


@dataclass(frozen=True)
class EstatisticaMateria:
    materia: Materia
    pdfs: int
    trechos: int


def validar_nome_materia(nome: str) -> str:
    nome = nome.strip()
    if not nome:
        raise ErroDisciplina("O nome da matéria não pode estar vazio.")
    if nome in {".", ".."} or ".." in nome:
        raise ErroDisciplina("O nome da matéria não pode conter navegação de pastas.")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', nome):
        raise ErroDisciplina("O nome contém caracteres proibidos pelo Windows.")
    if nome.endswith((" ", ".")):
        raise ErroDisciplina("O nome não pode terminar com espaço ou ponto.")
    if nome.casefold() in NOMES_RESERVADOS_WINDOWS:
        raise ErroDisciplina("Esse nome é reservado pelo Windows.")
    if len(nome) > 100:
        raise ErroDisciplina("O nome da matéria deve ter no máximo 100 caracteres.")
    if nome.casefold() in {SEM_DISCIPLINA.casefold(), TODAS_DISCIPLINAS.casefold()}:
        raise ErroDisciplina("Esse nome é reservado pelo sistema.")
    return nome


def _pasta_segura(nome: str, raiz_documentos: Path = PASTA_DOCUMENTOS) -> Path:
    nome = validar_nome_materia(nome)
    raiz = raiz_documentos.resolve()
    destino = (raiz / nome).resolve()
    if destino.parent != raiz:
        raise ErroDisciplina("A pasta da matéria deve ficar diretamente dentro de Documentos/.")
    return destino


def carregar_materias(caminho: Path = ARQUIVO_DISCIPLINAS) -> list[Materia]:
    if not caminho.exists():
        return []
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ErroDisciplina(f"Não foi possível ler o registro de matérias: {erro}") from erro
    itens = dados.get("disciplinas", []) if isinstance(dados, dict) else []
    try:
        materias = [Materia(**item) for item in itens]
    except (TypeError, ValueError) as erro:
        raise ErroDisciplina("O registro de matérias possui formato inválido.") from erro
    return sorted(materias, key=lambda materia: materia.nome.casefold())


def salvar_materias(materias: list[Materia], caminho: Path = ARQUIVO_DISCIPLINAS) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "versao": 1,
        "disciplinas": [asdict(materia) for materia in sorted(materias, key=lambda item: item.nome.casefold())],
    }
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")
    temporario.replace(caminho)


def _agora_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def criar_materia(
    nome: str,
    descricao: str = "",
    *,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    arquivo_registro: Path = ARQUIVO_DISCIPLINAS,
    gerar_id: Callable[[], object] = uuid.uuid4,
    criada_em: str | None = None,
) -> Materia:
    nome = validar_nome_materia(nome)
    materias = carregar_materias(arquivo_registro)
    if any(materia.nome.casefold() == nome.casefold() for materia in materias):
        raise ErroDisciplina(f"A matéria '{nome}' já está cadastrada.")
    pasta = _pasta_segura(nome, raiz_documentos)
    if pasta.exists():
        raise ErroDisciplina(f"A pasta '{nome}' já existe em Documentos/.")

    pasta.mkdir(parents=True)
    materia = Materia(
        id=str(gerar_id()),
        nome=nome,
        pasta=nome,
        descricao=descricao.strip(),
        criada_em=criada_em or _agora_iso(),
    )
    try:
        salvar_materias([*materias, materia], arquivo_registro)
    except Exception:
        pasta.rmdir()
        raise
    return materia


def garantir_materias_padrao(
    *,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    arquivo_registro: Path = ARQUIVO_DISCIPLINAS,
) -> list[Materia]:
    materias = carregar_materias(arquivo_registro)
    nomes = {materia.nome.casefold() for materia in materias}
    alterado = False
    for nome in MATERIAS_PADRAO:
        pasta = _pasta_segura(nome, raiz_documentos)
        pasta.mkdir(parents=True, exist_ok=True)
        if nome.casefold() not in nomes:
            materias.append(
                Materia(str(uuid.uuid4()), nome, nome, "", _agora_iso())
            )
            nomes.add(nome.casefold())
            alterado = True
    if alterado or not arquivo_registro.exists():
        salvar_materias(materias, arquivo_registro)
    return sorted(materias, key=lambda materia: materia.nome.casefold())


def obter_materia(materia_id: str, caminho: Path = ARQUIVO_DISCIPLINAS) -> Materia:
    for materia in carregar_materias(caminho):
        if materia.id == materia_id:
            return materia
    raise ErroDisciplina("Matéria não encontrada.")


def editar_materia(
    materia_id: str,
    novo_nome: str,
    descricao: str,
    *,
    confirmar_renomeacao: bool = False,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    arquivo_registro: Path = ARQUIVO_DISCIPLINAS,
) -> Materia:
    novo_nome = validar_nome_materia(novo_nome)
    materias = carregar_materias(arquivo_registro)
    atual = next((materia for materia in materias if materia.id == materia_id), None)
    if atual is None:
        raise ErroDisciplina("Matéria não encontrada.")
    if any(
        materia.id != materia_id and materia.nome.casefold() == novo_nome.casefold()
        for materia in materias
    ):
        raise ErroDisciplina(f"A matéria '{novo_nome}' já está cadastrada.")

    renomeando = atual.nome != novo_nome
    if renomeando and not confirmar_renomeacao:
        raise ErroDisciplina("Confirme explicitamente a renomeação da matéria e da pasta.")

    origem = _pasta_segura(atual.pasta, raiz_documentos)
    destino = _pasta_segura(novo_nome, raiz_documentos)
    moveu = False
    if renomeando:
        if destino.exists():
            raise ErroDisciplina(f"A pasta de destino '{novo_nome}' já existe.")
        origem.mkdir(parents=True, exist_ok=True)
        origem.rename(destino)
        moveu = True

    atualizada = Materia(
        id=atual.id,
        nome=novo_nome,
        pasta=novo_nome,
        descricao=descricao.strip(),
        criada_em=atual.criada_em,
    )
    novas = [atualizada if materia.id == materia_id else materia for materia in materias]
    try:
        salvar_materias(novas, arquivo_registro)
    except Exception:
        if moveu and destino.exists() and not origem.exists():
            destino.rename(origem)
        raise
    return atualizada


def excluir_materia(
    materia_id: str,
    *,
    confirmar: bool = False,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    arquivo_registro: Path = ARQUIVO_DISCIPLINAS,
) -> None:
    if not confirmar:
        raise ErroDisciplina("Confirme explicitamente a exclusão da matéria vazia.")
    materias = carregar_materias(arquivo_registro)
    materia = next((item for item in materias if item.id == materia_id), None)
    if materia is None:
        raise ErroDisciplina("Matéria não encontrada.")
    pasta = _pasta_segura(materia.pasta, raiz_documentos)
    if pasta.exists() and any(pasta.iterdir()):
        raise ErroDisciplina(
            "A matéria possui arquivos e não pode ser excluída. "
            "Mova ou remova os arquivos manualmente primeiro."
        )
    if pasta.exists():
        pasta.rmdir()
    try:
        salvar_materias([item for item in materias if item.id != materia_id], arquivo_registro)
    except Exception:
        pasta.mkdir(parents=True, exist_ok=True)
        raise


def mover_pdf(
    caminho_relativo: str,
    materia_destino_id: str | None,
    *,
    confirmar: bool = False,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    arquivo_registro: Path = ARQUIVO_DISCIPLINAS,
) -> Path:
    if not confirmar:
        raise ErroDisciplina("Confirme explicitamente a movimentação do PDF.")
    raiz = raiz_documentos.resolve()
    relativo = Path(caminho_relativo.replace("\\", "/"))
    origem = (raiz / relativo).resolve()
    try:
        origem.relative_to(raiz)
    except ValueError as erro:
        raise ErroDisciplina("O arquivo deve estar dentro de Documentos/.") from erro
    if not origem.is_file() or origem.suffix.casefold() != ".pdf":
        raise ErroDisciplina("O PDF de origem não foi encontrado.")

    if materia_destino_id is None:
        pasta_destino = raiz
    else:
        materia = obter_materia(materia_destino_id, arquivo_registro)
        pasta_destino = _pasta_segura(materia.pasta, raiz_documentos)
        pasta_destino.mkdir(parents=True, exist_ok=True)
    destino = (pasta_destino / origem.name).resolve()
    try:
        destino.relative_to(raiz)
    except ValueError as erro:
        raise ErroDisciplina("O destino deve estar dentro de Documentos/.") from erro
    if destino == origem:
        raise ErroDisciplina("O PDF já está nessa matéria.")
    if destino.exists():
        raise ErroDisciplina(f"Já existe um arquivo chamado '{destino.name}' no destino.")
    origem.rename(destino)
    return destino.relative_to(raiz)


def _contar_trechos_por_disciplina(colecao: object | None) -> dict[str, int]:
    if colecao is None:
        return {}
    try:
        metadados = colecao.get(include=["metadatas"]).get("metadatas") or []
    except Exception:
        return {}
    contagem: dict[str, int] = {}
    for metadado in metadados:
        disciplina = str((metadado or {}).get("disciplina") or SEM_DISCIPLINA)
        contagem[disciplina] = contagem.get(disciplina, 0) + 1
    return contagem


def estatisticas_materias(
    materias: list[Materia],
    *,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    colecao: object | None = None,
) -> list[EstatisticaMateria]:
    trechos = _contar_trechos_por_disciplina(colecao)
    resultado: list[EstatisticaMateria] = []
    for materia in materias:
        pasta = _pasta_segura(materia.pasta, raiz_documentos)
        pdfs = sum(1 for arquivo in pasta.rglob("*") if arquivo.is_file() and arquivo.suffix.casefold() == ".pdf") if pasta.exists() else 0
        resultado.append(EstatisticaMateria(materia, pdfs, trechos.get(materia.nome, 0)))
    return resultado


def resumo_biblioteca(
    materias: list[Materia],
    *,
    raiz_documentos: Path = PASTA_DOCUMENTOS,
    colecao: object | None = None,
) -> dict[str, int]:
    pdfs = sum(
        1
        for arquivo in raiz_documentos.rglob("*")
        if arquivo.is_file() and arquivo.suffix.casefold() == ".pdf"
    ) if raiz_documentos.exists() else 0
    try:
        trechos = int(colecao.count()) if colecao is not None else 0
    except Exception:
        trechos = 0
    return {"materias": len(materias), "pdfs": pdfs, "trechos": trechos}


def disciplinas_com_trechos(colecao: object | None) -> list[str]:
    contagem = _contar_trechos_por_disciplina(colecao)
    return sorted((nome for nome, total in contagem.items() if total > 0), key=str.casefold)
