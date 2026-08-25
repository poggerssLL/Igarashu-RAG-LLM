from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.disciplinas import (
    ErroDisciplina,
    carregar_materias,
    criar_materia,
    editar_materia,
    estatisticas_materias,
    excluir_materia,
    mover_pdf,
    resumo_biblioteca,
    validar_nome_materia,
)


def criar_teste(tmp_path: Path, nome: str = "Controle Linear"):
    documentos = tmp_path / "Documentos"
    registro = tmp_path / "Dados" / "disciplinas.json"
    materia = criar_materia(
        nome,
        "Descrição",
        raiz_documentos=documentos,
        arquivo_registro=registro,
        gerar_id=lambda: "id-1",
        criada_em="2026-01-01T00:00:00+00:00",
    )
    return materia, documentos, registro


@pytest.mark.parametrize(
    "nome",
    ["", "../Fora", "Controle/Linear", "Aula?", "CON", "Final."],
)
def test_nome_de_materia_rejeita_valores_inseguros(nome):
    with pytest.raises(ErroDisciplina):
        validar_nome_materia(nome)


def test_criar_e_editar_descricao_da_materia(tmp_path):
    materia, documentos, registro = criar_teste(tmp_path)

    assert (documentos / "Controle Linear").is_dir()
    assert carregar_materias(registro)[0].descricao == "Descrição"

    atualizada = editar_materia(
        materia.id,
        materia.nome,
        "Nova descrição",
        raiz_documentos=documentos,
        arquivo_registro=registro,
    )
    assert atualizada.descricao == "Nova descrição"


def test_renomear_exige_confirmacao_e_move_a_pasta(tmp_path):
    materia, documentos, registro = criar_teste(tmp_path)
    pdf = documentos / materia.pasta / "aula.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    with pytest.raises(ErroDisciplina, match="Confirme"):
        editar_materia(
            materia.id,
            "Controle Moderno",
            "",
            raiz_documentos=documentos,
            arquivo_registro=registro,
        )

    atualizada = editar_materia(
        materia.id,
        "Controle Moderno",
        "",
        confirmar_renomeacao=True,
        raiz_documentos=documentos,
        arquivo_registro=registro,
    )
    assert atualizada.pasta == "Controle Moderno"
    assert (documentos / "Controle Moderno" / "aula.pdf").exists()
    assert not (documentos / "Controle Linear").exists()


def test_excluir_bloqueia_materia_com_arquivos_e_remove_apenas_vazia(tmp_path):
    materia, documentos, registro = criar_teste(tmp_path)
    pdf = documentos / materia.pasta / "aula.pdf"
    pdf.write_bytes(b"%PDF-1.7")

    with pytest.raises(ErroDisciplina, match="possui arquivos"):
        excluir_materia(
            materia.id,
            confirmar=True,
            raiz_documentos=documentos,
            arquivo_registro=registro,
        )

    pdf.unlink()
    excluir_materia(
        materia.id,
        confirmar=True,
        raiz_documentos=documentos,
        arquivo_registro=registro,
    )
    assert carregar_materias(registro) == []
    assert not (documentos / materia.pasta).exists()


def test_mover_pdf_exige_confirmacao_e_preserva_arquivo(tmp_path):
    origem, documentos, registro = criar_teste(tmp_path, "Origem")
    destino = criar_materia(
        "Destino",
        raiz_documentos=documentos,
        arquivo_registro=registro,
        gerar_id=lambda: "id-2",
    )
    arquivo = documentos / origem.pasta / "aula.pdf"
    arquivo.write_bytes(b"%PDF-1.7\nconteudo")

    with pytest.raises(ErroDisciplina, match="Confirme"):
        mover_pdf(
            "Origem/aula.pdf",
            destino.id,
            raiz_documentos=documentos,
            arquivo_registro=registro,
        )

    relativo = mover_pdf(
        "Origem/aula.pdf",
        destino.id,
        confirmar=True,
        raiz_documentos=documentos,
        arquivo_registro=registro,
    )
    assert relativo.as_posix() == "Destino/aula.pdf"
    assert (documentos / relativo).read_bytes().endswith(b"conteudo")


def test_estatisticas_contam_pdfs_e_trechos_por_materia(tmp_path):
    materia, documentos, registro = criar_teste(tmp_path)
    (documentos / materia.pasta / "a.pdf").write_bytes(b"%PDF-1.7")
    (documentos / materia.pasta / "b.PDF").write_bytes(b"%PDF-1.7")
    colecao = Mock()
    colecao.get.return_value = {
        "metadatas": [
            {"disciplina": materia.nome},
            {"disciplina": materia.nome},
            {"disciplina": "Outra"},
        ]
    }
    colecao.count.return_value = 3

    estatistica = estatisticas_materias(
        carregar_materias(registro),
        raiz_documentos=documentos,
        colecao=colecao,
    )[0]
    resumo = resumo_biblioteca(
        [materia], raiz_documentos=documentos, colecao=colecao
    )

    assert (estatistica.pdfs, estatistica.trechos) == (2, 2)
    assert resumo == {"materias": 1, "pdfs": 2, "trechos": 3}
