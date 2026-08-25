from pathlib import Path

import pytest

from src.chat import remover_secao_fontes
from src.ui_utils import nome_pdf_seguro, salvar_pdf


def test_nome_pdf_remove_caminho_e_caracteres_invalidos():
    assert nome_pdf_seguro("../../Aula: controle?.PDF") == "Aula_ controle_.pdf"
    assert nome_pdf_seguro(r"C:\privado\CON.pdf") == "_CON.pdf"


def test_nome_pdf_rejeita_outra_extensao():
    with pytest.raises(ValueError, match="extensão .pdf"):
        nome_pdf_seguro("anotacoes.txt")


def test_salvar_pdf_valida_assinatura_e_nao_sobrescreve(tmp_path: Path):
    destino = salvar_pdf("aula.pdf", b"%PDF-1.7\nconteudo", tmp_path)
    assert destino.read_bytes().startswith(b"%PDF-")

    with pytest.raises(FileExistsError):
        salvar_pdf("aula.pdf", b"%PDF-1.7\nnovo", tmp_path)
    with pytest.raises(ValueError, match="assinatura PDF"):
        salvar_pdf("falso.pdf", b"nao e pdf", tmp_path)


def test_remover_secao_fontes_para_exibicao_separada():
    resposta = "Corpo da resposta.\n\nFontes\n- [aula.pdf, p. 2]"
    assert remover_secao_fontes(resposta) == "Corpo da resposta."
