import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import dividir_texto, similaridade


def test_dividir_texto_preserva_conteudo():
    texto = "controle automatico " * 100
    trechos = dividir_texto(texto, tamanho=100, sobreposicao=20)
    assert len(trechos) > 1
    assert all(trecho for trecho in trechos)


def test_similaridade_cosseno():
    assert similaridade([1, 0], [1, 0]) == 1.0
    assert similaridade([1, 0], [0, 1]) == 0.0
