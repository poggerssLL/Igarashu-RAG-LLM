import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from src.ingest import (
    ErroIngestao,
    dividir_em_trechos,
    executar_ingestao,
    gerar_id_estavel,
    inferir_disciplina,
    preparar_colecao,
)


def test_divisao_ignora_texto_vazio():
    assert dividir_em_trechos("  \n\t  ") == []


def test_divisao_respeita_tamanho_e_cria_sobreposicao():
    texto = " ".join(f"palavra{i}" for i in range(300))
    trechos = dividir_em_trechos(texto, tamanho=120, sobreposicao=20)

    assert len(trechos) > 1
    assert all(trecho.strip() for trecho in trechos)
    assert all(len(trecho) <= 120 for trecho in trechos)
    assert any(palavra in trechos[1] for palavra in trechos[0].split()[-3:])


def test_divisao_rejeita_parametros_invalidos():
    with pytest.raises(ValueError):
        dividir_em_trechos("texto", tamanho=100, sobreposicao=100)


def test_id_e_estavel_e_normaliza_separadores_e_caixa():
    primeiro = gerar_id_estavel("Aulas\\Controle.PDF", pagina=3, indice_trecho=2)
    segundo = gerar_id_estavel("aulas/controle.pdf", pagina=3, indice_trecho=2)

    assert primeiro == segundo
    assert len(primeiro) == 64


def test_id_muda_com_a_origem_do_trecho():
    base = gerar_id_estavel("controle.pdf", pagina=1, indice_trecho=0)

    assert base != gerar_id_estavel("controle.pdf", pagina=2, indice_trecho=0)
    assert base != gerar_id_estavel("controle.pdf", pagina=1, indice_trecho=1)
    assert base != gerar_id_estavel("outro.pdf", pagina=1, indice_trecho=0)


def test_inferencia_de_disciplina_pelo_primeiro_diretorio(tmp_path):
    documentos = tmp_path / "Documentos"
    dentro = documentos / "Sinais e Sistemas" / "Aulas" / "aula.pdf"
    raiz = documentos / "solto.pdf"

    assert inferir_disciplina(dentro, documentos) == "Sinais e Sistemas"
    assert inferir_disciplina(raiz, documentos) == "Sem disciplina"


def test_preparar_colecao_recria_somente_quando_solicitado():
    cliente = Mock()
    cliente.list_collections.return_value = [SimpleNamespace(name="materiais_estudo")]
    cliente.get_or_create_collection.return_value = "colecao"

    assert preparar_colecao(cliente, recriar_indice=False) == "colecao"
    cliente.delete_collection.assert_not_called()

    preparar_colecao(cliente, recriar_indice=True)
    cliente.delete_collection.assert_called_once_with(name="materiais_estudo")


def test_reindexacao_exige_confirmacao_antes_de_qualquer_operacao():
    with pytest.raises(ErroIngestao, match="confirmação explícita"):
        executar_ingestao(recriar_indice=True, confirmar_reindexacao=False)
