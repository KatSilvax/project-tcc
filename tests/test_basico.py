import sys
from pathlib import Path

import joblib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from regras import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_FILES,
    colunas_faltantes,
    criar_alvo,
    politicas_recomendadas,
)


@pytest.mark.parametrize(
    ('resposta', 'esperado'),
    [
        ('nao nunca', 0),
        ('NAO NUNCA', 0),
        ('sim, algumas vezes', 1),
        ('prefiro não responder', 1),
        (None, 1),
    ],
)
def test_criar_alvo(resposta, esperado):
    assert criar_alvo(resposta) == esperado


def perfil(**alteracoes):
    base = {
        'renda_total': 'mais de 6 salarios minimos',
        'trabalha_atualmente': 'nao',
        'recebe_auxilio_da_insti': 'nao',
        'cidade': 'Jardim',
        'idade': '19 a 24 anos',
    }
    base.update(alteracoes)
    return base


def test_politicas_para_renda_baixa():
    politicas = politicas_recomendadas(perfil(renda_total='ate 1 salario minimo'))
    assert 'Auxílio permanência' in politicas
    assert 'Alimentação/refeição' in politicas


def test_politicas_para_quem_trabalha():
    politicas = politicas_recomendadas(
        perfil(trabalha_atualmente='sim, em periodo integral')
    )
    assert 'Auxílio permanência' in politicas
    assert 'Apoio pedagógico' in politicas


def test_politica_de_transporte_fora_de_jardim():
    politicas = politicas_recomendadas(perfil(cidade='Campo Grande'))
    assert 'Auxílio transporte' in politicas


@pytest.mark.parametrize('idade', ['15 a 18 anos', 'menos de 15 anos'])
def test_pe_de_meia_para_faixas_jovens(idade):
    assert 'Pé-de-Meia' in politicas_recomendadas(perfil(idade=idade))


def test_sem_pe_de_meia_para_faixa_adulta():
    assert 'Pé-de-Meia' not in politicas_recomendadas(
        perfil(idade='19 a 24 anos')
    )


def test_upload_com_todas_as_colunas_obrigatorias():
    assert colunas_faltantes(FEATURE_COLUMNS) == []


def test_upload_com_coluna_faltando():
    colunas = [coluna for coluna in FEATURE_COLUMNS if coluna != 'cidade']
    assert colunas_faltantes(colunas) == ['cidade']


def test_carregamento_dos_modelos():
    modelos_dir = PROJECT_ROOT / 'models'
    ausentes = [
        arquivo for arquivo in MODEL_FILES.values()
        if not (modelos_dir / arquivo).exists()
    ]
    if ausentes:
        pytest.skip(f'Modelos ausentes: {", ".join(ausentes)}')

    from model_wrapper import GBGridSearchWrapper
    sys.modules['__main__'].GBGridSearchWrapper = GBGridSearchWrapper

    for arquivo in MODEL_FILES.values():
        modelo = joblib.load(modelos_dir / arquivo)
        assert hasattr(modelo, 'predict')
        assert hasattr(modelo, 'predict_proba')
