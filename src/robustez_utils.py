"""Utilitários compartilhados para avaliação estatística do projeto."""

import importlib.util
import os
from functools import lru_cache

import numpy as np
import pandas as pd


FEATURES = [
    'curso',
    'periodo',
    'idade',
    'cidade',
    'renda_total',
    'trabalha_atualmente',
    'recebe_auxilio_da_insti',
]
TARGET = 'risco_evasao'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Respostas_evasao_limpo.csv')


def criar_alvo(df):
    """Cria o alvo binário usado pelo benchmark, sem alterar o DataFrame original."""
    resultado = df.copy()
    resultado[TARGET] = resultado['ja_pensou_em_desistir'].apply(
        lambda value: 0 if isinstance(value, str) and 'nao nunca' in value.lower() else 1
    )
    return resultado


def carregar_base_real(path=REAL_DATA_PATH):
    """Carrega a base real e retorna somente entradas e alvo do benchmark."""
    df = criar_alvo(pd.read_csv(path))
    df = df[FEATURES + [TARGET]].copy()
    df['periodo'] = pd.to_numeric(df['periodo'], errors='coerce').fillna(1).astype(int)
    return df


@lru_cache(maxsize=1)
def carregar_modulo_benchmark():
    """Importa as definições existentes sem executar o fluxo principal do benchmark."""
    path = os.path.join(PROJECT_ROOT, 'scripts', '02_benchmark_modelos_tcc.py')
    spec = importlib.util.spec_from_file_location('evasao_zero_benchmark_definicoes', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Não foi possível carregar o benchmark em {path}.')
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def construir_modelos(catboost_grid_reduzido=True):
    """Retorna as fábricas dos três modelos definidos no benchmark original.

    O grid reduzido do CatBoost é aplicado apenas no experimento de robustez para
    tornar dez repetições executáveis; a classe e a implementação continuam sendo
    as do benchmark original.
    """
    benchmark = carregar_modulo_benchmark()
    factories = dict(benchmark.MODELOS)

    if catboost_grid_reduzido:
        factory_original = factories['CatBoost (SOTA)']

        def factory_catboost_robusto():
            modelo = factory_original()
            if hasattr(modelo, '_param_grid'):
                modelo._param_grid = {
                    'depth': [3],
                    'l2_leaf_reg': [5],
                    'iterations': [100, 200],
                    'learning_rate': [0.05, 0.1],
                }
            return modelo

        factories['CatBoost (SOTA)'] = factory_catboost_robusto

    return factories


def gerar_sinteticos(train_df, n=1000, random_state=42):
    """Gera sintéticos usando somente o DataFrame de treino recebido."""
    benchmark = carregar_modulo_benchmark()
    np.random.seed(random_state)
    sinteticos = benchmark.gerar_sinteticos(train_df, n=n)
    return sinteticos


def versoes_bibliotecas():
    """Retorna versões das bibliotecas principais usadas na avaliação."""
    import joblib
    import sklearn
    import scipy

    import sdv

    return {
        'python': '.'.join(str(part) for part in __import__('sys').version_info[:3]),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
        'scipy': scipy.__version__,
        'joblib': joblib.__version__,
        'sdv': getattr(sdv, '__version__', 'não identificada'),
    }
