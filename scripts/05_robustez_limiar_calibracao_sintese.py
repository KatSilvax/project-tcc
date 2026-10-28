"""Robustez de limiar, calibração e quantidade de dados sintéticos.

Todos os ajustes deste experimento acontecem dentro do treino de cada fold.
Os arquivos de saída são novos e o script recusa sobrescrever resultados.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from robustez_utils import (  # noqa: E402
    FEATURES,
    TARGET,
    carregar_base_real,
    construir_modelos,
    gerar_sinteticos,
    versoes_bibliotecas,
)

N_REPETICOES = 5
N_SPLITS = 5
RANDOM_STATE = 42
RECALL_MINIMO = 0.70
ESTRATEGIA_LIMIAR = 'f1'
N_SINTETICOS = [0, 250, 500, 1000, 2000]
CALIBRACAO_CENARIOS = {'A_Real': 0, 'B_SDV1000': 1000}
OUTPUTS = [
    'resultados_robustez_limiar.csv',
    'resultados_robustez_calibracao.csv',
    'resultados_robustez_sintese.csv',
]
FIGURES = [
    os.path.join('figures', 'tcc', 'rob_calibracao.png'),
    os.path.join('figures', 'tcc', 'rob_sintese_metricas.png'),
]


def ajustar_modelo(factory, X, y):
    """Cria e ajusta uma nova instância de modelo somente no treino."""
    modelo = factory()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        modelo.fit(X, y)
    return modelo


class FittedClassifierAdapter(ClassifierMixin, BaseEstimator):
    """Adapta um modelo legado já ajustado ao contrato do scikit-learn."""

    def __init__(self, fitted_model):
        self.fitted_model = fitted_model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.fitted_model.predict(X)

    def predict_proba(self, X):
        return self.fitted_model.predict_proba(X)


def reduzir_grid_para_validacao(factory):
    """Reduz somente a busca interna para tornar a validação reproduzível."""
    def factory_reduzida():
        modelo = factory()
        if hasattr(modelo, '_param_grid'):
            modelo._param_grid = {
                'depth': [3],
                'l2_leaf_reg': [5],
                'iterations': [100],
                'learning_rate': [0.1],
            }
        elif hasattr(modelo, 'param_grid'):
            modelo.param_grid = {
                chave: [valores[0]]
                for chave, valores in modelo.param_grid.items()
            }
        return modelo

    return factory_reduzida


def metricas_limiar(y_true, proba, limiar):
    pred = (proba >= limiar).astype(int)
    return {
        'f1_macro': f1_score(y_true, pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'precision': precision_score(y_true, pred, zero_division=0),
    }


def escolher_limiar(factory, X_train, y_train, estrategia):
    """Escolhe o limiar com predições out-of-fold exclusivamente do treino."""
    inner_cv = RepeatedStratifiedKFold(
        n_splits=3, n_repeats=1, random_state=RANDOM_STATE
    )
    probabilidades = np.zeros(len(y_train), dtype=float)
    vistos = np.zeros(len(y_train), dtype=bool)
    for inner_train, inner_val in inner_cv.split(X_train, y_train):
        modelo = ajustar_modelo(
            factory,
            X_train.iloc[inner_train],
            y_train[inner_train],
        )
        probabilidades[inner_val] = modelo.predict_proba(
            X_train.iloc[inner_val]
        )[:, 1]
        vistos[inner_val] = True

    y_interno = y_train[vistos]
    proba_interna = probabilidades[vistos]
    candidatos = np.unique(np.clip(
        np.r_[0.0, proba_interna, 0.5, 1.0], 0.0, 1.0
    ))
    avaliados = [
        (float(limiar), metricas_limiar(y_interno, proba_interna, limiar))
        for limiar in candidatos
    ]
    if estrategia == 'recall_minimo':
        elegiveis = [
            item for item in avaliados
            if item[1]['recall'] >= RECALL_MINIMO
        ]
        if elegiveis:
            return max(
                elegiveis,
                key=lambda item: (item[1]['f1_macro'], -item[0]),
            )[0]
    return max(
        avaliados,
        key=lambda item: (item[1]['f1_macro'], item[1]['recall'], -item[0]),
    )[0]


def preparar_cenario(X_train, y_train, n_sinteticos, indice_fold):
    if n_sinteticos == 0:
        return X_train.copy(), y_train.copy()
    treino = X_train.copy()
    treino[TARGET] = y_train
    sinteticos = gerar_sinteticos(
        treino,
        n=n_sinteticos,
        random_state=RANDOM_STATE + indice_fold + n_sinteticos,
    )
    X_fit = pd.concat([sinteticos[FEATURES], X_train], ignore_index=True)
    y_fit = np.concatenate([sinteticos[TARGET].to_numpy(), y_train])
    return X_fit, y_fit


def validar_saidas_livres():
    existentes = [
        nome for nome in OUTPUTS
        if os.path.exists(os.path.join(PROJECT_ROOT, nome))
    ]
    figuras_existentes = [
        caminho for caminho in FIGURES
        if os.path.exists(os.path.join(PROJECT_ROOT, caminho))
    ]
    if existentes or figuras_existentes:
        itens = existentes + figuras_existentes
        raise FileExistsError(
            'Saídas ou figuras já existentes; nada será sobrescrito: '
            + ', '.join(itens)
        )


def executar(repeticoes):
    validar_saidas_livres()
    df = carregar_base_real()
    factories = {
        nome: reduzir_grid_para_validacao(factory)
        for nome, factory in construir_modelos(
            catboost_grid_reduzido=True
        ).items()
    }
    modelos = list(factories)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=repeticoes,
        random_state=RANDOM_STATE,
    )
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    limiar_rows = []
    calibracao_rows = []
    sintese_rows = []
    curvas = []

    for indice, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        repeticao = (indice - 1) // N_SPLITS + 1
        fold = (indice - 1) % N_SPLITS + 1
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]

        # Limiar e calibração são avaliados no real e no SDV1000.
        for cenario, n_sinteticos in CALIBRACAO_CENARIOS.items():
            X_fit, y_fit = preparar_cenario(
                X_train, y_train, n_sinteticos, indice
            )
            for nome in modelos:
                factory = factories[nome]
                modelo = ajustar_modelo(factory, X_fit, y_fit)
                proba = modelo.predict_proba(X_val)[:, 1]
                limiar_otimo = escolher_limiar(
                    factory, X_fit, y_fit, ESTRATEGIA_LIMIAR
                )
                for nome_limiar, limiar in (
                    ('padrao_0_5', 0.5),
                    ('otimizado_treino', limiar_otimo),
                ):
                    limiar_rows.append({
                        'repeticao': repeticao,
                        'fold': fold,
                        'modelo': nome,
                        'cenario': cenario,
                        'estrategia': nome_limiar,
                        'limiar': limiar,
                        **metricas_limiar(y_val, proba, limiar),
                    })

                for metodo in ('sigmoid', 'isotonic'):
                    calibrado = CalibratedClassifierCV(
                        estimator=FrozenEstimator(
                            FittedClassifierAdapter(modelo)
                        ),
                        method=metodo,
                        cv=None,
                        n_jobs=1,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        calibrado.fit(X_fit, y_fit)
                    proba_calibrada = calibrado.predict_proba(X_val)[:, 1]
                    brier_antes = brier_score_loss(y_val, proba)
                    brier_depois = brier_score_loss(y_val, proba_calibrada)
                    calibracao_rows.extend([
                        {
                            'repeticao': repeticao,
                            'fold': fold,
                            'modelo': nome,
                            'cenario': cenario,
                            'metodo': metodo,
                            'etapa': 'antes',
                            'brier': brier_antes,
                        },
                        {
                            'repeticao': repeticao,
                            'fold': fold,
                            'modelo': nome,
                            'cenario': cenario,
                            'metodo': metodo,
                            'etapa': 'depois',
                            'brier': brier_depois,
                        },
                    ])
                    for etapa, probabilidades in (
                        ('antes', proba),
                        ('depois', proba_calibrada),
                    ):
                        fracao, media = calibration_curve(
                            y_val, probabilidades, n_bins=5, strategy='quantile'
                        )
                        for numero, (media_predita, fracao_positiva) in enumerate(
                            zip(media, fracao), start=1
                        ):
                            curvas.append({
                                'repeticao': repeticao,
                                'fold': fold,
                                'modelo': nome,
                                'cenario': cenario,
                                'metodo': metodo,
                                'etapa': etapa,
                                'bin': numero,
                                'probabilidade_media': media_predita,
                                'proporcao_positiva': fracao_positiva,
                            })

        # Sensibilidade: a validação é sempre o mesmo fold real intacto.
        for n_sinteticos in N_SINTETICOS:
            X_fit, y_fit = preparar_cenario(
                X_train, y_train, n_sinteticos, indice
            )
            for nome in modelos:
                modelo = ajustar_modelo(factories[nome], X_fit, y_fit)
                proba = modelo.predict_proba(X_val)[:, 1]
                pred = (proba >= 0.5).astype(int)
                sintese_rows.append({
                    'repeticao': repeticao,
                    'fold': fold,
                    'modelo': nome,
                    'n_sinteticos': n_sinteticos,
                    'roc_auc': roc_auc_score(y_val, proba),
                    'recall': recall_score(y_val, pred, zero_division=0),
                })

    df_limiar = pd.DataFrame(limiar_rows)
    df_calibracao = pd.DataFrame(calibracao_rows)
    df_sintese = pd.DataFrame(sintese_rows)
    df_curvas = pd.DataFrame(curvas)
    df_limiar.to_csv(os.path.join(PROJECT_ROOT, OUTPUTS[0]), index=False)
    df_calibracao.to_csv(os.path.join(PROJECT_ROOT, OUTPUTS[1]), index=False)
    df_sintese.to_csv(os.path.join(PROJECT_ROOT, OUTPUTS[2]), index=False)
    gerar_figura_calibracao(df_curvas)
    gerar_figura_sintese(df_sintese)
    print(f'Concluído com {repeticoes} repetição(ões).')
    print(f'Versões: {versoes_bibliotecas()}')


def gerar_figura_calibracao(df_curvas):
    fig, eixo = plt.subplots(figsize=(10, 7))
    for (modelo, cenario, metodo, etapa), grupo in df_curvas.groupby(
        ['modelo', 'cenario', 'metodo', 'etapa']
    ):
        grupo = grupo.groupby('bin', as_index=False).mean(numeric_only=True)
        eixo.plot(
            grupo['probabilidade_media'],
            grupo['proporcao_positiva'],
            marker='o',
            label=f'{modelo} | {cenario} | {metodo} | {etapa}',
        )
    eixo.plot([0, 1], [0, 1], '--', color='gray', label='perfeita')
    eixo.set_xlabel('Probabilidade média prevista')
    eixo.set_ylabel('Fração positiva observada')
    eixo.set_title('Curvas de calibração antes e depois')
    eixo.legend(fontsize=7, loc='best')
    eixo.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, FIGURES[0]), dpi=200)
    plt.close(fig)


def gerar_figura_sintese(df_sintese):
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    for eixo, metrica, titulo in zip(
        eixos,
        ('roc_auc', 'recall'),
        ('ROC-AUC', 'Recall'),
    ):
        resumo = df_sintese.groupby(
            ['modelo', 'n_sinteticos'], as_index=False
        )[metrica].agg(['mean', 'std']).reset_index()
        for modelo, grupo in resumo.groupby('modelo'):
            eixo.errorbar(
                grupo['n_sinteticos'],
                grupo['mean'],
                yerr=grupo['std'].fillna(0),
                marker='o',
                capsize=3,
                label=modelo,
            )
        eixo.set_xlabel('Linhas sintéticas por fold')
        eixo.set_ylabel(titulo)
        eixo.set_title(f'{titulo} x quantidade de sintéticos')
        eixo.set_xticks(N_SINTETICOS)
        eixo.grid(alpha=0.25)
    eixos[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, FIGURES[1]), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--repeticoes',
        type=int,
        default=N_REPETICOES,
        help=f'Número de repetições do CV (padrão: {N_REPETICOES}).',
    )
    args = parser.parse_args()
    if args.repeticoes < 1:
        parser.error('--repeticoes deve ser pelo menos 1.')
    executar(args.repeticoes)


if __name__ == '__main__':
    main()
