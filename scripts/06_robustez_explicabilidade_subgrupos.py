"""Explicabilidade, subgrupos e rastreabilidade da robustez estatística."""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from robustez_utils import (  # noqa: E402
    FEATURES,
    TARGET,
    carregar_base_real,
    carregar_modulo_benchmark,
    construir_modelos,
    gerar_sinteticos,
    versoes_bibliotecas,
)

N_REPETICOES = 5
N_SPLITS = 5
RANDOM_STATE = 42
MINIMO_SUBGRUPO = 20
OUTPUTS = [
    'resultados_robustez_importancia.csv',
    'resultados_robustez_subgrupos.csv',
    'resultados_robustez_rastreabilidade.csv',
]
FIGURE = os.path.join('figures', 'tcc', 'rob_importancia.png')


def preparar_cenario(X_train, y_train, n_sinteticos, indice_fold):
    if n_sinteticos == 0:
        return X_train.copy(), y_train.copy()
    treino = X_train.copy()
    treino[TARGET] = y_train
    sinteticos = gerar_sinteticos(
        treino,
        n=n_sinteticos,
        random_state=RANDOM_STATE + indice_fold,
    )
    return (
        pd.concat([sinteticos[FEATURES], X_train], ignore_index=True),
        np.concatenate([sinteticos[TARGET].to_numpy(), y_train]),
    )


def predizer(modelo, X):
    return np.asarray(modelo.predict(X)).astype(int), np.asarray(
        modelo.predict_proba(X)
    )[:, 1]


def pontuacao_auc(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])


def shap_importancia(modelo, X_fit, X_val):
    """Calcula SHAP nas sete variáveis originais, sem depender do encoder interno."""
    amostra_original = X_val.iloc[: min(32, len(X_val))].copy()
    categoricas = [coluna for coluna in FEATURES if coluna != 'periodo']
    codigos = {}
    for coluna in categoricas:
        valores = X_fit[coluna].fillna('<NA>').astype(str).unique()
        codigos[coluna] = {valor: indice for indice, valor in enumerate(valores)}

    def codificar(frame):
        resultado = pd.DataFrame(index=frame.index)
        for coluna in FEATURES:
            if coluna in codigos:
                resultado[coluna] = (
                    frame[coluna].fillna('<NA>').astype(str)
                    .map(codigos[coluna]).fillna(-1).astype(float)
                )
            else:
                resultado[coluna] = pd.to_numeric(
                    frame[coluna], errors='coerce'
                ).fillna(0.0)
        return resultado

    amostra = codificar(amostra_original)

    def predicao(dados):
        frame = pd.DataFrame(dados, columns=FEATURES)
        for coluna, mapa in codigos.items():
            valores = list(mapa)
            indices = np.clip(np.rint(frame[coluna]).astype(int), 0, len(valores) - 1)
            frame[coluna] = [valores[indice] for indice in indices]
        return modelo.predict_proba(frame)[:, 1]

    try:
        explicador = shap.Explainer(
            predicao,
            amostra,
            feature_names=FEATURES,
            algorithm='permutation',
        )
        valores = explicador(
            amostra,
            max_evals=2 * len(FEATURES) + 1,
        ).values
        if valores.ndim == 3:
            valores = valores[:, :, 0]
        return np.abs(valores).mean(axis=0), ''
    except Exception as erro:
        return np.full(len(FEATURES), np.nan), str(erro)


def grupos_subpopulacao(df):
    grupos = {
        'curso': df['curso'].fillna('<NA>').astype(str),
        'faixa_renda': df['renda_total'].fillna('<NA>').astype(str),
        'trabalha_atualmente': df['trabalha_atualmente'].fillna('<NA>').astype(str),
    }
    return grupos


def implementar_rastreabilidade(factories):
    benchmark = carregar_modulo_benchmark()
    registros = []
    for nome, factory in factories.items():
        implementacao = 'não identificada'
        if nome == 'TabPFN / HistGB (SOTA)':
            modelo = factory()
            texto = repr(modelo)
            implementacao = (
                'TabPFN real'
                if 'TabPFNClassifier' in texto
                else 'fallback HistGradientBoosting'
            )
        elif nome == 'CatBoost (SOTA)':
            implementacao = 'CatBoostWrapper'
        elif nome == 'GradientBoosting (Baseline)':
            implementacao = 'GBGridSearchWrapper'
        registros.append({
            'modelo': nome,
            'implementacao': implementacao,
            'modulo_benchmark': benchmark.__file__,
            'seed': RANDOM_STATE,
        })
    versoes = versoes_bibliotecas()
    try:
        import catboost
        versoes['catboost'] = catboost.__version__
    except ImportError:
        versoes['catboost'] = 'não instalado'
    try:
        versoes['shap'] = shap.__version__
    except AttributeError:
        versoes['shap'] = 'não identificada'
    try:
        import tabpfn
        versoes['tabpfn'] = getattr(tabpfn, '__version__', 'instalado')
    except ImportError:
        versoes['tabpfn'] = 'não instalado'
    for registro in registros:
        registro.update({
            'python': versoes['python'],
            'numpy': versoes['numpy'],
            'pandas': versoes['pandas'],
            'sklearn': versoes['scikit-learn'],
            'catboost': versoes['catboost'],
            'shap': versoes['shap'],
            'sdv': versoes['sdv'],
            'tabpfn': versoes['tabpfn'],
        })
    return pd.DataFrame(registros)


def validar_saidas_livres():
    caminhos = OUTPUTS + [FIGURE]
    existentes = [
        caminho for caminho in caminhos
        if os.path.exists(os.path.join(PROJECT_ROOT, caminho))
    ]
    if existentes:
        raise FileExistsError(
            'Saídas já existentes; nada será sobrescrito: '
            + ', '.join(existentes)
        )


def executar(repeticoes):
    validar_saidas_livres()
    df = carregar_base_real()
    factories = construir_modelos(catboost_grid_reduzido=True)
    modelos = list(factories)
    grupos = grupos_subpopulacao(df)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=repeticoes,
        random_state=RANDOM_STATE,
    )
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    importancia = []
    subgrupos = []
    rankings = []

    for indice, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        repeticao = (indice - 1) // N_SPLITS + 1
        fold = (indice - 1) % N_SPLITS + 1
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]
        for cenario, n_sinteticos in (('A_Real', 0), ('B_SDV1000', 1000)):
            X_fit, y_fit = preparar_cenario(
                X_train, y_train, n_sinteticos, indice
            )
            for nome in modelos:
                modelo = factories[nome]()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    modelo.fit(X_fit, y_fit)
                pred, proba = predizer(modelo, X_val)
                perm = permutation_importance(
                    modelo,
                    X_val,
                    y_val,
                    scoring=pontuacao_auc,
                    n_repeats=5,
                    random_state=RANDOM_STATE,
                )
                shap_values, shap_erro = shap_importancia(modelo, X_fit, X_val)
                ranking = pd.Series(shap_values, index=FEATURES).rank(
                    ascending=False, method='average'
                )
                for variavel, perm_mean, perm_std, shap_mean in zip(
                    FEATURES,
                    perm.importances_mean,
                    perm.importances_std,
                    shap_values,
                ):
                    importancia.append({
                        'repeticao': repeticao,
                        'fold': fold,
                        'modelo': nome,
                        'cenario': cenario,
                        'variavel': variavel,
                        'permutation_media': perm_mean,
                        'permutation_desvio': perm_std,
                        'shap_media_abs': shap_mean,
                        'shap_erro': shap_erro,
                    })
                for variavel, posicao in ranking.items():
                    rankings.append({
                        'repeticao': repeticao,
                        'fold': fold,
                        'modelo': nome,
                        'cenario': cenario,
                        'variavel': variavel,
                        'ranking_shap': posicao,
                    })
                for nome_grupo, categorias in grupos.items():
                    for categoria in categorias.iloc[val_idx].unique():
                        mascara = categorias.iloc[val_idx].to_numpy() == categoria
                        y_grupo = y_val[mascara]
                        proba_grupo = proba[mascara]
                        confiavel = (
                            len(y_grupo) >= MINIMO_SUBGRUPO
                            and len(np.unique(y_grupo)) == 2
                        )
                        subgrupos.append({
                            'repeticao': repeticao,
                            'fold': fold,
                            'modelo': nome,
                            'cenario': cenario,
                            'variavel_grupo': nome_grupo,
                            'categoria': categoria,
                            'n': len(y_grupo),
                            'confiavel': confiavel,
                            'roc_auc': (
                                roc_auc_score(y_grupo, proba_grupo)
                                if len(np.unique(y_grupo)) == 2 else np.nan
                            ),
                            'recall': recall_score(
                                y_grupo,
                                pred[mascara],
                                zero_division=0,
                            ),
                        })

    df_importancia = pd.DataFrame(importancia)
    df_rankings = pd.DataFrame(rankings)
    ranking_rows = []
    correlacoes = []
    for (modelo, cenario), grupo in df_rankings.groupby(
        ['modelo', 'cenario']
    ):
        matriz = grupo.pivot_table(
            index='fold',
            columns='variavel',
            values='ranking_shap',
            aggfunc='mean',
        )
        valores = []
        for indice_a in range(len(matriz)):
            for indice_b in range(indice_a + 1, len(matriz)):
                pares = matriz.iloc[[indice_a, indice_b]].dropna(axis=1)
                if pares.shape[1] > 1:
                    valores.append(spearmanr(pares.iloc[0], pares.iloc[1]).statistic)
        correlacoes.append({
            'modelo': modelo,
            'cenario': cenario,
            'ranking_spearman_media': np.mean(valores) if valores else np.nan,
            'ranking_spearman_desvio': np.std(valores, ddof=1) if len(valores) > 1 else np.nan,
        })
    for chaves, grupo in df_rankings.groupby(
        ['modelo', 'cenario', 'variavel']
    ):
        ranking_rows.append({
            'modelo': chaves[0],
            'cenario': chaves[1],
            'variavel': chaves[2],
            'ranking_shap_media': grupo['ranking_shap'].mean(),
            'ranking_shap_desvio': grupo['ranking_shap'].std(ddof=1),
        })
    df_importancia = df_importancia.merge(
        pd.DataFrame(ranking_rows),
        on=['modelo', 'cenario', 'variavel'],
        how='left',
    )
    df_importancia = df_importancia.merge(
        pd.DataFrame(correlacoes),
        on=['modelo', 'cenario'],
        how='left',
    )
    df_importancia.to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[0]), index=False
    )
    pd.DataFrame(subgrupos).to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[1]), index=False
    )
    implementar_rastreabilidade(factories).to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[2]), index=False
    )
    gerar_figura_importancia(df_importancia)
    print(f'Concluído com {repeticoes} repetição(ões).')


def gerar_figura_importancia(df):
    resumo = df.groupby(['modelo', 'variavel'], as_index=False)[
        'shap_media_abs'
    ].agg(['mean', 'std']).reset_index()
    fig, eixo = plt.subplots(figsize=(12, 7))
    for modelo, grupo in resumo.groupby('modelo'):
        grupo = grupo.sort_values('mean')
        eixo.errorbar(
            grupo['mean'],
            grupo['variavel'],
            xerr=grupo['std'].fillna(0),
            fmt='o-',
            capsize=3,
            label=modelo,
        )
    eixo.set_xlabel('Importância SHAP absoluta média ± desvio')
    eixo.set_ylabel('Variável')
    eixo.set_title('Estabilidade da importância das variáveis')
    eixo.legend(fontsize=8, loc='best')
    eixo.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, FIGURE), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeticoes', type=int, default=N_REPETICOES)
    args = parser.parse_args()
    if args.repeticoes < 1:
        parser.error('--repeticoes deve ser pelo menos 1.')
    executar(args.repeticoes)


if __name__ == '__main__':
    main()
