"""Avaliação estatística repetida dos modelos de evasão.

Este script cria arquivos novos e recusa sobrescrever qualquer saída existente.
O cenário B mantém o protocolo sem vazamento: SDV recebe apenas o treino de
cada repetição/fold, e a validação permanece na partição real intacta.
"""

import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

N_REPETICOES = 10
N_SPLITS = 5
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
METRICAS_IC = ['roc_auc', 'f1_macro', 'recall', 'precision', 'brier', 'bss']
METRICAS_TESTE = ['roc_auc', 'f1_macro', 'recall', 'precision', 'brier']
OUTPUTS = [
    'resultados_robustez_base.csv',
    'resultados_robustez_basica.csv',
    'resultados_robustez_testes.csv',
    'resultados_robustez_confusao.csv',
]


def construir_baselines():
    """Cria baselines e regressão logística com o mesmo contrato dos modelos."""
    categorias = [
        'curso',
        'idade',
        'cidade',
        'renda_total',
        'trabalha_atualmente',
        'recebe_auxilio_da_insti',
    ]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorias),
    ], remainder='passthrough')
    def factory_logistic():
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver='lbfgs',
                random_state=RANDOM_STATE,
            )),
        ])
    return {
        'Dummy (prior)': lambda: DummyClassifier(
            strategy='prior',
            random_state=RANDOM_STATE,
        ),
        'Dummy (stratified)': lambda: DummyClassifier(
            strategy='stratified',
            random_state=RANDOM_STATE,
        ),
        'LogisticRegression': factory_logistic,
    }


def descrever_base(df):
    """Retorna descrição longa da base em formato tabular."""
    rows = []
    rows.extend([
        {'secao': 'resumo', 'variavel': 'n_linhas', 'categoria': '', 'valor': len(df), 'contagem': ''},
        {'secao': 'resumo', 'variavel': 'proporcao_classe_0', 'categoria': '', 'valor': (df[TARGET] == 0).mean(), 'contagem': int((df[TARGET] == 0).sum())},
        {'secao': 'resumo', 'variavel': 'proporcao_classe_1', 'categoria': '', 'valor': (df[TARGET] == 1).mean(), 'contagem': int((df[TARGET] == 1).sum())},
    ])
    for coluna in FEATURES:
        nulos = int(df[coluna].isna().sum())
        rows.append({'secao': 'nulos', 'variavel': coluna, 'categoria': '', 'valor': nulos / len(df), 'contagem': nulos})
        distribuicao = df[coluna].value_counts(dropna=False)
        for categoria, contagem in distribuicao.items():
            rows.append({
                'secao': 'distribuicao',
                'variavel': coluna,
                'categoria': '<NA>' if pd.isna(categoria) else str(categoria),
                'valor': contagem / len(df),
                'contagem': int(contagem),
            })
    return pd.DataFrame(rows)


def calcular_metricas(y_true, pred, proba, prevalencia):
    """Calcula métricas discriminação, classificação, calibração e skill."""
    brier = brier_score_loss(y_true, proba)
    brier_baseline = brier_score_loss(y_true, np.full(len(y_true), prevalencia))
    return {
        'roc_auc': roc_auc_score(y_true, proba),
        'pr_auc': average_precision_score(y_true, proba),
        'f1_macro': f1_score(y_true, pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'precision': precision_score(y_true, pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, pred),
        'mcc': matthews_corrcoef(y_true, pred),
        'brier': brier,
        'bss': 1 - (brier / brier_baseline) if brier_baseline else np.nan,
    }


def bootstrap_ic(values, seed):
    """IC 95% bootstrap da média dos valores obtidos nos folds."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    amostras = rng.choice(clean, size=(N_BOOTSTRAP, len(clean)), replace=True)
    medias = amostras.mean(axis=1)
    return float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))


def holm_adjust(p_values):
    """Aplica correção de Holm a uma sequência de p-valores."""
    p_values = np.asarray(p_values, dtype=float)
    ordem = np.argsort(p_values)
    ajustados = np.empty(len(p_values), dtype=float)
    maior = 0.0
    for pos, indice in enumerate(ordem):
        ajustado = min(1.0, (len(p_values) - pos) * p_values[indice])
        maior = max(maior, ajustado)
        ajustados[indice] = maior
    return ajustados


def validar_saida_livre():
    existentes = [
        nome for nome in OUTPUTS
        if os.path.exists(os.path.join(PROJECT_ROOT, nome))
    ]
    if existentes:
        raise FileExistsError(
            'Saídas já existentes; para não sobrescrever resultados: '
            + ', '.join(existentes)
        )


def executar(repeticoes):
    validar_saida_livre()
    df = carregar_base_real()
    df_descricao = descrever_base(df)
    df_descricao.to_csv(os.path.join(PROJECT_ROOT, OUTPUTS[0]), index=False)

    factories = {}
    factories.update(construir_baselines())
    factories.update(construir_modelos(catboost_grid_reduzido=True))
    modelos = list(factories)
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=repeticoes,
        random_state=RANDOM_STATE,
    )
    X = df[FEATURES]
    y = df[TARGET].to_numpy()
    registros = []
    confusoes = []
    fold_metricas = []

    for indice, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        repeticao = (indice - 1) // N_SPLITS + 1
        fold = (indice - 1) % N_SPLITS + 1
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]

        cenarios = {'A_Real': (X_train, y_train)}
        treino_com_alvo = X_train.copy()
        treino_com_alvo[TARGET] = y_train
        sinteticos = gerar_sinteticos(
            treino_com_alvo,
            n=1000,
            random_state=RANDOM_STATE + indice,
        )
        cenarios['B_SDV1000'] = (
            pd.concat([sinteticos[FEATURES], X_train], ignore_index=True),
            np.concatenate([sinteticos[TARGET].to_numpy(), y_train]),
        )

        for cenario, (X_fit, y_fit) in cenarios.items():
            prevalencia = float(y_fit.mean())
            for nome in modelos:
                modelo = factories[nome]()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    modelo.fit(X_fit, y_fit)
                pred = np.asarray(modelo.predict(X_val)).astype(int)
                proba = np.asarray(modelo.predict_proba(X_val))[:, 1]
                metricas = calcular_metricas(y_val, pred, proba, prevalencia)
                fold_metricas.append({
                    'repeticao': repeticao,
                    'fold': fold,
                    'modelo': nome,
                    'cenario': cenario,
                    **metricas,
                })
                tn, fp, fn, tp = confusion_matrix(
                    y_val, pred, labels=[0, 1]
                ).ravel()
                confusoes.append({
                    'repeticao': repeticao,
                    'fold': fold,
                    'modelo': nome,
                    'cenario': cenario,
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp),
                    'n_validacao': len(y_val),
                })

    df_folds = pd.DataFrame(fold_metricas)
    resultados = []
    for grupo_indice, ((modelo, cenario), grupo) in enumerate(
        df_folds.groupby(['modelo', 'cenario'])
    ):
        for metrica in METRICAS_IC:
            low, high = bootstrap_ic(
                grupo[metrica].to_numpy(),
                seed=(
                    RANDOM_STATE
                    + grupo_indice * len(METRICAS_IC)
                    + METRICAS_IC.index(metrica)
                ),
            )
            resultados.append({
                'modelo': modelo,
                'cenario': cenario,
                'metrica': metrica,
                'media': grupo[metrica].mean(),
                'desvio_padrao': grupo[metrica].std(ddof=1),
                'ic95_inferior': low,
                'ic95_superior': high,
                'n_folds': len(grupo),
            })

    testes = []
    for cenario in df_folds['cenario'].unique():
        for metrica in METRICAS_TESTE:
            pares = []
            for modelo_a, modelo_b in itertools.combinations(modelos, 2):
                serie_a = df_folds.query(
                    'cenario == @cenario and modelo == @modelo_a'
                ).sort_values(['repeticao', 'fold'])[metrica].to_numpy()
                serie_b = df_folds.query(
                    'cenario == @cenario and modelo == @modelo_b'
                ).sort_values(['repeticao', 'fold'])[metrica].to_numpy()
                diferencas = serie_a - serie_b
                try:
                    teste = wilcoxon(
                        serie_a,
                        serie_b,
                        alternative='two-sided',
                        zero_method='wilcox',
                    )
                    p = float(teste.pvalue)
                    estatistica = float(teste.statistic)
                except ValueError:
                    p = 1.0
                    estatistica = np.nan
                pares.append({
                    'cenario': cenario,
                    'metrica': metrica,
                    'modelo_a': modelo_a,
                    'modelo_b': modelo_b,
                    'estatistica_wilcoxon': estatistica,
                    'p_valor': p,
                    'diferenca_media_a_menos_b': float(diferencas.mean()),
                    'n_pares': len(diferencas),
                })
            ajustados = holm_adjust([item['p_valor'] for item in pares])
            for item, ajustado in zip(pares, ajustados):
                item['p_valor_holm'] = float(ajustado)
                item['significativo_05'] = bool(ajustado < 0.05)
                testes.append(item)

    pd.DataFrame(resultados).to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[1]), index=False
    )
    pd.DataFrame(testes).to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[2]), index=False
    )
    pd.DataFrame(confusoes).groupby(
        ['modelo', 'cenario'], as_index=False
    )[['tn', 'fp', 'fn', 'tp', 'n_validacao']].sum().to_csv(
        os.path.join(PROJECT_ROOT, OUTPUTS[3]), index=False
    )
    print(f'Concluído com {repeticoes} repetição(ões).')
    print(f'Versões: {versoes_bibliotecas()}')


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
