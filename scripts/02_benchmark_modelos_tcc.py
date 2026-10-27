"""
Benchmark Científico de Modelos para Predição de Evasão Escolar - IFMS
Protocolo TSTR (Train on Synthetic, Test on Real) com validação cruzada rigorosa.

Correções em relação à versão anterior:
  - Modelo 1 (GradientBoosting) agora usa GridSearchCV(n_jobs=1) dentro de cada fold.
  - Modelo 2 (CatBoost) com thread_count=2 e verbose=0.
  - Modelo 3 (TabPFN) com fallback automático para HistGradientBoosting se faltar RAM.
  - gc.collect() ao final de cada fold.
  - Gráfico de Barras Real vs Sintético (SDV) substitui o ablation study.
  - n_jobs=1 em todo o pipeline para respeitar a RAM disponível.

Autor: IFMS - TCC 2025
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    brier_score_loss, roc_curve
)
from sklearn.calibration import CalibrationDisplay

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DE CAMINHOS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Respostas_evasao_limpo.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'tcc')
RESULTS_PATH = os.path.join(BASE_DIR, 'resultados_benchmark_tcc.csv')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
FEATURES = ['curso', 'periodo', 'idade', 'cidade', 'renda_total',
            'trabalha_atualmente', 'recebe_auxilio_da_insti']
TARGET = 'risco_evasao'
N_SPLITS = 5          # folds do CV externo (TSTR)
N_INNER_CV = 3       # folds do GridSearchCV interno
N_SINTETICOS = 1000
RANDOM_STATE = 42

CAT_COLS = ['curso', 'idade', 'cidade', 'renda_total', 'trabalha_atualmente', 'recebe_auxilio_da_insti']
NUM_COLS = ['periodo']

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'font.family': 'DejaVu Sans',
})


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------
def carregar_dados():
    """Carrega o CSV real e cria a variável alvo binária (sem leakage)."""
    df = pd.read_csv(DATA_PATH)
    df['risco_evasao'] = df['ja_pensou_em_desistir'].apply(
        lambda x: 0 if isinstance(x, str) and 'nao nunca' in x.lower() else 1
    )
    df = df[FEATURES + [TARGET]].copy()
    df['periodo'] = pd.to_numeric(df['periodo'], errors='coerce').fillna(1).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. DEFINIÇÃO DOS MODELOS
# ---------------------------------------------------------------------------
class GBGridSearchWrapper:
    """
    Modelo 1 (Atual): GradientBoostingClassifier com OneHotEncoder e
    GridSearchCV interno (n_jobs=1).  O GridSearchCV roda dentro de cada
    fold TSTR apenas nos dados de treino daquele fold, eliminando leakage.
    """

    def __init__(self):
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
            ('num', SimpleImputer(strategy='median'), NUM_COLS),
        ])
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ])
        self.param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [2, 3, 4],
        }
        # Árvores rasas + min_samples_leaf para small data desbalanceado
        self.pipeline.named_steps['classifier'].min_samples_leaf = 5
        self.best_estimator_ = None

    def fit(self, X, y):
        inner_cv = StratifiedKFold(
            n_splits=N_INNER_CV, shuffle=True, random_state=RANDOM_STATE
        )
        grid = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=1,          # crítico para RAM
            verbose=0,
        )
        grid.fit(X, y)
        self.best_estimator_ = grid.best_estimator_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class CatBoostWrapper:
    """
    Modelo 2 (SOTA 1): CatBoostClassifier passando colunas categóricas
    nativamente via cat_features (sem OneHotEncoder).
    auto_class_weights='Balanced' para small data desbalanceado.
    Grid de hiperparâmetros rasos e regularizados para ~194 amostras.
    """

    def __init__(self):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=3,
            l2_leaf_reg=5,
            auto_class_weights='Balanced',
            thread_count=2,
            verbose=0,
            random_seed=RANDOM_STATE,
            eval_metric='AUC',
        )
        self.cat_features = CAT_COLS
        # Grid para tuning interno (feito via set_params antes do fit)
        self._param_grid = {
            'depth': [2, 3, 4],
            'l2_leaf_reg': [3, 5, 10],
            'iterations': [100, 200],
            'learning_rate': [0.03, 0.1],
        }
        self._best_params = None

    def fit(self, X, y):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)
        # GridSearchCV interno (n_jobs=1) para tuning sem leakage
        from sklearn.model_selection import GridSearchCV as _GSCV
        inner_cv = StratifiedKFold(
            n_splits=N_INNER_CV, shuffle=True, random_state=RANDOM_STATE
        )
        grid = _GSCV(
            estimator=self.model,
            param_grid=self._param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=1,
            verbose=0,
        )
        grid.fit(X, y, cat_features=self.cat_features)
        self.model = grid.best_estimator_
        self._best_params = grid.best_params_
        return self

    def predict(self, X):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def criar_pipeline_catboost():
    """Factory do CatBoost com fallback para HistGradientBoosting."""
    try:
        return CatBoostWrapper()
    except ImportError:
        print("  [AVISO] catboost não instalado. Usando HistGradientBoosting como substituto.")
        return criar_pipeline_hist_gradient_boosting(learning_rate=0.1)


def criar_pipeline_hist_gradient_boosting(learning_rate=0.05):
    """Cria o fallback HGB com tuning interno e sem leakage."""
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        ), CAT_COLS),
        ('num', SimpleImputer(strategy='median'), NUM_COLS),
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=learning_rate,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        )),
    ])
    return GridSearchCV(
        estimator=pipeline,
        param_grid={
            'classifier__max_depth': [2, 3, 4],
            'classifier__l2_regularization': [1.0, 5.0],
        },
        cv=StratifiedKFold(
            n_splits=N_INNER_CV, shuffle=True, random_state=RANDOM_STATE
        ),
        scoring='roc_auc',
        n_jobs=1,
        verbose=0,
    )


def criar_pipeline_tabpfn():
    """
    Modelo 3 (SOTA 2): TabPFNClassifier com OrdinalEncoder em Pipeline.
    Fallback automático para HistGradientBoostingClassifier se tabpfn
    não estiver instalado ou faltar RAM.
    """
    try:
        # Verifica RAM disponível antes de tentar carregar TabPFN
        try:
            import psutil
            ram_disponivel_gb = psutil.virtual_memory().available / (1024 ** 3)
            if ram_disponivel_gb < 2.0:
                raise RuntimeError(
                    f"RAM insuficiente para TabPFN ({ram_disponivel_gb:.1f} GB livre)"
                )
        except ImportError:
            pass  # psutil não instalado — continua e deixa o fit decidir

        from tabpfn import TabPFNClassifier
        preprocessor = ColumnTransformer([
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1
            ), CAT_COLS),
            ('num', SimpleImputer(strategy='median'), NUM_COLS),
        ])
        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', TabPFNClassifier(device='cpu', N_ensemble_configurations=4)),
        ])
    except (ImportError, RuntimeError, Exception) as e:
        print(f"  [AVISO] TabPFN indisponível ({e}). Usando HistGradientBoosting.")
        return criar_pipeline_hist_gradient_boosting(learning_rate=0.05)


# Dicionário de modelos: (nome -> factory)
# O GB é tratado como wrapper (já treina no __init__/fit com GridSearchCV).
MODELOS = {
    'GradientBoosting (Baseline)': lambda: GBGridSearchWrapper(),
    'CatBoost (SOTA)': criar_pipeline_catboost,
    'TabPFN / HistGB (SOTA)': criar_pipeline_tabpfn,
}


# ---------------------------------------------------------------------------
# 3. GERAÇÃO DE DADOS SINTÉTICOS (SDV) — SEM LEAKAGE
# ---------------------------------------------------------------------------
def gerar_sinteticos(train_df, n=N_SINTETICOS):
    """
    Ajusta o GaussianCopulaSynthesizer APENAS nos dados de treino do fold
    atual e gera n amostras sintéticas.  Isso elimina data leakage: o
    sintetizador nunca vê os dados de validação.
    """
    metadata = Metadata.detect_from_dataframes(data={'table': train_df})
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit({'table': train_df})
    amostras = synth.sample(num_rows=n)
    # SDV 2.0 retorna dict {nome_tabela: DataFrame}
    if isinstance(amostras, dict):
        amostras = amostras['table']
    return amostras


# ---------------------------------------------------------------------------
# 4. CÁLCULO DE MÉTRICAS
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, y_pred, y_proba, tempo):
    return {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'brier': brier_score_loss(y_true, y_proba),
        'tempo_s': tempo,
    }


# ---------------------------------------------------------------------------
# 5. LOOP PRINCIPAL DE BENCHMARK
# ---------------------------------------------------------------------------
def executar_benchmark(df):
    """
    Executa o protocolo TSTR com StratifiedKFold(n_splits=5).
    Para cada fold:
      Cenário A — treina apenas nos dados reais do fold.
      Cenário B — treina nos 1.000 sintéticos + reais do fold.
    A validação é sempre na partição real intacta.
    """
    X = df[FEATURES]
    y = df[TARGET].values

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    resultados = []
    oof_probas = {nome: {'A': np.zeros(len(y)), 'B': np.zeros(len(y))} for nome in MODELOS}
    oof_preds = {nome: {'A': np.zeros(len(y)), 'B': np.zeros(len(y))} for nome in MODELOS}

    # Guarda os sintéticos do último fold para a figura Real vs Sintético
    sinteticos_ultimo_fold = None
    reais_ultimo_fold = None

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold_idx}/{N_SPLITS}")
        print(f"{'=' * 60}")

        X_train_real = X.iloc[train_idx].reset_index(drop=True)
        y_train_real = y[train_idx]
        X_val_real = X.iloc[val_idx].reset_index(drop=True)
        y_val_real = y[val_idx]

        # Gera sintéticos APENAS com dados de treino deste fold
        print("  Gerando dados sintéticos (SDV)...")
        train_df_fold = X_train_real.copy()
        train_df_fold[TARGET] = y_train_real
        sinteticos_df = gerar_sinteticos(train_df_fold)
        X_sint = sinteticos_df[FEATURES]
        y_sint = sinteticos_df[TARGET].values

        # Cenário B: treino híbrido (sintético + real)
        X_hibrido = pd.concat([X_sint, X_train_real], ignore_index=True)
        y_hibrido = np.concatenate([y_sint, y_train_real])

        # Guarda para a figura Real vs Sintético
        if fold_idx == N_SPLITS:
            sinteticos_ultimo_fold = sinteticos_df
            reais_ultimo_fold = train_df_fold

        for nome, factory in MODELOS.items():
            print(f"  Modelo: {nome}")

            # --- Cenário A: apenas dados reais ---
            modelo_a = factory()
            t0 = time.time()
            modelo_a.fit(X_train_real, y_train_real)
            tempo_a = time.time() - t0
            proba_a = modelo_a.predict_proba(X_val_real)[:, 1]
            pred_a = modelo_a.predict(X_val_real)
            metricas_a = calcular_metricas(y_val_real, pred_a, proba_a, tempo_a)
            resultados.append({'fold': fold_idx, 'modelo': nome, 'cenario': 'A_Real', **metricas_a})
            oof_probas[nome]['A'][val_idx] = proba_a
            oof_preds[nome]['A'][val_idx] = pred_a

            # Libera memória do modelo A
            del modelo_a
            gc.collect()

            # --- Cenário B: dados sintéticos + reais ---
            modelo_b = factory()
            t0 = time.time()
            modelo_b.fit(X_hibrido, y_hibrido)
            tempo_b = time.time() - t0
            proba_b = modelo_b.predict_proba(X_val_real)[:, 1]
            pred_b = modelo_b.predict(X_val_real)
            metricas_b = calcular_metricas(y_val_real, pred_b, proba_b, tempo_b)
            resultados.append({'fold': fold_idx, 'modelo': nome, 'cenario': 'B_SDV1000', **metricas_b})
            oof_probas[nome]['B'][val_idx] = proba_b
            oof_preds[nome]['B'][val_idx] = pred_b

            print(f"    A) AUC={metricas_a['roc_auc']:.3f} | B) AUC={metricas_b['roc_auc']:.3f}")

            # Libera memória do modelo B
            del modelo_b
            gc.collect()

        # Limpa variáveis pesadas do fold
        del X_sint, y_sint, X_hibrido, y_hibrido, train_df_fold, sinteticos_df
        gc.collect()
        print(f"  [RAM] gc.collect() executado para o fold {fold_idx}")

    return (
        pd.DataFrame(resultados),
        oof_probas,
        oof_preds,
        y,
        sinteticos_ultimo_fold,
        reais_ultimo_fold,
    )


# ---------------------------------------------------------------------------
# 6. TABELA DE RESULTADOS CONSOLIDADA
# ---------------------------------------------------------------------------
def consolidar_resultados(df_resultados):
    """Média ± Desvio Padrão de cada métrica por modelo/cenário."""
    metricas_cols = ['roc_auc', 'f1_macro', 'f1_weighted', 'recall',
                     'precision', 'brier', 'tempo_s']
    rows = []
    for (modelo, cenario), grupo in df_resultados.groupby(['modelo', 'cenario']):
        row = {'Modelo': modelo, 'Cenario': cenario}
        for m in metricas_cols:
            row[m] = f"{grupo[m].mean():.3f} ± {grupo[m].std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. FIGURAS CIENTÍFICAS
# ---------------------------------------------------------------------------
def fig_curva_roc(oof_probas, y_true):
    """Figura 1: Curvas ROC comparativas OOF (Out-of-Fold)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ['#2196F3', '#F44336', '#4CAF50']

    for (nome, cenarios), cor in zip(oof_probas.items(), cores):
        # Usa o melhor cenário (B)
        proba = cenarios['B']
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, color=cor, lw=2, label=f'{nome} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Classificador Aleatório')
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC Comparativas OOF — Predição de Evasão (IFMS)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '01_curva_roc_comparativa.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 01_curva_roc_comparativa.png")


def fig_calibracao(oof_probas, y_true):
    """Figura 2: Curva de Calibração (Reliability Diagram)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ['#2196F3', '#F44336', '#4CAF50']

    for (nome, cenarios), cor in zip(oof_probas.items(), cores):
        CalibrationDisplay.from_predictions(
            y_true, cenarios['B'], n_bins=5, ax=ax,
            name=nome, color=cor,
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfeita Calibração')
    ax.set_title('Curva de Calibração — Confiabilidade das Probabilidades')
    ax.set_xlabel('Probabilidade Prevista')
    ax.set_ylabel('Frequência Observada')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '02_curva_calibracao_brier.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 02_curva_calibracao_brier.png")


def fig_real_vs_sintetico(df_real, df_sintetico):
    """
    Figura 3: Gráfico de Barras comparando a distribuição das variáveis
    categóricas entre dados reais (80 amostras) e sintéticos (1.000 amostras).
    """
    vars_plot = ['curso', 'idade', 'renda_total', 'trabalha_atualmente']
    titulos = {
        'curso': 'Curso',
        'idade': 'Faixa Etária',
        'renda_total': 'Renda Total',
        'trabalha_atualmente': 'Trabalha Atualmente',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, var in zip(axes, vars_plot):
        real_counts = df_real[var].value_counts(normalize=True).sort_index()
        sint_counts = df_sintetico[var].value_counts(normalize=True).sort_index()

        categorias = sorted(set(real_counts.index) | set(sint_counts.index))
        real_vals = [real_counts.get(c, 0) for c in categorias]
        sint_vals = [sint_counts.get(c, 0) for c in categorias]

        x = np.arange(len(categorias))
        width = 0.35

        ax.bar(x - width / 2, real_vals, width, label='Real (80)', color='#42A5F5', alpha=0.85)
        ax.bar(x + width / 2, sint_vals, width, label='Sintético (1.000)', color='#EF5350', alpha=0.85)

        ax.set_title(titulos[var], fontsize=12)
        ax.set_ylabel('Proporção')
        ax.set_xticks(x)
        ax.set_xticklabels(categorias, rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Distribuição Real vs Sintético (SDV GaussianCopula)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_real_vs_sintetico_sdv.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] 03_real_vs_sintetico_sdv.png")


def fig_shap(df_original, melhor_modelo_nome, melhor_pipeline):
    """Figuras 4 e 5: SHAP sobre as 80 amostras reais."""
    try:
        import shap

        X_real = df_original[FEATURES]
        y_real = df_original[TARGET].values

        # Retreina o melhor modelo em todos os dados reais para SHAP
        melhor_pipeline.fit(X_real, y_real)

        try:
            if hasattr(melhor_pipeline, 'named_steps'):
                # Pipeline sklearn (GB ou TabPFN/HistGB)
                X_transformado = melhor_pipeline.named_steps['preprocessor'].transform(X_real)
                clf = melhor_pipeline.named_steps['classifier']
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_transformado)
                ohe = melhor_pipeline.named_steps['preprocessor'].named_transformers_['cat']
                feature_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS
                X_plot = pd.DataFrame(X_transformado, columns=feature_names)
            elif hasattr(melhor_pipeline, 'model'):
                # CatBoostWrapper
                explainer = shap.TreeExplainer(melhor_pipeline.model)
                X_cat = X_real.copy()
                for col in CAT_COLS:
                    X_cat[col] = X_cat[col].astype(str)
                shap_values = explainer.shap_values(X_cat)
                feature_names = FEATURES
                X_plot = X_cat
            else:
                # GBGridSearchWrapper
                explainer = shap.TreeExplainer(
                    melhor_pipeline.best_estimator_.named_steps['classifier']
                )
                X_transformado = melhor_pipeline.best_estimator_.named_steps[
                    'preprocessor'
                ].transform(X_real)
                shap_values = explainer.shap_values(X_transformado)
                ohe = melhor_pipeline.best_estimator_.named_steps[
                    'preprocessor'
                ].named_transformers_['cat']
                feature_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS
                X_plot = pd.DataFrame(X_transformado, columns=feature_names)
        except Exception:
            explainer = shap.Explainer(melhor_pipeline.predict_proba, X_real)
            shap_values = explainer(X_real).values[:, :, 1]
            feature_names = FEATURES
            X_plot = X_real

        # Figura 4: Beeswarm
        fig4 = plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
            X_plot, feature_names=feature_names,
            show=False, plot_type='dot',
        )
        plt.title('SHAP — Impacto das Variáveis no Risco de Evasão (Beeswarm)',
                  fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '04_shap_summary_beeswarm.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print("  [OK] 04_shap_summary_beeswarm.png")

        # Figura 5: Bar importance
        fig5 = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
            X_plot, feature_names=feature_names,
            show=False, plot_type='bar',
        )
        plt.title('SHAP — Importância Global das Variáveis (Bar)', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, '05_shap_bar_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig5)
        print("  [OK] 05_shap_bar_importance.png")

    except ImportError:
        print("  [AVISO] shap não instalado. Pulando figuras SHAP.")
    except Exception as e:
        print(f"  [AVISO] Erro ao gerar SHAP: {e}")


# ---------------------------------------------------------------------------
# 8. TREINO FINAL E SALVAMENTO DOS MODELOS
# ---------------------------------------------------------------------------
# Importa o wrapper de um módulo importável para que joblib.load() funcione
# no dashboard Streamlit sem precisar da classe definida em __main__.
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
from model_wrapper import DashboardCompatibleModel


def treinar_e_salvar_modelos(df):
    """Treina cada modelo em todos os dados reais e salva em models/."""
    X = df[FEATURES]
    y = df[TARGET].values

    nomes_arquivo = {
        'GradientBoosting (Baseline)': 'gb_baseline_ifms.joblib',
        'CatBoost (SOTA)': 'catboost_sota_ifms.joblib',
        'TabPFN / HistGB (SOTA)': 'tabpfn_sota_ifms.joblib',
    }

    modelos_salvos = {}
    for nome, factory in MODELOS.items():
        print(f"  Treinando modelo final: {nome}")
        modelo = factory()
        modelo.fit(X, y)
        path = os.path.join(MODEL_DIR, nomes_arquivo[nome])

        # CatBoostWrapper precisa de wrapping compatível para o dashboard
        if hasattr(modelo, 'model') and hasattr(modelo, 'cat_features'):
            # Extrai o modelo CatBoost raw e envolve no wrapper compatível
            modelo_salvar = DashboardCompatibleModel(
                modelo.model, cat_features=modelo.cat_features
            )
        else:
            modelo_salvar = modelo

        joblib.dump(modelo_salvar, path)
        modelos_salvos[nome] = modelo_salvar
        print(f"    Salvo em: {path}")
        gc.collect()

    return modelos_salvos


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  BENCHMARK CIENTÍFICO — EVASÃO ZERO (IFMS TCC 2025)")
    print("=" * 60)

    print("\n[1/5] Carregando dados reais...")
    df = carregar_dados()
    print(f"  Total de amostras: {len(df)} | "
          f"Risco=1: {df[TARGET].sum()} ({df[TARGET].mean():.1%})")

    print("\n[2/5] Executando benchmark com validação cruzada (TSTR)...")
    (df_resultados, oof_probas, oof_preds, y_true,
     sinteticos_ultimo, reais_ultimo) = executar_benchmark(df)

    print("\n[3/5] Consolidando resultados...")
    df_consolidado = consolidar_resultados(df_resultados)
    df_consolidado.to_csv(RESULTS_PATH, index=False)
    print(f"  Tabela salva em: {RESULTS_PATH}")
    print("\n" + df_consolidado.to_string(index=False))

    print("\n[4/5] Gerando figuras científicas...")
    fig_curva_roc(oof_probas, y_true)
    fig_calibracao(oof_probas, y_true)
    fig_real_vs_sintetico(reais_ultimo, sinteticos_ultimo)

    # Determina modelo vencedor pelo ROC-AUC médio no Cenário B
    auc_por_modelo = (
        df_resultados[df_resultados['cenario'] == 'B_SDV1000']
        .groupby('modelo')['roc_auc'].mean()
    )
    melhor_nome = auc_por_modelo.idxmax()
    print(f"\n  Modelo vencedor: {melhor_nome} "
          f"(AUC médio B = {auc_por_modelo[melhor_nome]:.3f})")

    melhor_pipeline = MODELOS[melhor_nome]()
    fig_shap(df, melhor_nome, melhor_pipeline)

    print("\n[5/5] Treinando e salvando modelos finais...")
    modelos_salvos = treinar_e_salvar_modelos(df)

    # Salva o vencedor como modelo principal do dashboard
    joblib.dump(
        modelos_salvos[melhor_nome],
        os.path.join(MODEL_DIR, 'modelo_evasao_ifms.joblib'),
    )
    print(f"  Modelo principal salvo: models/modelo_evasao_ifms.joblib ({melhor_nome})")

    print("\n" + "=" * 60)
    print("  BENCHMARK CONCLUÍDO COM SUCESSO!")
    print(f"  Figuras:    {FIGURES_DIR}")
    print(f"  Resultados: {RESULTS_PATH}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
