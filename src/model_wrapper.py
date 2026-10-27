"""
Wrapper compatível para serialização de modelos via joblib.

O DashboardCompatibleModel expõe a interface predict() / predict_proba()
que o dashboard Streamlit espera, aceitando um DataFrame bruto com as
7 features originais. Funciona com CatBoost (cat_features nativo) e
Pipelines sklearn.

Este módulo deve ser importável tanto pelo script de benchmark quanto
pelo dashboard para que o joblib.load() encontre a classe.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class GBGridSearchWrapper:
    """Wrapper importável usado pelo modelo GradientBoosting salvo com joblib."""

    def __init__(self):
        categorical_columns = [
            'curso',
            'idade',
            'cidade',
            'renda_total',
            'trabalha_atualmente',
            'recebe_auxilio_da_insti',
        ]
        numeric_columns = ['periodo']
        self.pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    categorical_columns,
                ),
                ('num', SimpleImputer(strategy='median'), numeric_columns),
            ])),
            ('classifier', GradientBoostingClassifier(random_state=42)),
        ])
        self.pipeline.named_steps['classifier'].min_samples_leaf = 5
        self.best_estimator_ = None

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


class DashboardCompatibleModel:
    """
    Wrapper que serializa corretamente via joblib e expõe a interface
    predict() / predict_proba() que o dashboard Streamlit espera,
    aceitando um DataFrame bruto com as 7 features originais.
    Funciona com CatBoost (cat_features nativo) e Pipelines sklearn.
    """

    def __init__(self, modelo, cat_features=None):
        self.modelo = modelo
        self.cat_features = cat_features or []

    def _prepare(self, X):
        X = X.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)
        return X

    def predict(self, X):
        X = self._prepare(X)
        return self.modelo.predict(X)

    def predict_proba(self, X):
        X = self._prepare(X)
        return self.modelo.predict_proba(X)
