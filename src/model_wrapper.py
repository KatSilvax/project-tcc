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
