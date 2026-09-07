import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# Configuração de Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dados_sinteticos_ifms.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def main():
    print("=" * 60)
    print(" OTIMIZAÇÃO AVANÇADA DO MODELO PREDITIVO - IFMS")
    print("=" * 60)

    # 1. Carregamento
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['risco_evasao'])
    y = df['risco_evasao']

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Pré-processador com filtro de ruído (min_frequency)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(
                handle_unknown='infrequent_if_exist', 
                drop='first', 
                min_frequency=0.05
            ), categorical_cols)
        ],
        remainder='passthrough'
    )

    # 3. Pipeline com Gradient Boosting
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    # 4. Busca em Grade otimizando para ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5, 7]
    }

    print("\nIniciando otimização com Gradient Boosting (Métrica: ROC-AUC)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    # 5. Resultados e Avaliação
    melhor_modelo = grid_search.best_estimator_
    print(f"\nMelhores Hiperparâmetros: {grid_search.best_params_}")
    print(f"Score ROC-AUC Médio (Validação Cruzada): {grid_search.best_score_:.4f}")

    # Relatório de classificação na base total para checar falsos negativos
    y_pred = melhor_modelo.predict(X)
    print("\nRelatório de Classificação (Visão Geral):")
    print(classification_report(y, y_pred))

    # 6. Salvar Modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'modelo_evasao_ifms.joblib')
    joblib.dump(melhor_modelo, model_path)
    
    print("\n" + "=" * 60)
    print(f"[SUCESSO] Modelo Gradient Boosting salvo em: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()