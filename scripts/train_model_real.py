import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    print("=" * 60)
    print("TREINO DO MODELO - DADOS DO FORMULÁRIO DE EVASÃO (K-FOLD 5 + GRIDSEARCH)")
    print("=" * 60)
    
    # 1. Carregar os dados
    print("\n1. Carregando dados do arquivo...")
    nome_ficheiro = 'C:/Users/denne/evasao-zero/data/Respostas_evasao_limpo.csv'
    
    try:
        df = pd.read_csv(nome_ficheiro)
    except FileNotFoundError:
        print(f"   [ERRO] O arquivo '{nome_ficheiro}' não foi encontrado.")
        return 

    df.columns = df.columns.str.strip()
    
    # 2. Criar a variável alvo (Target)
    print(f"\n2. Classificando o risco de evasão...")
    coluna_alvo = 'ja_pensou_em_desistir'
    
    if coluna_alvo not in df.columns:
        print(f"   [ERRO FATAL] A coluna '{coluna_alvo}' não existe!")
        return

    # 0 = Não, 1 = Sim/Risco
    df['risco_evasao'] = df[coluna_alvo].apply(
        lambda x: 0 if isinstance(x, str) and 'nao nunca' in x.lower() else 1
    )
    X_raw = df.drop(columns=[coluna_alvo, 'motivo_de_desistencia', 'motivo_agrupado', 'risco_evasao'], errors='ignore')
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df['risco_evasao']
    features_finais = X.columns.tolist()

    # 3. Configurar a Validação Cruzada (5 Folds = 80/20 cinco vezes)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n3. Iniciando Ciclos de Treino (Cada ciclo: 80% Treino / 20% Teste)")
    
    historico_accuracy = []
    melhor_modelo = None
    melhor_acc = 0

    # LOOP DOS 5 FOLDS
    for i, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        print(f"\n--- CICLO {i} ---")
        print("Buscando os melhores hiperparâmetros (Isso pode demorar um pouco)...")
        
        # Divisão dos dados do Fold atual
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        
        # Parâmetros para testar no GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }

        # Inicializa o modelo base
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Procura a melhor combinação (fazendo um sub-kfold de 3)
        # verbose=0 para não poluir demais a tela durante os ciclos
        grid_search = GridSearchCV(
            estimator=rf_base, 
            param_grid=param_grid, 
            cv=3, 
            n_jobs=-1, 
            scoring='accuracy', 
            verbose=0
        )

        # Treina para achar os melhores hiperparâmetros no fold atual
        grid_search.fit(X_train_cv, y_train_cv)

        # Usa o melhor modelo encontrado
        modelo = grid_search.best_estimator_
        print(f"Melhores parâmetros neste fold: {grid_search.best_params_}")
        
        # Testando o melhor modelo do ciclo
        y_pred = modelo.predict(X_test_cv)
        acc = accuracy_score(y_test_cv, y_pred)
        historico_accuracy.append(acc)
        
        print(f"Acurácia: {acc:.4f}")
        print(classification_report(y_test_cv, y_pred, zero_division=0))
        
        # Guardar o melhor modelo para salvar no final
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_modelo = modelo

    # 4. Resultado Final da Validação
    print("\n" + "=" * 60)
    print(f"MÉDIA DE ACURÁCIA (5 FOLDS): {np.mean(historico_accuracy):.4f}")
    print("=" * 60)

    # 5. Importância das respostas (Baseado no melhor modelo de todos os ciclos)
    feature_importance = pd.DataFrame({
        'Fator de Risco': features_finais,
        'Peso (%)': melhor_modelo.feature_importances_ * 100 
    }).sort_values('Peso (%)', ascending=False)
    
    print("\n📊 OS 10 MAIORES ÍNDICES DE RISCO DE EVASÃO (Melhor Ciclo)")
    for i, (idx, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        fator_limpo = row['Fator de Risco'].replace('_', ' ').title()
        barra = "█" * int(row['Peso (%)'] / 0.5)
        print(f" {i:02d}. {fator_limpo:<45} | {row['Peso (%)']:>5.2f}% | {barra}")

    # 6. Guardar o modelo
    print(f"\n10. Guardando o melhor modelo no disco...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(melhor_modelo, 'models/modelo_formulario.joblib')
    joblib.dump(features_finais, 'models/colunas_formulario.joblib')
    
    print("\n" + "=" * 60)
    print("[SUCESSO] PROCESSO CONCLUÍDO!")
    print("=" * 60)

if __name__ == "__main__":
    main()