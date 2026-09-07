import pandas as pd
import joblib

def main():
    print("=" * 60)
    print(" TESTE EM MASSA (LOTE) COM DADOS SIMULADOS")
    print("=" * 60)

    # 1. Carregar o modelo
    try:
        modelo = joblib.load('models/modelo_formulario.joblib')
        colunas_treino = joblib.load('models/colunas_formulario.joblib')
    except FileNotFoundError:
        print("   [ERRO] Ficheiros do modelo não encontrados na pasta 'models/'.")
        return

    # 2. Carregar o ficheiro CSV que acabou de descarregar
    nome_ficheiro = 'C:/Users/denne/Downloads/dados_teste_simulados.csv'
    try:
        df_novos = pd.read_csv(nome_ficheiro)
        print(f"   [OK] {len(df_novos)} alunos fictícios carregados.\n")
    except FileNotFoundError:
        print(f"   [ERRO] O ficheiro '{nome_ficheiro}' não foi encontrado.")
        return

    # 3. Guardar os gabaritos (respostas reais) e os nomes, e depois retirá-los dos dados
    nomes = df_novos['Nome']
    gabarito_real = df_novos['ja_pensou_em_desistir'].apply(
        lambda x: 0 if isinstance(x, str) and 'nao nunca' in x.lower() else 1
    )
    
    # Remove as colunas que o modelo não deve ver para prever
    dados_para_analise = df_novos.drop(columns=['Nome', 'ja_pensou_em_desistir'])

    # 4. Preparação matemática mágica
    dados_processados = pd.get_dummies(dados_para_analise)
    dados_finais = dados_processados.reindex(columns=colunas_treino, fill_value=0)

  # 5. Previsões
    probabilidades = modelo.predict_proba(dados_finais)[:, 1] * 100

    # 6. Mostrar os resultados (AGORA COM A REGRA DOS 40%)
    print(f"{'ALUNO':<20} | {'PREVISÃO DO MODELO':<18} | {'REALIDADE (Gabarito)':<20} | {'PROBABILIDADE (%)'}")
    print("-" * 82)
    
    acertos = 0
    verdadeiros_positivos = 0 # Quantos alunos em risco real nós conseguimos salvar?
    total_risco_real = sum(gabarito_real)

    for i in range(len(nomes)):
        previsao_ajustada = 1 if probabilidades[i] >= 40.0 else 0
        
        prev_str = "🚨 ALTO RISCO" if previsao_ajustada == 1 else "✅ Estável"
        real_str = "🚨 ALTO RISCO" if gabarito_real.iloc[i] == 1 else "✅ Estável"
        
        # Conta acertos gerais
        if previsao_ajustada == gabarito_real.iloc[i]:
            acertos += 1
            
        # Conta quantos alunos em risco nós conseguimos detetar a tempo
        if previsao_ajustada == 1 and gabarito_real.iloc[i] == 1:
            verdadeiros_positivos += 1
            
        if i < 15:
            print(f"{nomes.iloc[i]:<20} | {prev_str:<18} | {real_str:<20} | {probabilidades[i]:.1f}%")
            
    print(f"\n... e assim por diante (mostrando apenas os 15 primeiros).")
    
    # 7. Resumo do Teste Otimizado
    print("\n" + "=" * 60)
    print(" RESUMO DA SIMULAÇÃO (RADAR AJUSTADO PARA 40%)")
    print("=" * 60)
    print(f" Total de alunos testados         : {len(nomes)}")
    print(f" Acertos Globais                  : {acertos} ({acertos/len(nomes)*100:.1f}%)")
    print(f" Alunos em Risco Real detetados   : {verdadeiros_positivos} de {total_risco_real}")
    print("=" * 60)

if __name__ == "__main__":
    main()