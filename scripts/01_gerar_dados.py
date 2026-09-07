import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality
import os

# 1. Configuração de Caminhos Relativos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN = os.path.join(BASE_DIR, 'data', 'Respostas_evasao_limpo.csv')
DATA_OUT = os.path.join(BASE_DIR, 'data', 'dados_sinteticos_ifms.csv')

def main():
    print("=" * 60)
    print(" GERAÇÃO DE DADOS SINTÉTICOS (SDV) - IFMS JARDIM")
    print("=" * 60)

    # 2. Carregamento
    print("\n1. Carregando dados reais...")
    df = pd.read_csv(DATA_IN)
    
    # 3. Engenharia de Features e Prevenção de Data Leakage
    print("2. Tratando variável alvo e removendo vazamento de dados...")
    
    # Padroniza a variável alvo (0 para sem risco, 1 para com risco)
    df['risco_evasao'] = df['ja_pensou_em_desistir'].apply(
        lambda x: 0 if isinstance(x, str) and 'nao nunca' in x.lower() else 1
    )
    
    # Remove as justificativas pós-fato (o modelo deve prever antes de acontecer)
    colunas_vazamento = ['ja_pensou_em_desistir', 'motivo_de_desistencia', 'motivo_agrupado']
    df_limpo = df.drop(columns=[col for col in colunas_vazamento if col in df.columns])
    
    print(f"   Colunas mantidas para síntese: {df_limpo.columns.tolist()}")

    # 4. Detecção de Metadados (Uso da classe Metadata correta)
    print("\n3. Mapeando metadados estatísticos das colunas...")
    metadata = Metadata.detect_from_dataframe(data=df_limpo)

    # 5. Treinamento do Sintetizador
    print("4. Treinando o modelo gerador (GaussianCopula)...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df_limpo)

    # 6. Geração da Nova Base
    NUM_ALUNOS = 1000
    print(f"5. Gerando {NUM_ALUNOS} alunos sintéticos...")
    synthetic_data = synthesizer.sample(num_rows=NUM_ALUNOS)

    # 7. Avaliação de Qualidade
    print("\n6. Avaliando a fidelidade estatística dos dados gerados...")
    quality_report = evaluate_quality(
        real_data=df_limpo,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    # 8. Exportação
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    synthetic_data.to_csv(DATA_OUT, index=False)
    print("\n" + "=" * 60)
    print(f"[SUCESSO] Arquivo salvo em: {DATA_OUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()