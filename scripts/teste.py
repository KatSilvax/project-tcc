import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import evaluate_quality

# 1. Carregar seus dados
real_data = pd.read_csv('C:/Users/denne/evasao-zero/data/Respostas_evasao_limpo.csv')

# 2. FORMA CORRETA: Detectar metadados usando o método de classe
# Isso já cria o objeto Metadata populado com a sua tabela
metadata = Metadata.detect_from_dataframe(data=real_data)

# 3. Validar se a idade deve ser numérica (opcional mas recomendado)
# Se 'idade' estiver como categorical, o SDV não criará idades novas, 
# apenas repetirá as que já existem. Para criar idades novas, use:
# metadata.update_column(column_name='idade', sdtype='numerical')

# 4. Treinar o modelo
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

# 5. Gerar novas respostas
synthetic_data = synthesizer.sample(num_rows=1000)

# 6. Salvar os resultados
synthetic_data.to_csv('respostas_sinteticas.csv', index=False)
metadata.save_to_json('metadata_evasao.json')

print("Dados gerados com sucesso!")

# 7. VERIFICAÇÃO DE QUALIDADE
# Vamos ver quão bom ficou o resultado em comparação ao real
report = evaluate_quality(real_data, synthetic_data, metadata)