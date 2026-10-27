# Evasão Zero

Aplicação de análise e predição de risco de evasão estudantil para o IFMS,
usando dados anonimizados e sintéticos derivados das respostas de pesquisa.

## Fluxo principal

1. `data/Respostas_evasao_limpo.csv` é a base de respostas usada no benchmark.
2. `scripts/02_benchmark_modelos_tcc.py` compara os modelos e treina os artefatos finais.
3. `deployments/dashboard/app.py` carrega os três modelos e os dados sincronizados com o S3.
4. `scripts/03_upload_s3.py` publica os três modelos e a base sintética no bucket configurado.

O benchmark também gera os resultados em `resultados_benchmark_tcc.csv` e as
figuras em `figures/tcc/`.

## Estrutura

```text
evasao-zero/
├── assets/                  # Imagens da documentação
├── data/                    # Bases de entrada e dados sintéticos
├── deployments/dashboard/   # Dashboard Streamlit e dependências de deploy
├── figures/tcc/             # Figuras geradas pelo benchmark
├── models/                  # Modelos treinados pelo benchmark
├── notebooks/               # Análise exploratória
├── scripts/                 # Geração, treino, benchmark e upload
├── src/                     # Código compartilhado para desserialização
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

Artefatos locais como `.venv/`, `catboost_info/`, caches do S3 e `*.egg-info/`
não fazem parte do código-fonte e são ignorados pelo Git.

## Instalação

Requer Python 3.9 ou superior.

```powershell
uv sync
```

Ou, usando `pip`:

```powershell
pip install -r requirements.txt
```

## Executar o dashboard

O dashboard depende de credenciais AWS configuradas no ambiente para baixar os
três modelos e os dados do S3. Depois de gerar ou atualizar os modelos, publique
os arquivos com o script de upload antes de iniciar o dashboard:

```powershell
uv run streamlit run deployments/dashboard/app.py
```

Para executar sem `uv`:

```powershell
streamlit run deployments/dashboard/app.py
```

## Reproduzir o pipeline

Para gerar a base sintética a partir da base de respostas:

```powershell
uv run python scripts/01_gerar_dados.py
```

Para executar o benchmark completo e regenerar modelos, resultados e figuras:

```powershell
uv run python scripts/02_benchmark_modelos_tcc.py
```

Para publicar os artefatos no S3:

```powershell
uv run python scripts/03_upload_s3.py
```

Na tela, o dashboard executa cada perfil nos três modelos simultaneamente e
mostra a probabilidade, a classificação e a taxa média de risco lado a lado.
No modo de lote, todas as colunas de resultado são exibidas automaticamente,
sem necessidade de selecionar ou filtrar um modelo.

## Licença

Este projeto está sob a licença MIT. Consulte [LICENSE](LICENSE).
