# 📊 Projeto de Predição de Evasão Escolar

## 🎯 O que é este projeto?

Sistema de análise e predição de risco de evasão escolar usando dados do INSE (Indicador de Nível Socioeconômico) e taxas de abandono.

## 📁 Estrutura do Projeto

```
evasao-zero/
├── data/
│   └── dados_limpos.csv          # Dados das escolas (INSE + abandono)
├── models/
│   ├── modelo_evasao.joblib      # Modelo treinado (Random Forest)
│   └── colunas_modelo.joblib     # Features do modelo
├── deployments/dashboard/
│   ├── app_simples.py            # ✅ DASHBOARD PRINCIPAL (USE ESTE!)
│   ├── modelo_evasao.joblib      # Cópia do modelo
│   └── colunas_modelo.joblib     # Cópia das features
└── scripts/
    └── train_model_real.py       # Script para treinar o modelo
```

## 🚀 Como Usar

### 1. Executar o Dashboard (Recomendado)

```bash
cd c:\Users\katys\OneDrive\Desktop\IC-EVASAO\evasao-zero
streamlit run deployments\dashboard\app_simples.py
```

O dashboard mostra:
- 📈 Métricas gerais (total de escolas, risco, taxa de abandono)
- 📊 Gráficos de distribuição por INSE, rede, localização
- 🔮 Predição de risco para novas escolas

### 2. Retreinar o Modelo (Opcional)

Se você atualizar os dados:

```bash
python scripts\train_model_real.py
```

## 📊 Sobre os Dados

**Fonte**: Dados do INSE/SAEB de escolas brasileiras

**Features do Modelo**:
- MEDIA_INSE: Média do indicador socioeconômico
- PC_NIVEL_1 a PC_NIVEL_8: Percentual de alunos em cada nível
- TP_TIPO_REDE: Tipo de rede (Federal/Estadual/Municipal)
- TP_LOCALIZACAO: Localização (Urbana/Rural)
- TP_CAPITAL: Se é capital ou não

**Variável Alvo**: Risco de evasão (1 = Alto risco se taxa de abandono > 5%)

## 📈 Performance do Modelo

- **Acurácia**: 90.82%
- **Algoritmo**: Random Forest (100 árvores)
- **Validação**: 5-fold cross-validation
- **Features mais importantes**: MEDIA_INSE, PC_NIVEL_2, PC_NIVEL_6

## ❓ Problemas Comuns

### Dashboard não abre?
```bash
# Instale o Streamlit
pip install streamlit

# Execute novamente
streamlit run deployments\dashboard\app_simples.py
```

### Modelo não encontrado?
```bash
# Treine o modelo
python scripts\train_model_real.py
```

### Erro de importação?
```bash
# Instale as dependências
pip install pandas numpy scikit-learn joblib plotly
```

## 🗑️ Arquivos que Podem Ser Ignorados

- `notebooks/03_modelo_melhorado.ipynb` - Versão antiga
- `scripts/train_model_improved.py` - Versão antiga
- `deployments/dashboard/app.py` - Versão antiga (use app_simples.py)
- `COMO_TREINAR_MODELO.md` - Documentação antiga

## 📝 Resumo Rápido

1. **Dados**: `data/dados_limpos.csv` (58.774 escolas)
2. **Modelo**: `models/modelo_evasao.joblib` (já treinado)
3. **Dashboard**: `deployments/dashboard/app_simples.py` (pronto para usar)
4. **Comando**: `streamlit run deployments\dashboard\app_simples.py`

---

**Versão**: 2.0 (Simplificada)  
**Última atualização**: 2024
