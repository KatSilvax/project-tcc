# 🎯 GUIA RÁPIDO - 3 PASSOS

## ✅ PASSO 1: Abrir Terminal.
```
Pressione: Windows + R
Digite: cmd
Pressione: Enter
```

## ✅ PASSO 2: Ir para a pasta do projeto
```bash
cd c:\Users\katys\OneDrive\Desktop\IC-EVASAO\evasao-zero
```

## ✅ PASSO 3: Executar o Dashboard
```bash
streamlit run deployments\dashboard\app_simples.py
```

---

## 🎉 PRONTO!

O navegador vai abrir automaticamente com o dashboard!

---

## 📁 Arquivos Importantes

| Arquivo | O que é | Precisa mexer? |
|---------|---------|----------------|
| `data/dados_limpos.csv` | Dados das escolas | ❌ Não |
| `models/modelo_evasao.joblib` | Modelo treinado | ❌ Não |
| `deployments/dashboard/app_simples.py` | Dashboard | ❌ Não |
| `scripts/train_model_real.py` | Treinar modelo | ⚠️ Só se atualizar dados |

---

## ❌ Se der erro

### Erro: "streamlit não encontrado"
```bash
pip install streamlit
```

### Erro: "arquivo não encontrado"
Verifique se está na pasta correta:
```bash
dir
```
Deve mostrar as pastas: data, models, deployments, scripts

---

## 🔄 Para Retreinar o Modelo

Só faça isso se você tiver NOVOS DADOS:

```bash
python scripts\train_model_real.py
```

Depois execute o dashboard normalmente.

---

**Dúvidas?** Leia o arquivo `LEIA-ME.md` para mais detalhes.
