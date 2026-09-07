import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

st.set_page_config(
    page_title="Dashboard de Evasão Escolar",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    caminhos = [
        '../../data/dados_limpos.csv',
        'data/dados_limpos.csv',
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dados_limpos.csv')
    ]
    
    for caminho in caminhos:
        if os.path.exists(caminho):
            return pd.read_csv(caminho)
    
    st.error("Arquivo dados_limpos.csv não encontrado!")
    return None

@st.cache_resource
def carregar_modelo():
    caminhos_modelo = [
        'modelo_evasao.joblib',
        '../../models/modelo_evasao.joblib',
        os.path.join(os.path.dirname(__file__), 'modelo_evasao.joblib')
    ]
    
    caminhos_colunas = [
        'colunas_modelo.joblib',
        '../../models/colunas_modelo.joblib',
        os.path.join(os.path.dirname(__file__), 'colunas_modelo.joblib')
    ]
    
    modelo = colunas = None
    
    for caminho in caminhos_modelo:
        if os.path.exists(caminho):
            modelo = joblib.load(caminho)
            break
    
    for caminho in caminhos_colunas:
        if os.path.exists(caminho):
            colunas = joblib.load(caminho)
            break
    
    return modelo, colunas

# Carregar dados
df = carregar_dados()
modelo, colunas_modelo = carregar_modelo()

if df is None:
    st.stop()

# Título
st.title("📊 Dashboard de Análise de Evasão Escolar")
st.markdown("Análise baseada em dados do INSE e taxas de abandono escolar")

# Métricas principais
st.markdown("### 📈 Métricas Gerais")

df['TAXA_ABANDONO_GERAL'] = pd.to_numeric(df['TAXA_ABANDONO_GERAL'], errors='coerce')
df['risco_evasao'] = (df['TAXA_ABANDONO_GERAL'] > 5).astype(int)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Escolas", f"{len(df):,}")
col2.metric("Escolas com Alto Risco", f"{df['risco_evasao'].sum():,}")
col3.metric("Taxa Média de Abandono", f"{df['TAXA_ABANDONO_GERAL'].mean():.2f}%")
col4.metric("INSE Médio", f"{df['MEDIA_INSE'].mean():.2f}")

st.markdown("---")

# Gráficos
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📊 Distribuição por Classificação INSE")
    inse_dist = df['INSE_CLASSIFICACAO'].value_counts().reset_index()
    inse_dist.columns = ['Classificação', 'Quantidade']
    
    fig1 = px.bar(inse_dist, x='Classificação', y='Quantidade',
                  title='Escolas por Nível INSE',
                  color='Quantidade',
                  color_continuous_scale='Blues')
    st.plotly_chart(fig1, use_container_width=True)

with col_g2:
    st.subheader("🎯 Taxa de Abandono por Tipo de Rede")
    rede_map = {1: 'Federal', 2: 'Estadual', 3: 'Municipal'}
    df['Rede'] = df['TP_TIPO_REDE'].map(rede_map)
    
    taxa_rede = df.groupby('Rede')['TAXA_ABANDONO_GERAL'].mean().reset_index()
    
    fig2 = px.bar(taxa_rede, x='Rede', y='TAXA_ABANDONO_GERAL',
                  title='Taxa Média de Abandono por Rede',
                  labels={'TAXA_ABANDONO_GERAL': 'Taxa de Abandono (%)'},
                  color='TAXA_ABANDONO_GERAL',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

col_g3, col_g4 = st.columns(2)

with col_g3:
    st.subheader("📍 Escolas por Localização")
    loc_map = {1: 'Urbana', 2: 'Rural'}
    df['Localização'] = df['TP_LOCALIZACAO'].map(loc_map)
    
    loc_dist = df['Localização'].value_counts().reset_index()
    loc_dist.columns = ['Localização', 'Quantidade']
    
    fig3 = px.pie(loc_dist, values='Quantidade', names='Localização',
                  title='Distribuição por Localização')
    st.plotly_chart(fig3, use_container_width=True)

with col_g4:
    st.subheader("⚠️ Risco de Evasão")
    risco_dist = df['risco_evasao'].value_counts().reset_index()
    risco_dist.columns = ['Risco', 'Quantidade']
    risco_dist['Risco'] = risco_dist['Risco'].map({0: 'Baixo Risco', 1: 'Alto Risco'})
    
    fig4 = px.pie(risco_dist, values='Quantidade', names='Risco',
                  title='Distribuição de Risco de Evasão',
                  color='Risco',
                  color_discrete_map={'Baixo Risco': 'green', 'Alto Risco': 'red'})
    st.plotly_chart(fig4, use_container_width=True)

# Predição (se modelo disponível)
if modelo is not None and colunas_modelo is not None:
    st.markdown("---")
    st.markdown("### 🔮 Predição de Risco")
    
    with st.expander("Fazer Predição de Risco para uma Escola"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            media_inse = st.slider("Média INSE", 2.0, 8.0, 5.0, 0.1)
            tipo_rede = st.selectbox("Tipo de Rede", [1, 2, 3], format_func=lambda x: rede_map.get(x, x))
        
        with col_p2:
            localizacao = st.selectbox("Localização", [1, 2], format_func=lambda x: loc_map.get(x, x))
            capital = st.selectbox("Capital", [1, 2], format_func=lambda x: {1: 'Sim', 2: 'Não'}.get(x, x))
        
        with col_p3:
            pc_nivel_1 = st.slider("% Nível 1", 0.0, 100.0, 10.0)
            pc_nivel_2 = st.slider("% Nível 2", 0.0, 100.0, 20.0)
        
        if st.button("Prever Risco"):
            # Criar dados para predição
            dados_pred = pd.DataFrame({
                'MEDIA_INSE': [media_inse],
                'PC_NIVEL_1': [pc_nivel_1],
                'PC_NIVEL_2': [pc_nivel_2],
                'PC_NIVEL_3': [20.0],
                'PC_NIVEL_4': [20.0],
                'PC_NIVEL_5': [15.0],
                'PC_NIVEL_6': [10.0],
                'PC_NIVEL_7': [5.0],
                'PC_NIVEL_8': [0.0],
                'TP_TIPO_REDE': [tipo_rede],
                'TP_LOCALIZACAO': [localizacao],
                'TP_CAPITAL': [capital]
            })
            
            # Fazer predição
            predicao = modelo.predict(dados_pred)[0]
            prob = modelo.predict_proba(dados_pred)[0]
            
            if predicao == 1:
                st.error(f"⚠️ ALTO RISCO DE EVASÃO ({prob[1]*100:.1f}%)")
            else:
                st.success(f"✅ BAIXO RISCO DE EVASÃO ({prob[0]*100:.1f}%)")

st.markdown("---")
st.caption("Dashboard desenvolvido para análise de evasão escolar | Dados: INSE/SAEB")
