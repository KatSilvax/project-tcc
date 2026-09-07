import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import boto3

# --- Configuração da Página e Identidade Visual ---
st.set_page_config(page_title="Predikt IFMS | Evasão Zero", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .stApp { border-top: 6px solid #2E8B57; }
    .stButton>button { background-color: #2E8B57; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #1e5c3a; color: white; }
    h1, h2, h3 { color: #2E8B57; }
    </style>
""", unsafe_allow_html=True)

# --- Configuração AWS S3 ---
BUCKET_NAME = 'evasao-zero-dados-ifms-katy-849275'

@st.cache_resource
def inicializar_nuvem():
    """Conecta na AWS, baixa os artefatos do S3 e carrega na memória do Dashboard."""
    s3 = boto3.client('s3', region_name='us-east-1')
    modelo_local = 'modelo_nuvem_cache.joblib'
    dados_local = 'dados_nuvem_cache.csv'
    
    # Download do modelo do S3
    if not os.path.exists(modelo_local):
        s3.download_file(BUCKET_NAME, 'modelos/modelo_evasao_ifms.joblib', modelo_local)
    
    # Download dos dados base do S3 (cumprindo requisito do Trello)
    if not os.path.exists(dados_local):
        s3.download_file(BUCKET_NAME, 'dados/dados_sinteticos_ifms.csv', dados_local)
        
    modelo = joblib.load(modelo_local)
    df_base = pd.read_csv(dados_local)
    
    return modelo, df_base

# Executa a conexão com a nuvem e exibe spinner de carregamento
with st.spinner('☁️ Sincronizando modelo preditivo com a AWS S3...'):
    try:
        modelo, df_base = inicializar_nuvem()
    except Exception as e:
        st.error(f"🚨 Falha de comunicação com a Nuvem AWS: {e}")
        st.stop()

# --- BARRA LATERAL: Análise Individual ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Marca_IFMS.png/800px-Marca_IFMS.png", width=150)
st.sidebar.title("🔍 Análise Individual")
st.sidebar.markdown("Preencha o perfil para prever o risco imediato.")

with st.sidebar.form("form_individual"):
    curso = st.selectbox("Curso", ["computacao", "arquitetura", "edificacoes", "informatica"])
    periodo = st.number_input("Período (Semestre)", min_value=1, max_value=10, value=1)
    idade = st.selectbox("Faixa Etária", ["ate 20 anos", "21 a 30 anos", "31 a 40 anos", "mais de 40 anos"])
    cidade = st.selectbox("Cidade de Origem", ["Jardim", "Guia Lopes da Laguna", "Nioaque", "Outra"])
    renda = st.selectbox("Renda Familiar", ["ate 1 salario minimo", "entre 1 e 2 salarios minimos", "entre 2 e 3 salarios minimos", "mais de 3 salarios minimos"])
    trabalha = st.selectbox("Situação de Trabalho", ["nao", "sim, meio periodo", "sim, em periodo integral", "estagio"])
    auxilio = st.selectbox("Auxílio Institucional", ["nao", "sim", "ja solicitei e aguardo reposta"])
    
    analisar_btn = st.form_submit_button("Avaliar Risco")

if analisar_btn:
    aluno_df = pd.DataFrame([{
        'curso': curso, 'periodo': periodo, 'idade': idade, 
        'cidade': cidade, 'renda_total': renda, 
        'trabalha_atualmente': trabalha, 'recebe_auxilio_da_insti': auxilio
    }])
    
    predicao = modelo.predict(aluno_df)[0]
    probabilidade = modelo.predict_proba(aluno_df)[0][1]
    
    st.sidebar.markdown("---")
    if predicao == 1:
        st.sidebar.error(f"🚨 **ALTO RISCO DE EVASÃO**\n\nProbabilidade: {probabilidade:.1%}")
    else:
        st.sidebar.success(f"✅ **ALUNO ESTÁVEL**\n\nProbabilidade de risco: {probabilidade:.1%}")

# --- ÁREA PRINCIPAL: Análise em Lote ---
st.title("🎓 Predikt IFMS: Monitoramento em Nuvem")
st.markdown(f"**Status da Inteligência Artificial:** Online 🟢 | **Base de Conhecimento:** {len(df_base)} perfis sincronizados da AWS.")
st.markdown("Faça o upload da planilha da turma para identificar grupos de risco.")

arquivo_upload = st.file_uploader("Upload de Planilha CSV", type=["csv"])

if arquivo_upload is not None:
    try:
        df_lote = pd.read_csv(arquivo_upload)
        colunas_necessarias = ['curso', 'periodo', 'idade', 'cidade', 'renda_total', 'trabalha_atualmente', 'recebe_auxilio_da_insti']
        colunas_faltantes = [col for col in colunas_necessarias if col not in df_lote.columns]
        
        if colunas_faltantes:
            st.error(f"A planilha enviada não possui as colunas obrigatórias. Faltam: {', '.join(colunas_faltantes)}")
        else:
            with st.spinner("Processando lote..."):
                df_lote['risco_evasao_predito'] = modelo.predict(df_lote[colunas_necessarias])
                probabilidades = modelo.predict_proba(df_lote[colunas_necessarias])[:, 1]
                df_lote['probabilidade_risco'] = (probabilidades * 100).round(1)
                
                alunos_risco = df_lote[df_lote['risco_evasao_predito'] == 1]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Analisado", len(df_lote))
                col2.metric("Alunos em Risco", len(alunos_risco), delta_color="inverse")
                col3.metric("Taxa de Risco", f"{(len(alunos_risco)/len(df_lote))*100:.1f}%")
                
                st.markdown("---")
                col_graf1, col_graf2 = st.columns(2)
                
                with col_graf1:
                    st.subheader("📊 Risco por Renda")
                    risco_renda = df_lote.groupby('renda_total')['risco_evasao_predito'].mean().reset_index()
                    fig1 = px.bar(risco_renda, x='renda_total', y='risco_evasao_predito', 
                                  labels={'risco_evasao_predito': 'Taxa', 'renda_total': 'Renda'}, color='risco_evasao_predito', color_continuous_scale='Reds')
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with col_graf2:
                    st.subheader("🏢 Risco por Trabalho")
                    risco_trabalho = df_lote.groupby('trabalha_atualmente')['risco_evasao_predito'].mean().reset_index()
                    fig2 = px.pie(risco_trabalho, values='risco_evasao_predito', names='trabalha_atualmente', hole=0.4)
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📋 Lista Prioritária (Alunos em Risco)")
                st.dataframe(alunos_risco.sort_values(by='probabilidade_risco', ascending=False), use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("👆 Aguardando o envio da planilha. Você pode testar previsões individuais na barra lateral.")