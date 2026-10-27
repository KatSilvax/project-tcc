import streamlit as st
import pandas as pd
import joblib
import os
import sys
import boto3
import botocore.session

# Garante que o módulo model_wrapper seja importável para joblib.load()
# encontrar a classe DashboardCompatibleModel ao deserializar o modelo.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if os.path.join(_PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))

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
MODEL_FILES = {
    'GradientBoosting (Baseline)': 'gb_baseline_ifms.joblib',
    'CatBoost (SOTA)': 'catboost_sota_ifms.joblib',
    'TabPFN / HistGB (SOTA)': 'tabpfn_sota_ifms.joblib',
}
FEATURE_COLUMNS = [
    'curso',
    'periodo',
    'idade',
    'cidade',
    'renda_total',
    'trabalha_atualmente',
    'recebe_auxilio_da_insti',
]

@st.cache_resource
def inicializar_nuvem():
    """Baixa os três modelos e os dados do S3, carregando-os na memória."""
    bc_session = botocore.session.Session()
    resolver = bc_session.get_component('credential_provider')
    for provider in ['sso', 'sso-token', 'login']:
        try:
            resolver.remove(provider)
        except Exception:
            pass
    session = boto3.Session(botocore_session=bc_session)
    s3 = session.client('s3', region_name='us-east-1')
    modelos = {}
    for nome, arquivo in MODEL_FILES.items():
        caminho_local = os.path.join(_PROJECT_ROOT, 'models', arquivo)
        if not os.path.exists(caminho_local):
            os.makedirs(os.path.dirname(caminho_local), exist_ok=True)
            s3.download_file(BUCKET_NAME, f'modelos/{arquivo}', caminho_local)
        modelos[nome] = joblib.load(caminho_local)

    dados_local = os.path.join(_PROJECT_ROOT, 'data', 'dados_sinteticos_ifms.csv')
    if not os.path.exists(dados_local):
        os.makedirs(os.path.dirname(dados_local), exist_ok=True)
        s3.download_file(
            BUCKET_NAME,
            'dados/dados_sinteticos_ifms.csv',
            dados_local,
        )
        
    df_base = pd.read_csv(dados_local)
    
    return modelos, df_base

# Executa a conexão com a nuvem e exibe spinner de carregamento
with st.spinner('☁️ Sincronizando modelo preditivo com a AWS S3...'):
    try:
        modelos, df_base = inicializar_nuvem()
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
    
    st.sidebar.markdown("---")
    comparacao_individual = []
    for nome, modelo in modelos.items():
        predicao = modelo.predict(aluno_df)[0]
        probabilidade = modelo.predict_proba(aluno_df)[0][1]
        comparacao_individual.append({
            'Modelo': nome,
            'Classificação': 'Alto risco' if predicao == 1 else 'Estável',
            'Probabilidade de evasão': probabilidade,
        })

    resultado_individual = pd.DataFrame(comparacao_individual)
    st.sidebar.dataframe(
        resultado_individual.style.format({'Probabilidade de evasão': '{:.1%}'}),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Comparação dos modelos — perfil individual")
    st.bar_chart(
        resultado_individual.set_index('Modelo')['Probabilidade de evasão'],
        y_label='Probabilidade de evasão',
    )

# --- ÁREA PRINCIPAL: Análise em Lote ---
st.title("🎓 Predikt IFMS: Monitoramento em Nuvem")
st.markdown(f"**Status da Inteligência Artificial:** Online 🟢 | **Base de Conhecimento:** {len(df_base)} perfis sincronizados da AWS.")
st.markdown("Faça o upload da planilha da turma para identificar grupos de risco.")

arquivo_upload = st.file_uploader("Upload de Planilha CSV", type=["csv"])

if arquivo_upload is not None:
    try:
        df_lote = pd.read_csv(arquivo_upload)
        colunas_necessarias = FEATURE_COLUMNS
        colunas_faltantes = [col for col in colunas_necessarias if col not in df_lote.columns]
        
        if colunas_faltantes:
            st.error(f"A planilha enviada não possui as colunas obrigatórias. Faltam: {', '.join(colunas_faltantes)}")
        else:
            with st.spinner("Processando lote..."):
                comparacoes = []
                for nome, modelo in modelos.items():
                    predicoes = modelo.predict(df_lote[colunas_necessarias])
                    probabilidades = modelo.predict_proba(
                        df_lote[colunas_necessarias]
                    )[:, 1]
                    df_lote[f'{nome} — classificação'] = predicoes
                    df_lote[f'{nome} — probabilidade'] = (
                        probabilidades * 100
                    ).round(1)
                    comparacoes.append({
                        'Modelo': nome,
                        'Alunos em risco': int(predicoes.sum()),
                        'Taxa de risco': predicoes.mean(),
                        'Probabilidade média': probabilidades.mean(),
                    })

                resumo_modelos = pd.DataFrame(comparacoes)
                st.subheader("Comparação automática dos três modelos")
                st.dataframe(
                    resumo_modelos.style.format({
                        'Taxa de risco': '{:.1%}',
                        'Probabilidade média': '{:.1%}',
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
                st.bar_chart(
                    resumo_modelos.set_index('Modelo')[
                        ['Taxa de risco', 'Probabilidade média']
                    ],
                    y_label='Índice de evasão',
                )
                st.markdown("---")
                st.subheader("📋 Resultado por aluno")
                st.dataframe(df_lote, use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("👆 Aguardando o envio da planilha. Você pode testar previsões individuais na barra lateral.")