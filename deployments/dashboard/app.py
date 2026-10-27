import streamlit as st
import pandas as pd
import plotly.express as px
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
from model_wrapper import GBGridSearchWrapper

# Modelos antigos podem referenciar esta classe como __main__.GBGridSearchWrapper.
sys.modules['__main__'].GBGridSearchWrapper = GBGridSearchWrapper

# --- Configuração da Página e Identidade Visual ---
st.set_page_config(page_title="Predikt IFMS | Evasão Zero", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    :root { --ink: #17213d; --muted: #64708a; --cyan: #18b8d5; --orange: #ff9f43; }
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,.9); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #17213d 0%, #24345d 100%); }
    [data-testid="stSidebar"] * { color: #f8fbff !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] input { background: #25365f; border-color: #61729e; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -.03em; }
    .hero { padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(115deg, #17213d 0%, #2c4c83 58%, #18b8d5 100%);
        box-shadow: 0 16px 35px rgba(23,33,61,.18); margin-bottom: 1.4rem; }
    .hero h1 { color: white; font-size: 2.5rem; margin: 0; }
    .hero p { color: #dce8ff; font-size: 1.05rem; margin: .55rem 0 0; }
    .eyebrow { color: #8eeafa; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; }
    .metric-card { background: white; border: 1px solid #e6eaf2; border-radius: 18px;
        padding: 1.1rem 1.25rem; min-height: 122px; overflow: hidden;
        box-shadow: 0 7px 22px rgba(23,33,61,.06); }
    .metric-label { color: var(--muted); font-size: .85rem; }
    .metric-value { color: var(--ink); font-size: clamp(1.25rem, 2.1vw, 1.9rem);
        font-weight: 800; margin-top: .2rem; white-space: nowrap; }
    .metric-accent { color: #ff7b54; font-size: .8rem; font-weight: 700; }
    .section-note { color: var(--muted); margin-top: -.5rem; }
    .stButton>button { background: linear-gradient(90deg, #18b8d5, #2574d9); color: white;
        border: 0; border-radius: 10px; font-weight: 700; }
    .stButton>button:hover { filter: brightness(1.08); color: white; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e6eaf2;
        padding: 1rem; border-radius: 16px; }
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

POLICIES = {
    'Pé-de-Meia': {
        'description': 'Incentivo financeiro-educacional para permanência e conclusão no ensino médio público.',
        'link': 'https://www.gov.br/mec/pt-br/pe-de-meia',
    },
    'Auxílio permanência': {
        'description': 'Apoio socioeconômico para despesas que dificultam a permanência no curso.',
        'link': 'https://www.ifms.edu.br/',
    },
    'Alimentação/refeição': {
        'description': 'Acesso a refeição, auxílio alimentação ou restaurante estudantil, conforme oferta local.',
        'link': 'https://www.ifms.edu.br/',
    },
    'Auxílio transporte': {
        'description': 'Apoio para deslocamento entre a residência e o campus, conforme edital.',
        'link': 'https://www.ifms.edu.br/',
    },
    'Apoio psicossocial': {
        'description': 'Acolhimento e acompanhamento pela assistência estudantil e equipe multiprofissional.',
        'link': 'https://www.ifms.edu.br/',
    },
    'Apoio pedagógico': {
        'description': 'Monitoria, reforço, acompanhamento acadêmico e orientação para organização dos estudos.',
        'link': 'https://www.ifms.edu.br/',
    },
}


def politicas_recomendadas(perfil):
    """Sugere encaminhamentos; a elegibilidade deve ser confirmada pela instituição."""
    renda_baixa = perfil['renda_total'] in {
        'ate 1 salario minimo',
        'entre 1 e 2 salarios minimos',
        'nao informado',
    }
    trabalha = perfil['trabalha_atualmente'] != 'nao'
    solicitou_auxilio = perfil['recebe_auxilio_da_insti'] != 'nao'
    recomendadas = []

    if renda_baixa or solicitou_auxilio:
        recomendadas.extend(['Auxílio permanência', 'Alimentação/refeição'])
    if trabalha:
        recomendadas.extend(['Auxílio permanência', 'Apoio pedagógico'])
    if perfil['cidade'] != 'Jardim':
        recomendadas.append('Auxílio transporte')
    if perfil['idade'] in {'15 a 18 anos', 'menos de 15 anos'}:
        recomendadas.append('Pé-de-Meia')
    recomendadas.extend(['Apoio psicossocial', 'Apoio pedagógico'])
    return list(dict.fromkeys(recomendadas))


def analisar_base(df, modelos):
    """Executa os três modelos e acrescenta encaminhamentos por perfil."""
    df_resultado = df[FEATURE_COLUMNS].copy()
    comparacoes = []
    probabilidades = {}

    for nome, modelo in modelos.items():
        predicoes = modelo.predict(df_resultado)
        proba = modelo.predict_proba(df_resultado)[:, 1]
        probabilidades[nome] = proba
        df_resultado[f'{nome} — classificação'] = predicoes
        df_resultado[f'{nome} — probabilidade'] = (proba * 100).round(1)
        comparacoes.append({
            'Modelo': nome,
            'Alunos em risco': int(predicoes.sum()),
            'Taxa de risco': predicoes.mean(),
            'Probabilidade média': proba.mean(),
        })

    df_resultado['Políticas sugeridas'] = df_resultado.apply(
        lambda perfil: ', '.join(politicas_recomendadas(perfil)),
        axis=1,
    )
    return df_resultado, pd.DataFrame(comparacoes), probabilidades

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
logo_path = os.path.join(_PROJECT_ROOT, 'assets', 'logo.jpg')
st.sidebar.image(logo_path, use_container_width=True)
st.sidebar.markdown("## 🔍 Simulador de risco")
st.sidebar.caption("Monte um perfil e compare os três modelos em tempo real.")

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

# --- ÁREA PRINCIPAL: Análise e monitoramento ---
risco_declarado = int(df_base.get('risco_evasao', pd.Series(dtype=int)).sum())
taxa_declarada = risco_declarado / len(df_base) if len(df_base) else 0
st.markdown(
    '<div class="hero"><div class="eyebrow">IFMS • inteligência para permanência</div>'
    '<h1>Evasão Zero</h1>'
    '<p>Visão integrada de risco, comparação de modelos e encaminhamentos para apoiar cada estudante.</p>'
    '</div>',
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
for coluna, titulo, valor, detalhe in [
    (kpi1, 'Estudantes analisados', f'{len(df_base):,}', 'base ativa'),
    (kpi2, 'Risco declarado', f'{risco_declarado:,}', f'{taxa_declarada:.1%} da base'),
    (kpi3, 'Modelos em operação', '03', 'análise simultânea'),
    (kpi4, 'Status da IA', 'ONLINE', 'sincronização ativa'),
]:
    coluna.markdown(
        f'<div class="metric-card"><div class="metric-label">{titulo}</div>'
        f'<div class="metric-value">{valor}</div><div class="metric-accent">{detalhe}</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("Usar outra turma (opcional)"):
    arquivo_upload = st.file_uploader("Upload de planilha CSV", type=["csv"])

if arquivo_upload is not None:
    try:
        df_lote = pd.read_csv(arquivo_upload)
        colunas_faltantes = [col for col in FEATURE_COLUMNS if col not in df_lote.columns]
        
        if colunas_faltantes:
            st.error(f"A planilha enviada não possui as colunas obrigatórias. Faltam: {', '.join(colunas_faltantes)}")
        else:
            df_base = df_lote
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

with st.spinner("Analisando a base nos três modelos..."):
    df_resultado, resumo_modelos, probabilidades = analisar_base(df_base, modelos)

aba_visao, aba_modelos, aba_politicas, aba_detalhes = st.tabs([
    '📊 Visão geral', '🤖 Modelos', '🤝 Políticas públicas', '📋 Estudantes',
])

with aba_visao:
    st.subheader("Panorama de permanência estudantil")
    st.markdown(
        "Explore os sinais mais relevantes da base. Passe o mouse nos gráficos para ver detalhes.",
        help="As previsões são estimativas e não substituem a avaliação da equipe de assistência estudantil.",
    )
    vis1, vis2 = st.columns(2)
    curso_df = df_resultado.assign(
        Curso=df_resultado['curso'].str.title(),
        **{'Risco médio': df_resultado.filter(like='— probabilidade').mean(axis=1)},
    ).groupby('Curso', as_index=False)['Risco médio'].mean()
    renda_df = df_resultado.assign(
        Renda=df_resultado['renda_total'].str.replace('salarios minimos', 'salários mínimos'),
        **{'Risco médio': df_resultado.filter(like='— probabilidade').mean(axis=1)},
    ).groupby('Renda', as_index=False)['Risco médio'].mean()
    with vis1:
        fig = px.bar(curso_df, x='Curso', y='Risco médio', color='Risco médio',
                     color_continuous_scale=['#b5eff5', '#ff9f43'],
                     title='Índice médio por curso', text_auto='.1f')
        fig.update_layout(template='plotly_white', yaxis_title='Probabilidade (%)',
                          xaxis_title='', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with vis2:
        fig = px.bar(renda_df, x='Renda', y='Risco médio', color='Risco médio',
                     color_continuous_scale=['#b5eff5', '#ff9f43'],
                     title='Índice médio por faixa de renda', text_auto='.1f')
        fig.update_layout(template='plotly_white', yaxis_title='Probabilidade (%)',
                          xaxis_title='', coloraxis_showscale=False)
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

with aba_modelos:
    st.subheader("Os três modelos, lado a lado")
    grafico_modelos = resumo_modelos.melt(
        id_vars='Modelo',
        value_vars=['Taxa de risco', 'Probabilidade média'],
        var_name='Indicador',
        value_name='Valor',
    )
    fig = px.bar(grafico_modelos, x='Modelo', y='Valor', color='Indicador',
                 barmode='group', text_auto='.1%', title='Consenso e divergência entre modelos',
                 color_discrete_sequence=['#18b8d5', '#ff9f43'])
    fig.update_layout(template='plotly_white', yaxis_tickformat='.0%', xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        resumo_modelos.style.format({'Taxa de risco': '{:.1%}', 'Probabilidade média': '{:.1%}'}),
        hide_index=True, use_container_width=True,
    )

with aba_politicas:
    st.subheader("Encaminhamentos de permanência")
    st.caption("Sugestões baseadas nos sinais cadastrais. A elegibilidade deve ser confirmada pela instituição e pelos programas oficiais.")
    politicas_df = pd.DataFrame([
        {
            'Política/apoio': nome,
            'Indicações': int(df_resultado['Políticas sugeridas'].str.contains(nome, regex=False).sum()),
            'Descrição': dados['description'],
            'Fonte': dados['link'],
        }
        for nome, dados in POLICIES.items()
    ]).sort_values('Indicações', ascending=False)
    fig = px.bar(politicas_df.sort_values('Indicações'), x='Indicações', y='Política/apoio',
                 orientation='h', text='Indicações', title='Apoios mais indicados na base',
                 color='Indicações', color_continuous_scale=['#b5eff5', '#2574d9'])
    fig.update_layout(template='plotly_white', coloraxis_showscale=False, yaxis_title='')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(politicas_df, column_config={'Fonte': st.column_config.LinkColumn('Fonte oficial')},
                 hide_index=True, use_container_width=True)

with aba_detalhes:
    st.subheader("Lista de estudantes analisados")
    st.caption("Use a busca e a ordenação da tabela para localizar casos prioritários.")
    colunas_exibicao = FEATURE_COLUMNS + [
        coluna for coluna in df_resultado.columns
        if 'classificação' in coluna or 'probabilidade' in coluna
    ] + ['Políticas sugeridas']
    st.dataframe(df_resultado[colunas_exibicao], hide_index=True, use_container_width=True,
                 height=560)