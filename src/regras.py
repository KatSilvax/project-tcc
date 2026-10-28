"""Regras puras compartilhadas pelo dashboard e pelos testes."""

FEATURE_COLUMNS = [
    'curso',
    'periodo',
    'idade',
    'cidade',
    'renda_total',
    'trabalha_atualmente',
    'recebe_auxilio_da_insti',
]

MODEL_FILES = {
    'GradientBoosting (Baseline)': 'gb_baseline_ifms.joblib',
    'CatBoost (SOTA)': 'catboost_sota_ifms.joblib',
    'TabPFN / HistGB (SOTA)': 'tabpfn_sota_ifms.joblib',
}


def criar_alvo(valor):
    """Converte a resposta de desistência no alvo binário do projeto."""
    return 0 if isinstance(valor, str) and 'nao nunca' in valor.lower() else 1


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


def colunas_faltantes(colunas):
    """Retorna as entradas obrigatórias ausentes em uma planilha."""
    return [coluna for coluna in FEATURE_COLUMNS if coluna not in colunas]
