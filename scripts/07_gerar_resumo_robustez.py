"""Gera um resumo cauteloso dos resultados de robustez já calculados."""

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / 'resultados_robustez_resumo.md'


def ler(nome):
    caminho = PROJECT_ROOT / nome
    return pd.read_csv(caminho) if caminho.exists() else pd.DataFrame()


def tabela(df, colunas, linhas=20):
    if df.empty:
        return '_Arquivo não disponível._'
    disponiveis = [coluna for coluna in colunas if coluna in df.columns]
    limitado = df[disponiveis].head(linhas).fillna('NaN')
    cabecalho = '| ' + ' | '.join(disponiveis) + ' |'
    separador = '| ' + ' | '.join(['---'] * len(disponiveis)) + ' |'
    corpo = [
        '| ' + ' | '.join(str(valor) for valor in linha) + ' |'
        for linha in limitado.itertuples(index=False, name=None)
    ]
    return '\n'.join([cabecalho, separador] + corpo)


def interpretar_basico(df):
    if df.empty:
        return 'Os resultados básicos não estão disponíveis.'
    conclusao = (
        'A evidência deve ser considerada exploratória, pois os intervalos '
        'foram estimados com o número de repetições disponível.'
    )
    if 'significativo_05' in df.columns:
        conclusao += ' A significância formal deve ser consultada nos testes pareados com Holm.'
    return conclusao


def interpretar_sintese(df):
    if df.empty:
        return 'A análise de síntese não está disponível.'
    resumo = df.groupby(['modelo', 'n_sinteticos'])[
        ['roc_auc', 'recall']
    ].mean()
    mensagens = []
    for modelo in resumo.index.get_level_values(0).unique():
        serie = resumo.loc[modelo]
        base = serie.loc[0]
        maior = serie.loc[serie['roc_auc'].idxmax()]
        menor = serie.loc[serie['roc_auc'].idxmin()]
        if maior['roc_auc'] > base['roc_auc'] and menor['roc_auc'] < base['roc_auc']:
            efeito = 'varia conforme a quantidade; não há ganho monotônico'
        elif maior['roc_auc'] > base['roc_auc']:
            efeito = 'parece ajudar na média observada'
        elif menor['roc_auc'] < base['roc_auc']:
            efeito = 'parece piorar na média observada'
        else:
            efeito = 'parece indiferente na média observada'
        mensagens.append(f'- **{modelo}**: {efeito}; isso não demonstra causalidade.')
    return '\n'.join(mensagens)


def interpretar_baseline_trivial(df):
    if df.empty:
        return 'Não foi possível avaliar o baseline trivial.'
    auc = df[df['metrica'] == 'roc_auc']
    bss = df[df['metrica'] == 'bss']
    auc_superior = auc[
        auc['ic95_inferior'].astype(float) > 0.5
    ] if not auc.empty else pd.DataFrame()
    bss_superior = bss[
        bss['ic95_inferior'].astype(float) > 0.0
    ] if not bss.empty else pd.DataFrame()
    if auc_superior.empty:
        texto_auc = 'não houve IC de AUC inteiramente acima de 0,5'
    else:
        texto_auc = 'houve IC de AUC inteiramente acima de 0,5'
    if bss_superior.empty:
        texto_bss = 'não houve IC de BSS inteiramente acima de 0'
    else:
        texto_bss = 'houve IC de BSS inteiramente acima de 0'
    return (
        f'Nesta execução, {texto_auc} e {texto_bss}. '
        'Isso não substitui validação externa nem uma análise com mais repetições.'
    )


def gerar():
    basico = ler('resultados_robustez_basica.csv')
    testes = ler('resultados_robustez_testes.csv')
    limiar = ler('resultados_robustez_limiar.csv')
    calibracao = ler('resultados_robustez_calibracao.csv')
    sintese = ler('resultados_robustez_sintese.csv')
    subgrupos = ler('resultados_robustez_subgrupos.csv')
    rastreabilidade = ler('resultados_robustez_rastreabilidade.csv')
    linhas = [
        '# Resumo da robustez estatística',
        '',
        '> Este documento resume medições observacionais. Não representa validação',
        '> externa, não prova causalidade e não deve ser usado isoladamente para',
        '> decisões individuais sobre estudantes.',
        '',
        '## 1. Resultados básicos',
        '',
        tabela(basico, [
            'modelo', 'cenario', 'metrica', 'media', 'desvio_padrao',
            'ic95_inferior', 'ic95_superior', 'n_folds',
        ]),
        '',
        interpretar_basico(testes),
        '',
        '## 2. Comparações pareadas',
        '',
        tabela(testes, [
            'cenario', 'metrica', 'modelo_a', 'modelo_b',
            'diferenca_media_a_menos_b', 'p_valor_holm', 'significativo_05',
        ]),
        '',
        'A ausência de significância não prova equivalência; indica apenas que',
        'não foi observada evidência suficiente neste desenho e tamanho amostral.',
        '',
        '## 3. Baseline trivial e Brier Skill Score',
        '',
        'O baseline trivial é o classificador que prevê a prevalência. O Brier',
        'Skill Score positivo indica melhoria relativa ao baseline no conjunto',
        'avaliado. É necessário verificar simultaneamente o intervalo do BSS e',
        'a incerteza da AUC; este resumo não transforma uma média positiva em',
        'superioridade estatisticamente demonstrada.',
        '',
        interpretar_baseline_trivial(basico),
        '',
        tabela(basico[basico.get('metrica', pd.Series()).isin(['roc_auc', 'bss'])] if not basico.empty else basico, [
            'modelo', 'cenario', 'metrica', 'media', 'ic95_inferior', 'ic95_superior',
        ]),
        '',
        '## 4. Limiar de decisão',
        '',
        tabela(limiar, [
            'modelo', 'cenario', 'estrategia', 'limiar',
            'f1_macro', 'recall', 'precision',
        ]),
        '',
        'O limiar otimizado foi escolhido dentro do treino. Diferenças no fold',
        'de validação representam trade-offs entre F1, recall e precisão; não',
        'são evidência de que um limiar seja universalmente melhor.',
        '',
        '## 5. Calibração',
        '',
        tabela(calibracao, [
            'modelo', 'cenario', 'metodo', 'etapa', 'brier',
        ]),
        '',
        'Brier menor é preferível para calibração. Com amostras pequenas, sigmoid',
        'e isotonic podem aumentar a variância; a direção observada deve ser',
        'confirmada em mais repetições.',
        '',
        '## 6. Sensibilidade à síntese',
        '',
        tabela(sintese, [
            'modelo', 'n_sinteticos', 'roc_auc', 'recall',
        ]),
        '',
        interpretar_sintese(sintese),
        '',
        '## 7. Subgrupos',
        '',
        tabela(subgrupos, [
            'modelo', 'cenario', 'variavel_grupo', 'categoria',
            'n', 'confiavel', 'roc_auc', 'recall',
        ]),
        '',
        'Subgrupos com menos de 20 registros ou apenas uma classe foram marcados',
        '`confiavel=False`. AUC é indefinida quando há uma única classe. Mesmo',
        'subgrupos marcados como confiáveis continuam sujeitos a incerteza.',
        '',
        '## 8. Rastreabilidade',
        '',
        tabela(rastreabilidade, [
            'modelo', 'implementacao', 'python', 'numpy', 'pandas',
            'sklearn', 'catboost', 'shap', 'sdv', 'tabpfn', 'seed',
        ]),
        '',
        '## 9. Limitações e interpretação final',
        '',
        '- A base é pequena e a validação é interna; não há garantia de generalização.',
        '- Os dados sintéticos não substituem dados reais e podem alterar a distribuição.',
        '- A comparação entre modelos está sujeita à variância dos folds e dos ajustes.',
        '- Métricas agregadas não devem ser usadas para rotular ou punir estudantes.',
        '- A implementação real do terceiro modelo deve ser lida na tabela de rastreabilidade.',
        '',
    ]
    if OUTPUT.exists():
        raise FileExistsError(f'{OUTPUT} já existe; nada será sobrescrito.')
    OUTPUT.write_text('\n'.join(linhas), encoding='utf-8')
    print(f'Resumo salvo em {OUTPUT}')


if __name__ == '__main__':
    gerar()
