# Resumo da robustez estatística

> Este documento resume medições observacionais. Não representa validação
> externa, não prova causalidade e não deve ser usado isoladamente para
> decisões individuais sobre estudantes.

## 1. Resultados básicos

| modelo | cenario | metrica | media | desvio_padrao | ic95_inferior | ic95_superior | n_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost (SOTA) | A_Real | roc_auc | 0.5824801587301588 | 0.0742647359640531 | 0.5243650793650794 | 0.640595238095238 | 5 |
| CatBoost (SOTA) | A_Real | f1_macro | 0.5774461996415304 | 0.0301377869436178 | 0.5533708655876144 | 0.5996374757244323 | 5 |
| CatBoost (SOTA) | A_Real | recall | 0.5628571428571428 | 0.1407140842799299 | 0.4647619047619047 | 0.6847619047619047 | 5 |
| CatBoost (SOTA) | A_Real | precision | 0.4631031543052003 | 0.037556180234272 | 0.4297502131287297 | 0.4874509803921569 | 5 |
| CatBoost (SOTA) | A_Real | brier | 0.2458590062558333 | 0.0123879937623895 | 0.2362533362734453 | 0.2554646762382211 | 5 |
| CatBoost (SOTA) | A_Real | bss | -0.0474751877624114 | 0.055122582698946 | -0.0899275115306722 | -0.0044189034406978 | 5 |
| CatBoost (SOTA) | B_SDV1000 | roc_auc | 0.5715079365079365 | 0.0908332163139732 | 0.4957142857142857 | 0.6476984126984127 | 5 |
| CatBoost (SOTA) | B_SDV1000 | f1_macro | 0.5516150282879189 | 0.0915210525714241 | 0.4774206349206349 | 0.6116726319745547 | 5 |
| CatBoost (SOTA) | B_SDV1000 | recall | 0.5323809523809524 | 0.1779321891777264 | 0.3847619047619047 | 0.6619047619047619 | 5 |
| CatBoost (SOTA) | B_SDV1000 | precision | 0.4320728291316526 | 0.1078545048394189 | 0.3404761904761905 | 0.4985994397759103 | 5 |
| CatBoost (SOTA) | B_SDV1000 | brier | 0.2470024750671851 | 0.0079737778544825 | 0.2412072129887577 | 0.2533199259116146 | 5 |
| CatBoost (SOTA) | B_SDV1000 | bss | -0.0467753102572214 | 0.0308355810663279 | -0.0696446844679937 | -0.0237806012703138 | 5 |
| Dummy (prior) | A_Real | roc_auc | 0.5 | 0.0 | 0.5 | 0.5 | 5 |
| Dummy (prior) | A_Real | f1_macro | 0.3841157834101382 | 0.0045077031843762 | 0.3809523809523809 | 0.3879848310291859 | 5 |
| Dummy (prior) | A_Real | recall | 0.0 | 0.0 | 0.0 | 0.0 | 5 |
| Dummy (prior) | A_Real | precision | 0.0 | 0.0 | 0.0 | 0.0 | 5 |
| Dummy (prior) | A_Real | brier | 0.2347498233198625 | 0.002906615909125 | 0.2322643781372269 | 0.2367950052029136 | 5 |
| Dummy (prior) | A_Real | bss | 0.0 | 0.0 | 0.0 | 0.0 | 5 |
| Dummy (prior) | B_SDV1000 | roc_auc | 0.5 | 0.0 | 0.5 | 0.5 | 5 |
| Dummy (prior) | B_SDV1000 | f1_macro | 0.3841157834101382 | 0.0045077031843762 | 0.3809523809523809 | 0.3879848310291859 | 5 |

A evidência deve ser considerada exploratória, pois os intervalos foram estimados com o número de repetições disponível. A significância formal deve ser consultada nos testes pareados com Holm.

## 2. Comparações pareadas

| cenario | metrica | modelo_a | modelo_b | diferenca_media_a_menos_b | p_valor_holm | significativo_05 |
| --- | --- | --- | --- | --- | --- | --- |
| A_Real | roc_auc | Dummy (prior) | Dummy (stratified) | -0.0073095238095237 | 1.0 | False |
| A_Real | roc_auc | Dummy (prior) | LogisticRegression | -0.0800595238095238 | 1.0 | False |
| A_Real | roc_auc | Dummy (prior) | GradientBoosting (Baseline) | -0.1030039682539682 | 1.0 | False |
| A_Real | roc_auc | Dummy (prior) | CatBoost (SOTA) | -0.0824801587301587 | 1.0 | False |
| A_Real | roc_auc | Dummy (prior) | TabPFN / HistGB (SOTA) | -0.1031587301587301 | 0.9375 | False |
| A_Real | roc_auc | Dummy (stratified) | LogisticRegression | -0.07275 | 1.0 | False |
| A_Real | roc_auc | Dummy (stratified) | GradientBoosting (Baseline) | -0.0956944444444445 | 1.0 | False |
| A_Real | roc_auc | Dummy (stratified) | CatBoost (SOTA) | -0.0751706349206349 | 1.0 | False |
| A_Real | roc_auc | Dummy (stratified) | TabPFN / HistGB (SOTA) | -0.0958492063492063 | 1.0 | False |
| A_Real | roc_auc | LogisticRegression | GradientBoosting (Baseline) | -0.0229444444444444 | 1.0 | False |
| A_Real | roc_auc | LogisticRegression | CatBoost (SOTA) | -0.0024206349206349 | 1.0 | False |
| A_Real | roc_auc | LogisticRegression | TabPFN / HistGB (SOTA) | -0.0230992063492063 | 1.0 | False |
| A_Real | roc_auc | GradientBoosting (Baseline) | CatBoost (SOTA) | 0.0205238095238095 | 1.0 | False |
| A_Real | roc_auc | GradientBoosting (Baseline) | TabPFN / HistGB (SOTA) | -0.0001547619047618 | 1.0 | False |
| A_Real | roc_auc | CatBoost (SOTA) | TabPFN / HistGB (SOTA) | -0.0206785714285714 | 1.0 | False |
| A_Real | f1_macro | Dummy (prior) | Dummy (stratified) | -0.1179669451820243 | 0.9375 | False |
| A_Real | f1_macro | Dummy (prior) | LogisticRegression | -0.1681906209934528 | 0.9375 | False |
| A_Real | f1_macro | Dummy (prior) | GradientBoosting (Baseline) | -0.1883769082159223 | 0.9375 | False |
| A_Real | f1_macro | Dummy (prior) | CatBoost (SOTA) | -0.1933304162313923 | 0.9375 | False |
| A_Real | f1_macro | Dummy (prior) | TabPFN / HistGB (SOTA) | -0.1994284334309322 | 0.9375 | False |

A ausência de significância não prova equivalência; indica apenas que
não foi observada evidência suficiente neste desenho e tamanho amostral.

## 3. Baseline trivial e Brier Skill Score

O baseline trivial é o classificador que prevê a prevalência. O Brier
Skill Score positivo indica melhoria relativa ao baseline no conjunto
avaliado. É necessário verificar simultaneamente o intervalo do BSS e
a incerteza da AUC; este resumo não transforma uma média positiva em
superioridade estatisticamente demonstrada.

Nesta execução, houve IC de AUC inteiramente acima de 0,5 e não houve IC de BSS inteiramente acima de 0. Isso não substitui validação externa nem uma análise com mais repetições.

| modelo | cenario | metrica | media | ic95_inferior | ic95_superior |
| --- | --- | --- | --- | --- | --- |
| CatBoost (SOTA) | A_Real | roc_auc | 0.5824801587301588 | 0.5243650793650794 | 0.640595238095238 |
| CatBoost (SOTA) | A_Real | bss | -0.0474751877624114 | -0.0899275115306722 | -0.0044189034406978 |
| CatBoost (SOTA) | B_SDV1000 | roc_auc | 0.5715079365079365 | 0.4957142857142857 | 0.6476984126984127 |
| CatBoost (SOTA) | B_SDV1000 | bss | -0.0467753102572214 | -0.0696446844679937 | -0.0237806012703138 |
| Dummy (prior) | A_Real | roc_auc | 0.5 | 0.5 | 0.5 |
| Dummy (prior) | A_Real | bss | 0.0 | 0.0 | 0.0 |
| Dummy (prior) | B_SDV1000 | roc_auc | 0.5 | 0.5 | 0.5 |
| Dummy (prior) | B_SDV1000 | bss | 0.0 | 0.0 | 0.0 |
| Dummy (stratified) | A_Real | roc_auc | 0.5073095238095238 | 0.4770238095238094 | 0.5409761904761905 |
| Dummy (stratified) | A_Real | bss | -0.8667301827656011 | -0.983414189221712 | -0.7412886445133503 |
| Dummy (stratified) | B_SDV1000 | roc_auc | 0.5051190476190476 | 0.4823809523809522 | 0.5266666666666666 |
| Dummy (stratified) | B_SDV1000 | bss | -0.9015919432550572 | -0.9834613874360876 | -0.7981028985057232 |
| GradientBoosting (Baseline) | A_Real | roc_auc | 0.6030039682539682 | 0.5339920634920635 | 0.6622222222222223 |
| GradientBoosting (Baseline) | A_Real | bss | -0.1651904792212331 | -0.4356361358717744 | 0.0467646143593351 |
| GradientBoosting (Baseline) | B_SDV1000 | roc_auc | 0.5610198412698413 | 0.4572261904761904 | 0.6629166666666667 |
| GradientBoosting (Baseline) | B_SDV1000 | bss | -0.0088246719193662 | -0.0857856892390692 | 0.057392291152943 |
| LogisticRegression | A_Real | roc_auc | 0.5800595238095239 | 0.5271428571428571 | 0.628968253968254 |
| LogisticRegression | A_Real | bss | -0.0581612119125476 | -0.1495678921702468 | 0.0224875256063457 |
| LogisticRegression | B_SDV1000 | roc_auc | 0.5578293650793651 | 0.4693214285714286 | 0.62 |
| LogisticRegression | B_SDV1000 | bss | -0.0109309623526887 | -0.0688803782829926 | 0.0417783999509715 |

## 4. Limiar de decisão

| modelo | cenario | estrategia | limiar | f1_macro | recall | precision |
| --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting (Baseline) | A_Real | padrao_0_5 | 0.5 | 0.6657142857142857 | 0.5333333333333333 | 0.6153846153846154 |
| GradientBoosting (Baseline) | A_Real | otimizado_treino | 0.4299868749367577 | 0.6208333333333333 | 0.5333333333333333 | 0.5333333333333333 |
| CatBoost (SOTA) | A_Real | padrao_0_5 | 0.5 | 0.606060606060606 | 0.6 | 0.5 |
| CatBoost (SOTA) | A_Real | otimizado_treino | 0.4940940781784012 | 0.5828877005347594 | 0.6 | 0.4736842105263157 |
| TabPFN / HistGB (SOTA) | A_Real | padrao_0_5 | 0.5 | 0.507641196013289 | 0.5333333333333333 | 0.4 |
| TabPFN / HistGB (SOTA) | A_Real | otimizado_treino | 0.4110645528680784 | 0.5641025641025641 | 0.7333333333333333 | 0.4583333333333333 |
| GradientBoosting (Baseline) | B_SDV1000 | padrao_0_5 | 0.5 | 0.6762264150943397 | 0.4666666666666667 | 0.7 |
| GradientBoosting (Baseline) | B_SDV1000 | otimizado_treino | 0.3748434903693913 | 0.5629531970995385 | 0.6666666666666666 | 0.4545454545454545 |
| CatBoost (SOTA) | B_SDV1000 | padrao_0_5 | 0.5 | 0.5629531970995385 | 0.6666666666666666 | 0.4545454545454545 |
| CatBoost (SOTA) | B_SDV1000 | otimizado_treino | 0.5168053797282479 | 0.5629531970995385 | 0.6666666666666666 | 0.4545454545454545 |
| TabPFN / HistGB (SOTA) | B_SDV1000 | padrao_0_5 | 0.5 | 0.5629531970995385 | 0.6666666666666666 | 0.4545454545454545 |
| TabPFN / HistGB (SOTA) | B_SDV1000 | otimizado_treino | 0.5173977134840414 | 0.5873015873015873 | 0.6666666666666666 | 0.4761904761904761 |
| GradientBoosting (Baseline) | A_Real | padrao_0_5 | 0.5 | 0.5764411027568922 | 0.2666666666666666 | 0.6666666666666666 |
| GradientBoosting (Baseline) | A_Real | otimizado_treino | 0.3882045716762822 | 0.5760869565217391 | 0.5333333333333333 | 0.4705882352941176 |
| CatBoost (SOTA) | A_Real | padrao_0_5 | 0.5 | 0.6112956810631229 | 0.6666666666666666 | 0.5 |
| CatBoost (SOTA) | A_Real | otimizado_treino | 0.5009136367907763 | 0.6112956810631229 | 0.6666666666666666 | 0.5 |
| TabPFN / HistGB (SOTA) | A_Real | padrao_0_5 | 0.5 | 0.5984900480439259 | 0.5333333333333333 | 0.5 |
| TabPFN / HistGB (SOTA) | A_Real | otimizado_treino | 0.4783785138149219 | 0.6350267379679144 | 0.6666666666666666 | 0.5263157894736842 |
| GradientBoosting (Baseline) | B_SDV1000 | padrao_0_5 | 0.5 | 0.5666666666666667 | 0.2 | 1.0 |
| GradientBoosting (Baseline) | B_SDV1000 | otimizado_treino | 0.411798156590723 | 0.5584905660377358 | 0.3333333333333333 | 0.5 |

O limiar otimizado foi escolhido dentro do treino. Diferenças no fold
de validação representam trade-offs entre F1, recall e precisão; não
são evidência de que um limiar seja universalmente melhor.

## 5. Calibração

| modelo | cenario | metodo | etapa | brier |
| --- | --- | --- | --- | --- |
| GradientBoosting (Baseline) | A_Real | sigmoid | antes | 0.2195038352425787 |
| GradientBoosting (Baseline) | A_Real | sigmoid | depois | 0.2376914254163054 |
| GradientBoosting (Baseline) | A_Real | isotonic | antes | 0.2195038352425787 |
| GradientBoosting (Baseline) | A_Real | isotonic | depois | 0.246506875599191 |
| CatBoost (SOTA) | A_Real | sigmoid | antes | 0.2566894949621708 |
| CatBoost (SOTA) | A_Real | sigmoid | depois | 0.2436097166290604 |
| CatBoost (SOTA) | A_Real | isotonic | antes | 0.2566894949621708 |
| CatBoost (SOTA) | A_Real | isotonic | depois | 0.2381796938212398 |
| TabPFN / HistGB (SOTA) | A_Real | sigmoid | antes | 0.2401026917346291 |
| TabPFN / HistGB (SOTA) | A_Real | sigmoid | depois | 0.2376739477640345 |
| TabPFN / HistGB (SOTA) | A_Real | isotonic | antes | 0.2401026917346291 |
| TabPFN / HistGB (SOTA) | A_Real | isotonic | depois | 0.2398088989627585 |
| GradientBoosting (Baseline) | B_SDV1000 | sigmoid | antes | 0.2386153582852573 |
| GradientBoosting (Baseline) | B_SDV1000 | sigmoid | depois | 0.2498047113115393 |
| GradientBoosting (Baseline) | B_SDV1000 | isotonic | antes | 0.2386153582852573 |
| GradientBoosting (Baseline) | B_SDV1000 | isotonic | depois | 0.2671448946078568 |
| CatBoost (SOTA) | B_SDV1000 | sigmoid | antes | 0.2439097403579958 |
| CatBoost (SOTA) | B_SDV1000 | sigmoid | depois | 0.2296189832980535 |
| CatBoost (SOTA) | B_SDV1000 | isotonic | antes | 0.2439097403579958 |
| CatBoost (SOTA) | B_SDV1000 | isotonic | depois | 0.2603986821578666 |

Brier menor é preferível para calibração. Com amostras pequenas, sigmoid
e isotonic podem aumentar a variância; a direção observada deve ser
confirmada em mais repetições.

## 6. Sensibilidade à síntese

| modelo | n_sinteticos | roc_auc | recall |
| --- | --- | --- | --- |
| GradientBoosting (Baseline) | 0 | 0.6597222222222222 | 0.5333333333333333 |
| CatBoost (SOTA) | 0 | 0.5236111111111111 | 0.6 |
| TabPFN / HistGB (SOTA) | 0 | 0.5972222222222222 | 0.5333333333333333 |
| GradientBoosting (Baseline) | 250 | 0.5958333333333333 | 0.4 |
| CatBoost (SOTA) | 250 | 0.5625 | 0.6666666666666666 |
| TabPFN / HistGB (SOTA) | 250 | 0.5736111111111111 | 0.5333333333333333 |
| GradientBoosting (Baseline) | 500 | 0.6013888888888889 | 0.3333333333333333 |
| CatBoost (SOTA) | 500 | 0.6333333333333334 | 0.6666666666666666 |
| TabPFN / HistGB (SOTA) | 500 | 0.5986111111111111 | 0.6666666666666666 |
| GradientBoosting (Baseline) | 1000 | 0.5722222222222222 | 0.4666666666666667 |
| CatBoost (SOTA) | 1000 | 0.6236111111111111 | 0.6666666666666666 |
| TabPFN / HistGB (SOTA) | 1000 | 0.575 | 0.6666666666666666 |
| GradientBoosting (Baseline) | 2000 | 0.6625 | 0.5333333333333333 |
| CatBoost (SOTA) | 2000 | 0.6055555555555555 | 0.6666666666666666 |
| TabPFN / HistGB (SOTA) | 2000 | 0.6194444444444445 | 0.6666666666666666 |
| GradientBoosting (Baseline) | 0 | 0.5416666666666667 | 0.2666666666666666 |
| CatBoost (SOTA) | 0 | 0.7833333333333333 | 0.6666666666666666 |
| TabPFN / HistGB (SOTA) | 0 | 0.5958333333333333 | 0.5333333333333333 |
| GradientBoosting (Baseline) | 250 | 0.6513888888888889 | 0.1333333333333333 |
| CatBoost (SOTA) | 250 | 0.5902777777777778 | 0.4 |

- **CatBoost (SOTA)**: parece piorar na média observada; isso não demonstra causalidade.
- **GradientBoosting (Baseline)**: parece piorar na média observada; isso não demonstra causalidade.
- **TabPFN / HistGB (SOTA)**: parece piorar na média observada; isso não demonstra causalidade.

## 7. Subgrupos

| modelo | cenario | variavel_grupo | categoria | n | confiavel | roc_auc | recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting (Baseline) | A_Real | curso | computacao | 12 | False | 0.5555555555555556 | 0.6666666666666666 |
| GradientBoosting (Baseline) | A_Real | curso | informatica | 9 | False | 0.95 | 1.0 |
| GradientBoosting (Baseline) | A_Real | curso | edificacoes | 16 | False | 0.7000000000000001 | 0.0 |
| GradientBoosting (Baseline) | A_Real | curso | arquitetura | 2 | False | NaN | 0.0 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | entre 1 e 2 salarios minimos | 12 | False | 0.6111111111111112 | 0.5 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | mais de 6 salarios minimos | 2 | False | 1.0 | 1.0 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | entre 4 e 6 salarios minimos | 7 | False | 0.9 | 1.0 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | nao informado | 4 | False | 0.0 | 0.0 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | entre 2 e 4 salarios minimos | 12 | False | 0.7285714285714286 | 0.4 |
| GradientBoosting (Baseline) | A_Real | faixa_renda | ate 1 salario minimo | 2 | False | NaN | 0.0 |
| GradientBoosting (Baseline) | A_Real | trabalha_atualmente | sim, em periodo integral | 7 | False | 0.75 | 1.0 |
| GradientBoosting (Baseline) | A_Real | trabalha_atualmente | sim, em meio periodo | 7 | False | 0.9166666666666669 | 0.75 |
| GradientBoosting (Baseline) | A_Real | trabalha_atualmente | nao, mas estou procurando | 9 | False | 0.5555555555555556 | 0.3333333333333333 |
| GradientBoosting (Baseline) | A_Real | trabalha_atualmente | nao | 16 | False | 0.6272727272727273 | 0.2 |
| CatBoost (SOTA) | A_Real | curso | computacao | 12 | False | 0.5555555555555556 | 0.8333333333333334 |
| CatBoost (SOTA) | A_Real | curso | informatica | 9 | False | 0.4 | 1.0 |
| CatBoost (SOTA) | A_Real | curso | edificacoes | 16 | False | 0.1 | 0.0 |
| CatBoost (SOTA) | A_Real | curso | arquitetura | 2 | False | NaN | 0.0 |
| CatBoost (SOTA) | A_Real | faixa_renda | entre 1 e 2 salarios minimos | 12 | False | 0.6111111111111112 | 0.6666666666666666 |
| CatBoost (SOTA) | A_Real | faixa_renda | mais de 6 salarios minimos | 2 | False | 0.0 | 1.0 |

Subgrupos com menos de 20 registros ou apenas uma classe foram marcados
`confiavel=False`. AUC é indefinida quando há uma única classe. Mesmo
subgrupos marcados como confiáveis continuam sujeitos a incerteza.

## 8. Rastreabilidade

| modelo | implementacao | python | numpy | pandas | sklearn | catboost | shap | sdv | tabpfn | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting (Baseline) | GBGridSearchWrapper | 3.11.5 | 2.4.6 | 2.3.3 | 1.9.1 | 1.2.10 | 0.49.1 | 2.0.0 | não instalado | 42 |
| CatBoost (SOTA) | CatBoostWrapper | 3.11.5 | 2.4.6 | 2.3.3 | 1.9.1 | 1.2.10 | 0.49.1 | 2.0.0 | não instalado | 42 |
| TabPFN / HistGB (SOTA) | fallback HistGradientBoosting | 3.11.5 | 2.4.6 | 2.3.3 | 1.9.1 | 1.2.10 | 0.49.1 | 2.0.0 | não instalado | 42 |

## 9. Limitações e interpretação final

- A base é pequena e a validação é interna; não há garantia de generalização.
- Os dados sintéticos não substituem dados reais e podem alterar a distribuição.
- A comparação entre modelos está sujeita à variância dos folds e dos ajustes.
- Métricas agregadas não devem ser usadas para rotular ou punir estudantes.
- A implementação real do terceiro modelo deve ser lida na tabela de rastreabilidade.
