# Projeto de IA — Classificação Bayesiana

Estrutura inicial do estudo dirigido, organizada conforme a
[`RFC-0001`](rfcs/RFC-0001-padroes-de-engenharia-e-governanca.md).

## Estado atual

A RFC-0001 está implementada **somente no nível de estrutura e governança**.
Os módulos e testes existem como pontos de extensão, mas ainda não contêm a
lógica matemática do pipeline. Essa lógica deve ser adicionada apenas pelas
RFCs específicas que definirem os respectivos contratos.

## Parâmetros congelados

| Parâmetro | Valor |
|---|---|
| Dataset | `data/raw/bank.csv` |
| Target | `y` |
| Classes | `no -> 0`, `yes -> 1` |
| Features | `age`, `duration`, `marital` |
| Teste | `20%` |
| Semente | `42` |
| Estratificação | `y` |
| Laplace | `alpha = 1.0` |
| Ordem das classes | `[0, 1]` |
| Logaritmo | natural |

## Estrutura

```text
.
├── data/raw/bank.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_analises_univariadas.ipynb
│   └── 03_avaliacao_final.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── rfcs/
│   └── RFC-0001-padroes-de-engenharia-e-governanca.md
├── src/
│   ├── config.py
│   ├── data.py
│   ├── distributions.py
│   ├── evaluation.py
│   ├── mixed_naive_bayes.py
│   ├── run_experiment.py
│   └── univariate.py
├── tests/
│   ├── test_data.py
│   ├── test_distributions.py
│   ├── test_evaluation.py
│   ├── test_mixed_naive_bayes.py
│   └── test_univariate.py
└── requirements.txt
```

## Responsabilidades

- `src/config.py`: parâmetros imutáveis e nomes de colunas.
- `src/data.py`: leitura, validação, seleção e divisão dos dados.
- `src/distributions.py`: ajuste de distribuições e log-densidades.
- `src/univariate.py`: análises e classificadores univariados.
- `src/mixed_naive_bayes.py`: classificador Naive Bayes misto.
- `src/evaluation.py`: matriz de confusão e métricas.
- `src/run_experiment.py`: orquestração, sem duplicação de fórmulas.
- `notebooks/`: explicação e inspeção; nunca a única implementação.

## Governança

Cada mudança deve partir de uma RFC aplicável, ser feita em branch curta,
incluir testes proporcionais ao comportamento implementado e passar por revisão
cruzada. Mudanças nos parâmetros congelados ou nas fórmulas exigem atualização
de RFC.

O conjunto de teste não pode participar do ajuste de parâmetros, escolha de
distribuição ou tuning. O projeto não deve incluir preditores além dos três
definidos, técnicas de reamostragem, classificadores prontos como implementação
principal nem infraestrutura fora do escopo acadêmico.

## Dados

`bank.csv` é a amostra reduzida de 4.521 observações do conjunto
[Bank Marketing da UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing),
distribuído sob licença CC BY 4.0.
