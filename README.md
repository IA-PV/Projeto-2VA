# Projeto de IA — Classificação Bayesiana

Estudo dirigido organizado conforme a
[`RFC-0001`](rfcs/RFC-0001-padroes-de-engenharia-e-governanca.md).

## Estado atual

A ingestão, a validação e o split estratificado da RFC-0002 estão implementados.
A [RFC-0003](rfcs/RFC-0003-modelagem-probabilistica.md) acrescenta os priors,
as distribuições condicionais por classe e a comparação Gamma × Exponencial
exclusivamente no treino. As análises univariadas estão implementadas, assim
como o classificador conjunto.
A avaliação final permanece para a RFC consumidora correspondente.

## Execução

Na raiz do repositório, com Python 3.10 ou superior:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Para inspecionar as tabelas no notebook, instale o kernel no mesmo ambiente:

```powershell
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Abra [02_analises_univariadas.ipynb](notebooks/02_analises_univariadas.ipynb)
em um editor compatível com Jupyter, selecione o Python de `.venv` e execute
todas as células em ordem. O kernel pode iniciar na raiz ou em `notebooks/`.
O notebook gera tabelas de priors, parâmetros numéricos, contagens e
probabilidades categóricas, além de log-likelihood, AIC e KS para duração.

## Modelagem da RFC-0003

| Componente | Estimativa no treino de cada classe |
|---|---|
| Prior | `N_c / N`, sem suavização ou balanceamento |
| `age` | Normal: média e variância MLE com `ddof=0` |
| `duration` | Gamma: forma e escala MLE, localização fixa em zero |
| `marital` | Categórica: `(N_c,k + 1) / (N_c + 3)` |
| Comparação de `duration` | Exponencial: taxa `1 / média`, localização zero |

As funções de ajuste e avaliação estão separadas em
[`src/distributions.py`](src/distributions.py). `GaussianParams`, `GammaParams`
e `ExponentialParams` são dataclasses imutáveis. `fit_class_priors` recebe
somente `y_train`; os ajustes de atributos recebem os valores de treino da
classe correspondente. As log-densidades aceitam vetores NumPy; a função
`categorical_logpmf` calcula log-probabilidades e rejeita categorias desconhecidas.

**Fronteira com SciPy:** `scipy.stats.gamma.fit(values, floc=0)` resolve
numericamente o MLE Gamma. A localização deve ser exatamente zero, forma e
escala devem ser positivas e finitas, e `shape * scale` deve reproduzir a
média empírica. As log-densidades Normal, Gamma e Exponencial são implementadas
explicitamente; a Gamma usa `scipy.special.gammaln`, sem calcular `gamma(k)`
ou `log(pdf(x))`. SciPy também fornece as CDFs e o KS para diagnóstico e serve
de referência secundária nos testes. A combinação de evidências e a decisão
do classificador serão implementadas na própria base pelas RFCs consumidoras.

O AIC é calculado por `2*q - 2*sum(log_densidades)`: Normal e Gamma têm `q=2`,
Exponencial tem `q=1`. Só é comparado no mesmo atributo, classe e amostra de
treino. O KS com parâmetros estimados é um diagnóstico relativo, sem usar
p-valores como prova isolada de aderência. A Exponencial permanece comparativa;
a distribuição principal de duração é a Gamma.

Entradas inválidas ou resultados não finitos geram erros explícitos. Gamma
exige valores estritamente positivos e uma amostra não constante para MLE
finito. A variância Gaussiana só recebe `variance_floor=1e-12` quando a
estimativa é zero na precisão numérica; variâncias positivas são preservadas.
O domínio categórico é `("divorced", "married", "single")`, sem `<UNK>`.
As somas de probabilidades são verificadas com tolerância numérica.

A Normal aproxima uma idade discreta, limitada e possivelmente multimodal.
Densidade contínua não é probabilidade pontual; prior e probabilidade
categórica também se distinguem da posterior normalizada. O produto da prior
pelas verossimilhanças serve à decisão, mas precisa de normalização para
representar uma posterior.

Os testes incluem cálculos manuais, conferência secundária com SciPy,
estabilidade nas caudas, contratos inválidos e os valores aproximados da RFC
no split congelado. Os valores de referência não são constantes dos ajustes.

## Classificador misto da RFC-0005

[`MixedNaiveBayes`](src/mixed_naive_bayes.py) combina a Normal de `age`, a Gamma
de `duration` e a categórica de `marital`, reutilizando os ajustes da RFC-0003.
As priors são as frequências empíricas do treino. A combinação e a decisão
são próprias; SciPy fornece `logsumexp` para normalização, além do suporte
numérico já descrito para as distribuições.

```python
from src.mixed_naive_bayes import MixedNaiveBayes

model = MixedNaiveBayes().fit(split.X_train, split.y_train)
scores = model.joint_log_likelihood(X)  # (n_samples, 2), classes [0, 1]
log_posterior = model.predict_log_proba(X)
posterior = model.predict_proba(X)     # linhas somando 1
prediction = model.predict(X)         # (n_samples,), classes 0 ou 1
parameters = model.get_fitted_parameters()
```

`X` deve ser um DataFrame com exatamente `age`, `duration` e `marital`;
as colunas são reordenadas explicitamente. No ajuste, `y` deve ser uma Series
com o mesmo índice e ordem de linhas, contendo as duas classes. Nulos,
números não finitos, duração não positiva e categorias fora do domínio
congelado geram erro. `alpha` e `variance_floor` devem ser finitos e positivos;
os padrões são `1.0` e `1e-12`.

O score soma a log-prior e as três log-verossimilhanças. A evidência é comum
às classes e pode ser removida do `argmax`; para obter posteriores, a
normalização usa `logsumexp`, após subtrair o maior score de cada linha.
Em empate exato, a classe 0 vence. O ajuste é transacional: uma falha mantém
o objeto novo não ajustado ou preserva integralmente um ajuste anterior.
Predições reutilizam o estado aprendido e o modelo não armazena os dados.

A hipótese de independência condicional é uma aproximação: `age` e `marital`
permanecem relacionados dentro das classes. As posteriores refletem essa
hipótese e as famílias de distribuição adotadas.

Para reproduzir o ajuste e gerar a auditoria, execute na raiz:

```powershell
.\.venv\Scripts\python.exe -m src.run_experiment
```

O comando valida a fonte, usa o split congelado e ajusta somente no treino.
Gera [`model_parameters.json`](reports/metrics/model_parameters.json), com
classes, contagens, priors, parâmetros contínuos, probabilidades categóricas,
hiperparâmetros, ordem das features, SHA-256 do dataset e identificação do
split. O arquivo fica disponível para versionamento. A reconstrução ocorre
por `fit`; o JSON é uma cópia para inspeção e não uma entrada de treinamento.
O runner desta etapa não calcula métricas da avaliação final.

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
│   └── 02_analises_univariadas.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── distributions.py
│   ├── univariate.py
│   ├── plotting.py
│   ├── run_univariate.py
│   ├── mixed_naive_bayes.py
│   └── run_experiment.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_distributions.py
│   ├── test_univariate.py
│   └── test_mixed_naive_bayes.py
└── requirements.txt
```

## Responsabilidades

- `src/config.py`: parâmetros imutáveis e nomes de colunas.
- `src/data.py`: leitura, validação, seleção e divisão dos dados.
- `src/distributions.py`: ajuste de distribuições e log-densidades.
- `src/mixed_naive_bayes.py`: ajuste e inferência do classificador conjunto.
- `src/run_experiment.py`: orquestração do ajuste e exportação dos parâmetros.
- `notebooks/`: explicação e inspeção; nunca a única implementação.

O módulo `evaluation.py` permanece planejado na RFC-0001 para a etapa de
avaliação final.

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
