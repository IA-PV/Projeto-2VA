# Projeto de IA: Classificação Bayesiana Pura e Mista

Implementação rigorosa e reproduzível de um classificador probabilístico supervisionado baseado em **Inferência Bayesiana Pura** (sem estimadores caixa-preta de terceiros) sobre o dataset bancário [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), em conformidade com as diretrizes da disciplina e governado por especificações técnicas de engenharia e governança de software.

---

## 1. Título e Objetivo

- **Título**: Estudo Dirigido de Inferência e Classificação Bayesiana Supervisionada.
- **Objetivo**: Estimar probabilidades a priori e verossimilhanças condicionais a partir de dados reais, comparar formulações teóricas (Normal, Gamma e Categórica com Laplace), analisar o comportamento probabilístico univariado (razão de verossimilhanças $\Lambda(x)$, fronteiras de decisão e posteriors normalizadas) e combinar as evidências em um classificador ingênuo misto (*Mixed Naive Bayes*), auditando seu desempenho contra o holdout congelado e confrontando-o com o baseline majoritário.

---

## 2. Integrantes

- **Vitor Antônio Silvestre Santos**
- **Pedro Tobias Souza Guerra**

---

## 3. Origem e Versão do Dataset

- **Dataset**: Amostra reduzida oficial contendo **4.521 observações** e 17 atributos do conjunto [Bank Marketing da UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank.csv`).
- **Arquivo local**: [`data/raw/bank.csv`](data/raw/bank.csv).
- **Integridade Criptográfica (SHA-256 com LF normalizado)**:
  `dc8d576e9bda0f41ee891251bd84bab9a39ce576cba715aac08adc2374a01fde`
- **Substituição**: A fonte de dados é congelada. Qualquer alteração ou substituição exige atualização e revisão formal do contrato de dados.

---

## 4. Problema de Classificação

O problema consiste em prever se um cliente bancário subscreverá um depósito a prazo fixo (*term deposit*) após uma abordagem telefônica de marketing direto.
O desafio central reside no **desbalanceamento severo das classes**:
- Na base total de 4.521 amostras, **88,48%** (4.000) pertencem à classe negativa (`no`) e apenas **11,52%** (521) à classe positiva (`yes`).
- Na presença de fortes desbalanceamentos, a acurácia isolada é enganosa (um classificador ingênuo que preveja sempre `no` atinge ~88,5% de acurácia com recall zero). A modelagem bayesiana lida explicitamente com as probabilidades a priori e as razões de verossimilhança.

---

## 5. Alvo e Classes

A variável-alvo $y$ é mapeada de maneira estrita e determinística:
- `no` $\rightarrow \mathbf{0}$ (classe negativa: o cliente não subscreveu o produto).
- `yes` $\rightarrow \mathbf{1}$ (classe positiva: o cliente subscreveu o produto).
- A ordem canônica dos rótulos em todas as matrizes e métricas é $\mathbf{[0, 1]}$.

---

## 6. Três Atributos Selecionados

Conforme definido no contrato de dados e na especificação do projeto, foram selecionados exatamente três atributos de tipos distintos:
1. **`age`** (numérico contínuo/inteiro): Idade do cliente (faixa observada: 18 a 95 anos).
2. **`duration`** (numérico contínuo positivo): Duração do último contato telefônico em segundos (faixa observada: 0 a 4.918 segundos).
3. **`marital`** (categórico politômico): Estado civil do cliente com domínio estrito $\mathcal{D} = \{\text{divorced}, \text{married}, \text{single}\}$.

---

## 7. Resumo das Distribuições e Modelagem

Toda a estimação de parâmetros é realizada **estritamente sobre as 3.616 observações de treinamento** (livre de vazamento de dados / data leakage):

| Componente | Família Probabilística | Método de Estimação no Treino |
|---|---|---|
| **Prior** $P(Y=c)$ | Empírica | Frequência relativa: $N_c / N_{\text{train}}$ ($P(0) \approx 0{,}8847$, $P(1) \approx 0{,}1153$) |
| **`age`** | Normal (Gaussiana) | $\hat{\mu}_c$ e $\hat{\sigma}_c^2$ via MLE com $N_c$ no denominador (`ddof=0`) |
| **`duration`** | Gamma | Forma $\hat{k}_c$ e escala $\hat{\theta}_c$ via MLE com localização fixa em zero (`floc=0`) |
| **`marital`** | Categórica | Frequência com suavização de Laplace: $\frac{N_{c,k} + \alpha}{N_c + \alpha K}$ com $\alpha=1$ e $K=3$ |
| *Diagnóstico* | Exponencial | Taxa $\hat{\lambda}_c = 1/\bar{x}_c$; superada pela Gamma no critério AIC |

- **Razão de Verossimilhança Univariada**: $\Lambda(x) = \frac{p(x \mid Y=1)}{p(x \mid Y=0)}$.
- **Regra de Decisão MAP**: Decide classe 1 se $\Lambda(x) > \frac{P(Y=0)}{P(Y=1)} \approx 7{,}6715$.
- **Combinação Multivariada**: O modelo misto assume independência condicional ingênua e soma as evidências em escala logarítmica natural:
  $$\ln P(Y=c \mid \mathbf{x}) \propto \ln P(Y=c) + \ln p(\text{age} \mid c) + \ln p(\text{duration} \mid c) + \ln P(\text{marital} \mid c)$$
  As probabilidades posteriores normalizadas são obtidas numericamente via `logsumexp`.

---

## 8. Estrutura do Repositório

```text
.
├── data/
│   └── raw/
│       └── bank.csv             # Amostra reduzida congelada (UCI)
├── notebooks/
│   └── 02_analises_univariadas.ipynb # Inspeção e visualização univariada
├── reports/
│   ├── figures/                 # Gráficos e curvas gerados programaticamente
│   │   ├── age_conditional_and_decision.png
│   │   ├── duration_conditional_and_decision.png
│   │   ├── marital_conditional_probabilities.png
│   │   └── confusion_matrix.png
│   └── metrics/                 # Relatórios JSON/CSV auditáveis versionados
│       ├── data_split.json
│       ├── distribution_parameters.json
│       ├── model_parameters.json
│       ├── final_metrics.json
│       ├── run_manifest.json
│       ├── confusion_matrix.csv
│       ├── error_groups.csv
│       ├── age_univariate_examples.csv
│       ├── duration_univariate_examples.csv
│       └── marital_univariate_examples.csv
├── src/                         # Implementação da biblioteca científica
│   ├── __init__.py
│   ├── config.py                # Configuração centralizada e ExperimentConfig imutável
│   ├── data.py                  # Ingestão, validação de contrato e split estratificado
│   ├── distributions.py         # Ajuste de distribuições contínuas e categóricas
│   ├── univariate.py            # Cálculo de posteriors, odds ratio e fronteiras
│   ├── mixed_naive_bayes.py     # Classificador Bayesiano Misto próprio
│   ├── evaluation.py            # Métricas manuais, matriz de confusão e baseline
│   ├── plotting.py              # Visualizações padronizadas
│   ├── run_univariate.py        # Runner da análise univariada
│   ├── run_evaluation.py        # Runner auditado da avaliação no holdout
│   └── run_experiment.py        # Orquestrador oficial do estudo completo
├── tests/                       # Suíte automatizada de testes e validação matemática
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_data_leakage.py
│   ├── test_distributions.py
│   ├── test_univariate.py
│   ├── test_mixed_naive_bayes.py
│   ├── test_evaluation.py
│   ├── test_final_evaluation.py
│   └── test_reproducibility.py  # Testes de reprodutibilidade e conformidade
├── pytest.ini                   # Configuração de execução do pytest
├── requirements.txt             # Dependências diretas com versões exatas testadas
└── README.md                    # Documentação oficial de entrega
```

---

## 9. Pré-requisitos de Ambiente

- **Linguagem**: Python $\ge$ 3.10 (validado e testado no **Python 3.13.7**).
- **Sistema Operacional**: Windows, Linux ou macOS.
- **Gerenciador de Ambientes**: Módulo padrão `venv` do Python.

---

## 10. Instalação

Abra o terminal na raiz do repositório:

### No Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### No Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## 11. Comandos de Validação, Teste e Execução

### 11.1 Validação de Dados e Ambiente (Somente Leitura)
Executa todas as checagens formais de contrato, esquema, hash e split sem criar ou modificar nenhum arquivo em disco:
```bash
python -m src.run_experiment --validate-only
```

### 11.2 Execução do Estudo Completo de Ponta a Ponta
Regenera todos os parâmetros, tabelas CSV, figuras PNG, métricas e o manifesto de execução:
```bash
python -m src.run_experiment
```

### 11.3 Execução da Suíte de Testes Automatizada
Executa os 244 testes unitários, matemáticos e de integração:
```bash
pytest -q
```

### 11.4 Inspeção Interativa dos Notebooks
Para inspecionar as tabelas formatadas e diagnósticos interativamente:
```bash
python -m pip install ipykernel
```
Abra [`notebooks/02_analises_univariadas.ipynb`](notebooks/02_analises_univariadas.ipynb) no Jupyter Lab ou VS Code, selecione o kernel `.venv` e execute todas as células em ordem (`Run All`).

---

## 12. Descrição das Saídas e Métricas Auditadas

A execução do estudo consolida artefatos em `reports/`:

### 12.1 Manifesto de Execução ([`run_manifest.json`](reports/metrics/run_manifest.json))
Registra metadados de execução, plataforma, hash dos dados, hiperparâmetros e famílias de distribuição de acordo com a especificação de reprodutibilidade do projeto.

### 12.2 Métricas Finais Auditadas no Holdout ([`final_metrics.json`](reports/metrics/final_metrics.json))
Resultados obtidos sobre as 905 observações congeladas de teste:

| Métrica | Classificador Misto Bayesiano | Baseline Majoritário (sempre classe 0) |
|---|---|---|
| **Acurácia** | **$88{,}73\%$** ($0{,}8873$) | $88{,}51\%$ ($0{,}8851$) |
| **Precisão** | **$51{,}92\%$** ($0{,}5192$) | $0{,}00\%$ ($0{,}0000$) |
| **Recall (Sensibilidade)** | **$25{,}96\%$** ($0{,}2596$) | $0{,}00\%$ ($0{,}0000$) |
| **F1-Score** | **$34{,}62\%$** ($0{,}3462$) | $0{,}00\%$ ($0{,}0000$) |

### 12.3 Matriz de Confusão Oficial ([`confusion_matrix.csv`](reports/metrics/confusion_matrix.csv))
Ordem canônica $[0, 1]$ (linhas: real, colunas: predito):
- **Verdadeiros Negativos (VN)**: $776$ (cliente não aderiu e o modelo previu não adesão)
- **Falsos Positivos (FP)**: $25$ (cliente não aderiu, mas o modelo previu adesão)
- **Falsos Negativos (FN)**: $77$ (cliente aderiu, mas o modelo previu não adesão)
- **Verdadeiros Positivos (VP)**: $27$ (cliente aderiu e o modelo previu adesão)
- **Total**: $776 + 25 + 77 + 27 = 905$ observações.

---

## 13. Decisões de Reprodutibilidade

- **Semente e Divisão**: Semente fixa `random_state = 42`, divisão estratificada `test_size = 0.20` garantindo proporções idênticas em treino e teste.
- **Isolamento Total do Holdout**: O conjunto de teste nunca participa da estimativa de parâmetros, seleção de hiperparâmetros ou calibração de priors.
- **Portão de Congelamento**: A avaliação no holdout é protegida por trava auditável em `reports/metrics/freeze_checklist.json` e `evaluation_history.json`.
- **Portabilidade de Caminhos**: Uso exclusivo de `pathlib.Path` e caminhos relativos ao projeto, sem caminhos absolutos locais de máquina.
- **Normalização de Final de Linha**: SHA-256 computado com normalização de `\r\n` para `\n`, eliminando divergências entre sistemas operacionais.

---

## 14. Limitações Principais

1. **Hipótese Ingênua de Independência Condicional**: Assume que idade, duração e estado civil são independentes dadas as classes, embora atributos como idade e estado civil apresentem correlações empíricas evidentes.
2. **Variável `duration` Pós-Contato (Viés de Seleção)**: A duração da chamada só é conhecida após o encerramento do contato. Portanto, em um cenário de triagem bancária a priori (antes de discar para o cliente), essa variável não está disponível.
3. **Desbalanceamento Severo**: A probabilidade a priori da classe negativa ($~88,5\%$) impõe um limiar elevado ($\Lambda > 7{,}67$), fazendo com que o classificador seja conservador na atribuição da classe positiva.

---

## 15. Uso de Ferramentas de IA Generativa e Processo de Verificação

Em consonância com as práticas éticas e acadêmicas de integridade científica:
- **Áreas com Apoio de IA**: Auxílio no planejamento estrutural da documentação e sugestões de arquitetura de testes unitários.
- **Implementação e Auditoria Humana**: Toda a matemática, deduções analíticas de máxima verossimilhança (MLE), parametrizações de log-densidades e cálculos de matriz de confusão foram conferidos, implementados e auditados pelos integrantes.
- **Validação Cruzada por Oráculos**: Todas as implementações manuais foram estritamente validadas contra oráculos secundários independentes (`scipy.stats` para distribuições e `sklearn.metrics` para matriz de confusão e métricas).
- **Responsabilidade**: A IA atuou como ferramenta de produtividade e pair programming; a responsabilidade técnica e científica final pertence integralmente aos autores.

---

## 16. Especificações Técnicas e Módulos do Projeto

As decisões de arquitetura e governança do projeto foram organizadas em módulos com responsabilidades bem delimitadas:

| Módulo / Especificação | Escopo e Responsabilidade |
|---|---|
| **01. Engenharia e Governança** | Padrões de engenharia de software, tipagem estrita, estrutura de diretórios e governança do repositório |
| **02. Contrato de Dados e Divisão** | Ingestão, validação de integridade criptográfica (SHA-256), esquema e split estratificado congelado |
| **03. Modelagem Probabilística** | Modelagem probabilística teórica (Normal, Gamma, Laplace e priors empíricas de treino) |
| **04. Análises Univariadas** | Experimentos Bayesianos univariados, cálculo da razão $\Lambda(x)$, fronteiras analíticas e posteriors |
| **05. Classificador Misto** | Implementação do Classificador Naive Bayes Misto supervisionado conjunto |
| **06. Estratégia de Testes** | Estratégia de testes automatizados e validação matemática de fórmulas contra oráculos |
| **07. Avaliação e Erros** | Protocolo de avaliação oficial no holdout, cálculo manual de métricas e análise de erros |
| **08. Reprodutibilidade** | Orquestrador completo, configuração imutável, dependências fixadas e manifesto de execução |

---

## 17. Licença e Citação da Base de Dados

O dataset original está sob licença **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
Citação formal:
> **Moro, S., Cortez, P., & Rita, P. (2014).** *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, Elsevier, 62:22-31.
