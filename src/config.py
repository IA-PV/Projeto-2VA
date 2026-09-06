"""Configuração imutável do projeto — Governança, Contratos e Modelagem.

Todas as constantes usadas pelo pipeline residem aqui.
Nenhum módulo deve redefinir esses valores localmente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ──────────────────────────────────────────────
#  Identidade da fonte
# ──────────────────────────────────────────────

DATA_PATH: Path = Path("data/raw/bank.csv")
"""Caminho relativo ao repositório para o CSV versionado."""

EXPECTED_SHA256: str = (
    "dc8d576e9bda0f41ee891251bd84bab9a39ce576cba715aac08adc2374a01fde"
)
"""SHA-256 do conteúdo do bank.csv com line endings normalizados para LF."""

CSV_DELIMITER: str = ";"
"""Delimitador do CSV original do UCI."""

EXPECTED_ROWS: int = 4_521
"""Quantidade de observações (excluindo header)."""

EXPECTED_COLUMNS: list[str] = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y",
]
"""Nomes das 17 colunas na ordem do CSV original."""

# ──────────────────────────────────────────────
#  Seleção de atributos e codificação do alvo
# ──────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = ["age", "campaign", "loan"]
"""As três features selecionadas para o modelo."""

TARGET_COLUMN: str = "y"
"""Nome da coluna alvo no CSV."""

TARGET_MAPPING: dict[str, int] = {"no": 0, "yes": 1}
"""Codificação explícita do alvo. Não depende de ordenação."""

# ──────────────────────────────────────────────
#  Domínios esperados (identidade da versão)
# ──────────────────────────────────────────────

LOAN_CATEGORIES: tuple[str, ...] = ("no", "yes")
"""Categorias válidas de empréstimo pessoal na versão congelada."""

TARGET_CATEGORIES: frozenset[str] = frozenset({"no", "yes"})
"""Valores válidos do alvo."""

AGE_RANGE: tuple[int, int] = (19, 87)
"""Faixa observada de age na amostra reduzida congelada."""

CAMPAIGN_RANGE: tuple[int, int] = (1, 50)
"""Faixa observada de campaign na amostra reduzida congelada."""

# ──────────────────────────────────────────────
#  Divisão treino/teste
# ──────────────────────────────────────────────

TEST_SIZE: float = 0.20
"""Fração do dataset reservada para teste."""

RANDOM_STATE: int = 42
"""Semente para reprodutibilidade do split."""

# ──────────────────────────────────────────────
#  Suavização (Laplace)
# ──────────────────────────────────────────────

LAPLACE_ALPHA: float = 1.0
"""Parâmetro de suavização para probabilidades categóricas."""

VARIANCE_FLOOR: float = 1e-12
"""Variância substituta somente quando a estimativa Gaussiana é nula."""

# ──────────────────────────────────────────────
#  Ordem de classes
# ──────────────────────────────────────────────

CLASS_ORDER: list[int] = [0, 1]
"""Ordem canônica das classes para matrizes e métricas."""

# ──────────────────────────────────────────────
#  Caminhos de saída
# ──────────────────────────────────────────────

SPLIT_REPORT_PATH: Path = Path("reports/metrics/data_split.json")
"""Caminho para o artefato de auditoria do split."""

MODEL_PARAMETERS_PATH: Path = Path("reports/metrics/model_parameters.json")
"""Parâmetros do classificador misto e identificação do dataset/split."""

# ──────────────────────────────────────────────
#  Experimentos Bayesianos Univariados
# ──────────────────────────────────────────────

AGE_EXAMPLES: list[int] = [20, 40, 60, 80]
"""Valores predefinidos de age para tabelas de exemplo univariado."""

CAMPAIGN_EXAMPLES: list[int] = [1, 2, 5, 10]
"""Valores predefinidos de campaign para tabelas de exemplo univariado."""

UNIVARIATE_FIGURES_DIR: Path = Path("reports/figures")
"""Diretório de saída para figuras da análise univariada."""

UNIVARIATE_METRICS_DIR: Path = Path("reports/metrics")
"""Diretório de saída para métricas/tabelas da análise univariada."""

# ──────────────────────────────────────────────
#  Reprodutibilidade e Configuração
# ──────────────────────────────────────────────

DISTRIBUTIONS: dict[str, str] = {
    "age": "gaussian",
    "campaign": "gamma",
    "loan": "categorical_laplace",
}
"""Mapeamento canônico das famílias de distribuição por atributo."""

RUN_MANIFEST_PATH: Path = Path("reports/metrics/run_manifest.json")
"""Caminho padrão para o manifesto formal de execução."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuração imutável do estudo científico.

    Centraliza todos os parâmetros do pipeline de dados, modelagem,
    avaliação e caminhos de entrada e saída.
    """

    data_path: Path = Path("data/raw/bank.csv")
    output_dir: Path = Path("reports")
    feature_columns: tuple[str, ...] = ("age", "campaign", "loan")
    target_column: str = "y"
    test_size: float = 0.20
    random_state: int = 42
    laplace_alpha: float = 1.0
    class_order: tuple[int, int] = (0, 1)

    @property
    def metrics_dir(self) -> Path:
        """Diretório de métricas e relatórios JSON/CSV."""
        return self.output_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        """Diretório de figuras e gráficos PNG."""
        return self.output_dir / "figures"

    @property
    def split_report_path(self) -> Path:
        """Caminho do artefato de auditoria da divisão de dados."""
        return self.metrics_dir / "data_split.json"

    @property
    def model_parameters_path(self) -> Path:
        """Caminho dos parâmetros ajustados do modelo misto."""
        return self.metrics_dir / "model_parameters.json"

    @property
    def run_manifest_path(self) -> Path:
        """Caminho do manifesto formal de execução."""
        return self.metrics_dir / "run_manifest.json"

    @property
    def distribution_parameters_path(self) -> Path:
        """Caminho dos parâmetros univariados estimados."""
        return self.metrics_dir / "distribution_parameters.json"

    @property
    def final_metrics_path(self) -> Path:
        """Caminho do resultado da avaliação oficial no conjunto de teste."""
        return self.metrics_dir / "final_metrics.json"

