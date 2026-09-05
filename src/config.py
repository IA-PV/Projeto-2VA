"""Configuração imutável do projeto — RFC-0001 / RFC-0002.

Todas as constantes usadas pelo pipeline residem aqui.
Nenhum módulo deve redefinir esses valores localmente.
"""

from __future__ import annotations

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

FEATURE_COLUMNS: list[str] = ["age", "duration", "marital"]
"""As três features selecionadas para o modelo."""

TARGET_COLUMN: str = "y"
"""Nome da coluna alvo no CSV."""

TARGET_MAPPING: dict[str, int] = {"no": 0, "yes": 1}
"""Codificação explícita do alvo. Não depende de ordenação."""

# ──────────────────────────────────────────────
#  Domínios esperados (identidade da versão)
# ──────────────────────────────────────────────

MARITAL_CATEGORIES: frozenset[str] = frozenset({"divorced", "married", "single"})
"""Categorias válidas de estado civil na versão congelada."""

TARGET_CATEGORIES: frozenset[str] = frozenset({"no", "yes"})
"""Valores válidos do alvo."""

AGE_RANGE: tuple[int, int] = (18, 95)
"""Faixa observada de age na versão congelada (teste de identidade)."""

DURATION_RANGE: tuple[int, int] = (0, 4_918)
"""Faixa observada de duration (limite superior liberal para identidade)."""

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
