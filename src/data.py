"""Ingestão, validação, seleção e divisão dos dados — RFC-0002.

Responsabilidades
-----------------
- Ler o CSV bruto do UCI Bank Marketing.
- Validar integridade (hash), esquema e domínios.
- Codificar o alvo e selecionar as features do modelo.
- Executar uma única divisão estratificada e reproduzível.
- Gerar o artefato de auditoria JSON.

Não deve
--------
- Ajustar distribuições ou calcular parâmetros estatísticos.
- Remover observações por parecerem outliers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CSV_DELIMITER,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    MARITAL_CATEGORIES,
    RANDOM_STATE,
    SPLIT_REPORT_PATH,
    TARGET_CATEGORIES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Dataclass de resultado
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class DataSplit:
    """Resultado imutável da divisão treino/teste.

    Attributes
    ----------
    X_train : pd.DataFrame
        Features do conjunto de treinamento.
    X_test : pd.DataFrame
        Features do conjunto de teste.
    y_train : pd.Series
        Alvo codificado do treinamento.
    y_test : pd.Series
        Alvo codificado do teste.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


# ──────────────────────────────────────────────
#  Funções auxiliares internas
# ──────────────────────────────────────────────


def _compute_sha256(path: Path) -> str:
    """Calcula SHA-256 do arquivo normalizando CRLF → LF.

    Garante que o hash seja idêntico independentemente do sistema
    operacional em que o arquivo foi clonado.
    """
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


# ──────────────────────────────────────────────
#  Funções públicas
# ──────────────────────────────────────────────


def load_bank_data(path: Path) -> pd.DataFrame:
    """Lê o CSV bruto do Bank Marketing.

    Parameters
    ----------
    path : Path
        Caminho para ``bank.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame bruto com todas as 17 colunas e 4 521 linhas.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir ou não for um arquivo regular.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado ou não é regular: {path}"
        )
    df = pd.read_csv(path, sep=CSV_DELIMITER)
    logger.info("Arquivo carregado: %s (%d linhas, %d colunas)", path, len(df), len(df.columns))
    return df


def validate_raw_data(
    df: pd.DataFrame,
    path: Path | None = None,
    *,
    strict_hash: bool = True,
) -> None:
    """Executa as 12 validações obrigatórias da RFC-0002.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto retornado por :func:`load_bank_data`.
    path : Path | None
        Caminho do arquivo original, usado para validação de hash.
        Se ``None``, a validação de hash é ignorada.
    strict_hash : bool
        Se ``True`` (padrão), divergência de hash gera ``ValueError``.
        Se ``False``, gera apenas aviso — uso exclusivo para dev local.

    Raises
    ------
    ValueError
        Se qualquer validação falhar (exceto hash com ``strict_hash=False``).
    """
    errors: list[str] = []

    # ── 1. Arquivo existe e é regular ──
    if path is not None and not Path(path).is_file():
        errors.append(f"[V01] Arquivo não encontrado ou não regular: {path}")

    # ── 2. Hash corresponde à versão esperada ──
    if path is not None and Path(path).is_file():
        actual_hash = _compute_sha256(Path(path))
        if actual_hash != EXPECTED_SHA256:
            msg = (
                f"[V02] SHA-256 divergente.\n"
                f"  Esperado: {EXPECTED_SHA256}\n"
                f"  Obtido:   {actual_hash}"
            )
            if strict_hash:
                errors.append(msg)
            else:
                logger.warning("⚠️  OVERRIDE ATIVO — %s", msg)

    # ── 3. Delimitador produz exatamente 17 colunas ──
    if len(df.columns) != len(EXPECTED_COLUMNS):
        errors.append(
            f"[V03] Esperadas {len(EXPECTED_COLUMNS)} colunas, "
            f"encontradas {len(df.columns)}: {list(df.columns)}"
        )

    # ── 4. Nomes das colunas correspondem ao esquema ──
    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"[V04] Colunas não correspondem ao esquema.\n"
            f"  Esperado: {EXPECTED_COLUMNS}\n"
            f"  Obtido:   {list(df.columns)}"
        )

    # ── 5. Existem 4 521 linhas ──
    if len(df) != EXPECTED_ROWS:
        errors.append(
            f"[V05] Esperadas {EXPECTED_ROWS} linhas, encontradas {len(df)}."
        )

    # ── Colunas usadas pelo modelo ──
    used_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    available = [c for c in used_cols if c in df.columns]

    # ── 6. Não existem nulos nas quatro colunas usadas ──
    if set(used_cols).issubset(df.columns):
        null_counts = df[used_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(
                f"[V06] Nulos encontrados nas colunas do modelo: "
                f"{dict(cols_with_nulls)}"
            )

    # ── 7. age e duration são numéricos e finitos ──
    for col in ["age", "duration"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"[V07] Coluna '{col}' não é numérica.")
            elif not np.all(np.isfinite(df[col])):
                errors.append(f"[V07] Coluna '{col}' contém valores não finitos.")

    # ── 8. duration > 0 (requisito da Gamma) ──
    if "duration" in df.columns and pd.api.types.is_numeric_dtype(df["duration"]):
        non_positive = (df["duration"] <= 0).sum()
        if non_positive > 0:
            errors.append(
                f"[V08] {non_positive} valor(es) de 'duration' ≤ 0. "
                f"Requisito da distribuição Gamma: duration deve ser estritamente positivo."
            )

    # ── 9. Categorias de marital são as três esperadas ──
    if "marital" in df.columns:
        actual_cats = set(df["marital"].unique())
        if actual_cats != MARITAL_CATEGORIES:
            errors.append(
                f"[V09] Categorias de 'marital' inesperadas.\n"
                f"  Esperado: {sorted(MARITAL_CATEGORIES)}\n"
                f"  Obtido:   {sorted(actual_cats)}"
            )

    # ── 10. Alvo contém exatamente no e yes ──
    if TARGET_COLUMN in df.columns:
        actual_target = set(df[TARGET_COLUMN].unique())
        if actual_target != TARGET_CATEGORIES:
            errors.append(
                f"[V10] Valores do alvo '{TARGET_COLUMN}' inesperados.\n"
                f"  Esperado: {sorted(TARGET_CATEGORIES)}\n"
                f"  Obtido:   {sorted(actual_target)}"
            )

    # ── 11. Codificação gera apenas 0 e 1 ──
    if TARGET_COLUMN in df.columns:
        mapped = df[TARGET_COLUMN].map(TARGET_MAPPING)
        unmapped = mapped.isnull().sum() - df[TARGET_COLUMN].isnull().sum()
        if unmapped > 0:
            bad_vals = set(df[TARGET_COLUMN]) - set(TARGET_MAPPING.keys())
            errors.append(
                f"[V11] Valores do alvo sem mapeamento definido: {bad_vals}"
            )

    # ── 12. Sem linhas completamente duplicadas ──
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        errors.append(
            f"[V12] {n_dups} linha(s) completamente duplicada(s) encontrada(s)."
        )

    # ── Resultado ──
    if errors:
        separator = "\n  • "
        raise ValueError(
            f"Validação do dataset falhou com {len(errors)} erro(s):"
            f"{separator}{separator.join(errors)}"
        )

    logger.info("✅ Todas as 12 validações passaram.")


def prepare_model_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Seleciona features e codifica o alvo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto (já validado).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` onde X contém as features na ordem padronizada e
        y é o alvo codificado como 0/1.

    Raises
    ------
    ValueError
        Se colunas necessárias estiverem ausentes ou se houver
        valores do alvo sem mapeamento.
    """
    # Verifica presença das colunas
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas necessárias ausentes no DataFrame: {sorted(missing)}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN]

    # Codificação explícita
    y = y_raw.map(TARGET_MAPPING)
    unmapped_mask = y.isnull() & y_raw.notnull()
    if unmapped_mask.any():
        bad_vals = set(y_raw[unmapped_mask])
        raise ValueError(
            f"Valores do alvo sem mapeamento: {bad_vals}. "
            f"Mapeamento válido: {TARGET_MAPPING}"
        )

    y = y.astype(int)
    y.name = TARGET_COLUMN

    logger.info(
        "Model frame: %d amostras, features=%s, classes=%s",
        len(X),
        list(X.columns),
        sorted(y.unique()),
    )
    return X, y


def make_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> DataSplit:
    """Executa a divisão estratificada única e reproduzível.

    Parameters
    ----------
    X : pd.DataFrame
        Features selecionadas.
    y : pd.Series
        Alvo codificado como 0/1.

    Returns
    -------
    DataSplit
        Objeto imutável com X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split: treino=%d (classe0=%d, classe1=%d), teste=%d (classe0=%d, classe1=%d)",
        len(y_train),
        (y_train == 0).sum(),
        (y_train == 1).sum(),
        len(y_test),
        (y_test == 0).sum(),
        (y_test == 1).sum(),
    )

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def save_split_report(
    split: DataSplit,
    sha256: str,
    path: Path | None = None,
) -> Path:
    """Gera o artefato de auditoria JSON.

    Parameters
    ----------
    split : DataSplit
        Resultado da divisão.
    sha256 : str
        Hash SHA-256 do arquivo fonte.
    path : Path | None
        Caminho de saída. Se ``None``, usa ``SPLIT_REPORT_PATH``.

    Returns
    -------
    Path
        Caminho absoluto do arquivo gerado.
    """
    if path is None:
        path = SPLIT_REPORT_PATH
    path = Path(path)

    y_full = pd.concat([split.y_train, split.y_test])

    report = {
        "dataset": "bank.csv",
        "sha256": sha256,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "full_counts": {
            "0": int((y_full == 0).sum()),
            "1": int((y_full == 1).sum()),
        },
        "train_counts": {
            "0": int((split.y_train == 0).sum()),
            "1": int((split.y_train == 1).sum()),
        },
        "test_counts": {
            "0": int((split.y_test == 0).sum()),
            "1": int((split.y_test == 1).sum()),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Artefato de auditoria salvo em: %s", path)
    return path.resolve()
