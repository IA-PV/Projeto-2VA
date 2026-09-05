"""Fixtures compartilhadas para o test suite — RFC-0002."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DATA_PATH
from src.data import (
    DataSplit,
    load_bank_data,
    make_stratified_split,
    prepare_model_frame,
    validate_raw_data,
)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Raiz do projeto (diretório que contém ``data/``)."""
    # Resolve a partir do diretório deste arquivo de teste
    root = Path(__file__).resolve().parent.parent
    assert (root / "data" / "raw" / "bank.csv").is_file(), (
        "bank.csv não encontrado. Execute os testes a partir da raiz do projeto."
    )
    return root


@pytest.fixture(scope="session")
def csv_path(project_root: Path) -> Path:
    """Caminho absoluto para bank.csv."""
    return project_root / DATA_PATH


@pytest.fixture(scope="session")
def raw_df(csv_path: Path) -> "pd.DataFrame":
    """DataFrame bruto carregado uma única vez por sessão."""
    import pandas as pd  # noqa: F811 — lazy para não poluir namespace global

    return load_bank_data(csv_path)


@pytest.fixture(scope="session")
def validated_df(raw_df, csv_path: Path) -> "pd.DataFrame":
    """DataFrame bruto já validado (levanta se inválido)."""
    validate_raw_data(raw_df, path=csv_path)
    return raw_df


@pytest.fixture(scope="session")
def model_xy(validated_df) -> "tuple":
    """(X, y) prontos para o modelo."""
    return prepare_model_frame(validated_df)


@pytest.fixture(scope="session")
def data_split(model_xy) -> DataSplit:
    """DataSplit produzido pela divisão estratificada."""
    X, y = model_xy
    return make_stratified_split(X, y)
