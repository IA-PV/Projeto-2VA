"""Testes do módulo de dados — RFC-0002.

Testa:
- Carregamento correto do CSV.
- Hash SHA-256 (normalizado LF).
- Esquema (17 colunas, nomes, tipos).
- 12 validações obrigatórias.
- Codificação do alvo.
- Split estratificado (contagens exatas da RFC).
- Artefato de auditoria JSON.
- Erros esperados para DataFrames inválidos.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    MARITAL_CATEGORIES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.data import (
    DataSplit,
    _compute_sha256,
    load_bank_data,
    make_stratified_split,
    prepare_model_frame,
    save_split_report,
    validate_raw_data,
)


# ════════════════════════════════════════════════════════
#  Carregamento
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestLoadBankData:
    """Testes para ``load_bank_data``."""

    def test_returns_dataframe(self, raw_df: pd.DataFrame) -> None:
        assert isinstance(raw_df, pd.DataFrame)

    def test_column_count(self, raw_df: pd.DataFrame) -> None:
        assert len(raw_df.columns) == 17

    def test_row_count(self, raw_df: pd.DataFrame) -> None:
        assert len(raw_df) == EXPECTED_ROWS

    def test_column_names(self, raw_df: pd.DataFrame) -> None:
        assert list(raw_df.columns) == EXPECTED_COLUMNS

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_bank_data(tmp_path / "inexistente.csv")


# ════════════════════════════════════════════════════════
#  Hash SHA-256
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestHash:
    """Verificação do hash normalizado."""

    def test_sha256_matches(self, csv_path: Path) -> None:
        actual = _compute_sha256(csv_path)
        assert actual == EXPECTED_SHA256, (
            f"Hash divergente!\n  Esperado: {EXPECTED_SHA256}\n  Obtido:   {actual}"
        )


# ════════════════════════════════════════════════════════
#  Validações de domínio (arquivo real)
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestValidateRawData:
    """Testes usando o arquivo real."""

    def test_passes_all_validations(
        self, raw_df: pd.DataFrame, csv_path: Path
    ) -> None:
        # Não deve levantar
        validate_raw_data(raw_df, path=csv_path)

    def test_no_nulls_in_model_columns(self, raw_df: pd.DataFrame) -> None:
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert raw_df[col].isnull().sum() == 0, f"Nulos em '{col}'"

    def test_age_numeric_and_finite(self, raw_df: pd.DataFrame) -> None:
        assert pd.api.types.is_numeric_dtype(raw_df["age"])
        assert np.all(np.isfinite(raw_df["age"]))

    def test_duration_positive(self, raw_df: pd.DataFrame) -> None:
        assert (raw_df["duration"] > 0).all(), "duration deve ser > 0"

    def test_duration_numeric_and_finite(self, raw_df: pd.DataFrame) -> None:
        assert pd.api.types.is_numeric_dtype(raw_df["duration"])
        assert np.all(np.isfinite(raw_df["duration"]))

    def test_marital_categories(self, raw_df: pd.DataFrame) -> None:
        assert set(raw_df["marital"].unique()) == set(MARITAL_CATEGORIES)

    def test_target_categories(self, raw_df: pd.DataFrame) -> None:
        assert set(raw_df[TARGET_COLUMN].unique()) == {"no", "yes"}

    def test_no_full_duplicates(self, raw_df: pd.DataFrame) -> None:
        assert raw_df.duplicated().sum() == 0


# ════════════════════════════════════════════════════════
#  Validações com DataFrames inválidos
# ════════════════════════════════════════════════════════


class TestValidateRejectsInvalid:
    """Testes com DataFrames sintéticos que devem falhar."""

    def _make_valid_df(self) -> pd.DataFrame:
        """Constrói um mini-DataFrame que passa nas validações (sem path/hash)."""
        n = EXPECTED_ROWS
        data = {col: ["x"] * n for col in EXPECTED_COLUMNS}
        # Sobrescreve as colunas usadas pelo modelo
        data["age"] = list(range(30, 30 + n))
        data["duration"] = list(range(1, 1 + n))
        data["marital"] = ["married"] * (n // 3) + ["single"] * (n // 3) + ["divorced"] * (n - 2 * (n // 3))
        data["y"] = ["no"] * (n - 521) + ["yes"] * 521
        return pd.DataFrame(data)

    def test_wrong_column_count(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match=r"V03"):
            validate_raw_data(df)

    def test_extra_column_rejected(self) -> None:
        """Caso 5 da RFC-0006: coluna extra deve ser rejeitada."""
        df = self._make_valid_df().assign(extra_col=1)
        with pytest.raises(ValueError, match=r"V03"):
            validate_raw_data(df)

    def test_wrong_column_names(self) -> None:
        df = self._make_valid_df()
        df = df.rename(columns={"age": "idade"})
        with pytest.raises(ValueError, match=r"V04"):
            validate_raw_data(df)

    def test_incorrect_separator_detected(self, tmp_path: Path) -> None:
        """Caso 3 da RFC-0006: separador incorreto é detectado."""
        bad_csv = tmp_path / "bad_sep.csv"
        bad_csv.write_text("age,job,marital\n30,admin,single\n", encoding="utf-8")
        bad_df = load_bank_data(bad_csv)
        with pytest.raises(ValueError, match=r"V03|V04"):
            validate_raw_data(bad_df)

    def test_wrong_row_count(self) -> None:
        df = self._make_valid_df()
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match=r"V05"):
            validate_raw_data(df)

    @pytest.mark.parametrize("col", ["age", "duration", "marital", "y"])
    def test_null_in_any_model_column_rejected(self, col: str) -> None:
        """Caso 6 da RFC-0006: valor nulo em feature ou target é rejeitado."""
        df = self._make_valid_df()
        df.loc[0, col] = None
        with pytest.raises(ValueError, match=r"V06"):
            validate_raw_data(df)

    @pytest.mark.parametrize("bad_duration", [0, -1, -50])
    def test_duration_not_positive(self, bad_duration: int) -> None:
        """Caso 7 da RFC-0006: duration <= 0 é rejeitada."""
        df = self._make_valid_df()
        df.loc[0, "duration"] = bad_duration
        with pytest.raises(ValueError, match=r"V08"):
            validate_raw_data(df)

    def test_unexpected_marital_value(self) -> None:
        df = self._make_valid_df()
        df.loc[0, "marital"] = "widowed"
        with pytest.raises(ValueError, match=r"V09"):
            validate_raw_data(df)

    def test_unexpected_target_value(self) -> None:
        df = self._make_valid_df()
        df.loc[0, "y"] = "maybe"
        with pytest.raises(ValueError, match=r"V10"):
            validate_raw_data(df)

    def test_age_non_numeric_string(self) -> None:
        """V07: age contendo string deve ser rejeitado como não numérico."""
        df = self._make_valid_df()
        df["age"] = df["age"].astype(object)
        df.loc[0, "age"] = "trinta"
        with pytest.raises(ValueError, match=r"V07"):
            validate_raw_data(df)

    def test_unmappable_target_encoding(self) -> None:
        """V11: alvo com valor fora do TARGET_MAPPING deve ser rejeitado."""
        df = self._make_valid_df()
        df.loc[0, "y"] = "unknown"
        with pytest.raises(ValueError, match=r"V1[01]"):
            validate_raw_data(df)

    def test_exact_duplicate_row(self) -> None:
        """V12: linha completamente duplicada deve ser rejeitada."""
        df = self._make_valid_df()
        # Clona a primeira linha sobre a segunda para forçar duplicata exata
        df.iloc[1] = df.iloc[0]
        with pytest.raises(ValueError, match=r"V12"):
            validate_raw_data(df)


# ════════════════════════════════════════════════════════
#  Codificação do alvo (prepare_model_frame)
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestPrepareModelFrame:
    """Testes para ``prepare_model_frame``."""

    def test_returns_correct_features(self, validated_df: pd.DataFrame) -> None:
        X, _ = prepare_model_frame(validated_df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_target_is_binary(self, validated_df: pd.DataFrame) -> None:
        _, y = prepare_model_frame(validated_df)
        assert set(y.unique()) == {0, 1}

    def test_target_mapping_no_is_0(self, validated_df: pd.DataFrame) -> None:
        _, y = prepare_model_frame(validated_df)
        # No original, "no" → 0
        no_mask = validated_df[TARGET_COLUMN] == "no"
        assert (y[no_mask] == 0).all()

    def test_target_mapping_yes_is_1(self, validated_df: pd.DataFrame) -> None:
        _, y = prepare_model_frame(validated_df)
        yes_mask = validated_df[TARGET_COLUMN] == "yes"
        assert (y[yes_mask] == 1).all()

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"age": [30], "duration": [100]})
        with pytest.raises(ValueError, match="ausentes"):
            prepare_model_frame(df)

    def test_unmapped_target_raises(self) -> None:
        df = pd.DataFrame({
            "age": [30],
            "duration": [100],
            "marital": ["single"],
            "y": ["maybe"],
        })
        with pytest.raises(ValueError, match="mapeamento"):
            prepare_model_frame(df)

    def test_class_counts(self, validated_df: pd.DataFrame) -> None:
        _, y = prepare_model_frame(validated_df)
        assert (y == 0).sum() == 4000
        assert (y == 1).sum() == 521


# ════════════════════════════════════════════════════════
#  Split estratificado
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestMakeStratifiedSplit:
    """Testes para ``make_stratified_split``."""

    def test_returns_datasplit(self, data_split: DataSplit) -> None:
        assert isinstance(data_split, DataSplit)

    def test_train_size(self, data_split: DataSplit) -> None:
        assert len(data_split.y_train) == 3616

    def test_test_size(self, data_split: DataSplit) -> None:
        assert len(data_split.y_test) == 905

    def test_total_preserved(self, data_split: DataSplit) -> None:
        total = len(data_split.y_train) + len(data_split.y_test)
        assert total == EXPECTED_ROWS

    def test_train_class0_count(self, data_split: DataSplit) -> None:
        assert (data_split.y_train == 0).sum() == 3199

    def test_train_class1_count(self, data_split: DataSplit) -> None:
        assert (data_split.y_train == 1).sum() == 417

    def test_test_class0_count(self, data_split: DataSplit) -> None:
        assert (data_split.y_test == 0).sum() == 801

    def test_test_class1_count(self, data_split: DataSplit) -> None:
        assert (data_split.y_test == 1).sum() == 104

    def test_prior_class0(self, data_split: DataSplit) -> None:
        prior_0 = (data_split.y_train == 0).sum() / len(data_split.y_train)
        assert abs(prior_0 - 0.884679) < 1e-4

    def test_prior_class1(self, data_split: DataSplit) -> None:
        prior_1 = (data_split.y_train == 1).sum() / len(data_split.y_train)
        assert abs(prior_1 - 0.115321) < 1e-4

    def test_features_match(self, data_split: DataSplit) -> None:
        assert list(data_split.X_train.columns) == FEATURE_COLUMNS
        assert list(data_split.X_test.columns) == FEATURE_COLUMNS

    def test_no_index_overlap(self, data_split: DataSplit) -> None:
        """Treino e teste não compartilham índices."""
        overlap = set(data_split.X_train.index) & set(data_split.X_test.index)
        assert len(overlap) == 0

    def test_index_union_covers_full_dataset(self, data_split: DataSplit) -> None:
        """Caso 14 da RFC-0006: união dos índices cobre a base completa."""
        union = set(data_split.X_train.index) | set(data_split.X_test.index)
        assert union == set(range(EXPECTED_ROWS))
        assert len(data_split.X_train) + len(data_split.X_test) == EXPECTED_ROWS

    def test_frozen_dataclass(self, data_split: DataSplit) -> None:
        """DataSplit é imutável."""
        with pytest.raises(AttributeError):
            data_split.X_train = None  # type: ignore[misc]


# ════════════════════════════════════════════════════════
#  Artefato de auditoria
# ════════════════════════════════════════════════════════


@pytest.mark.regression
class TestSaveSplitReport:
    """Testes para ``save_split_report``."""

    def test_creates_json_file(
        self, data_split: DataSplit, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        save_split_report(data_split, sha256=EXPECTED_SHA256, path=out)
        assert out.is_file()

    def test_json_contents(
        self, data_split: DataSplit, tmp_path: Path
    ) -> None:
        out = tmp_path / "report.json"
        save_split_report(data_split, sha256=EXPECTED_SHA256, path=out)
        report = json.loads(out.read_text(encoding="utf-8"))

        assert report["dataset"] == "bank.csv"
        assert report["sha256"] == EXPECTED_SHA256
        assert report["random_state"] == RANDOM_STATE
        assert report["test_size"] == TEST_SIZE
        assert report["full_counts"] == {"0": 4000, "1": 521}
        assert report["train_counts"] == {"0": 3199, "1": 417}
        assert report["test_counts"] == {"0": 801, "1": 104}
