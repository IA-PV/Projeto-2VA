"""Testes de avaliação e métricas — RFC-0006 e RFC-0007.

Valida:
- Fixture manual obrigatória da RFC-0006.
- Matriz de confusão com convenção linhas=reais, colunas=preditas.
- Extração dos 4 componentes: VN=2, FP=1, FN=1, VP=2.
- Fórmulas manuais de acurácia, precisão, recall e F1.
- Comparação exata contra scikit-learn (com labels=[0, 1] e pos_label=1).
- Rejeição de entradas inválidas e proteção contra divisão por zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation import (
    compute_confusion_matrix,
    compute_majority_baseline,
    compute_metrics,
    extract_confusion_components,
)


@pytest.fixture
def rfc0006_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Fixture manual obrigatória da RFC-0006."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    return y_true, y_pred


class TestRFC0006ManualEvaluation:
    """Validação da fixture manual descrita na RFC-0006."""

    def test_confusion_matrix_values(self, rfc0006_fixture: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = rfc0006_fixture
        cm = compute_confusion_matrix(y_true, y_pred, labels=(0, 1))

        # Matriz esperada:
        # [[2, 1],
        #  [1, 2]]
        expected_cm = np.array([[2, 1], [1, 2]], dtype=int)
        np.testing.assert_array_equal(cm, expected_cm)

    def test_extracted_components(self, rfc0006_fixture: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = rfc0006_fixture
        cm = compute_confusion_matrix(y_true, y_pred)
        comps = extract_confusion_components(cm)

        assert comps["tn"] == 2
        assert comps["fp"] == 1
        assert comps["fn"] == 1
        assert comps["tp"] == 2

    def test_metrics_values(self, rfc0006_fixture: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = rfc0006_fixture
        metrics = compute_metrics(y_true, y_pred, pos_label=1)

        # acurácia=4/6, precisão=2/3, recall=2/3, F1=2/3
        assert metrics["accuracy"] == pytest.approx(4 / 6, rel=1e-12, abs=1e-12)
        assert metrics["precision"] == pytest.approx(2 / 3, rel=1e-12, abs=1e-12)
        assert metrics["recall"] == pytest.approx(2 / 3, rel=1e-12, abs=1e-12)
        assert metrics["f1"] == pytest.approx(2 / 3, rel=1e-12, abs=1e-12)

    def test_matches_sklearn_oracle(self, rfc0006_fixture: tuple[np.ndarray, np.ndarray]) -> None:
        """Compara a implementação manual com scikit-learn como oráculo secundário."""
        y_true, y_pred = rfc0006_fixture

        # Matriz
        cm_manual = compute_confusion_matrix(y_true, y_pred, labels=(0, 1))
        cm_sklearn = confusion_matrix(y_true, y_pred, labels=[0, 1])
        np.testing.assert_array_equal(cm_manual, cm_sklearn)

        # Métricas
        metrics = compute_metrics(y_true, y_pred, pos_label=1)
        assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred), rel=1e-12)
        assert metrics["precision"] == pytest.approx(
            precision_score(y_true, y_pred, pos_label=1), rel=1e-12
        )
        assert metrics["recall"] == pytest.approx(
            recall_score(y_true, y_pred, pos_label=1), rel=1e-12
        )
        assert metrics["f1"] == pytest.approx(
            f1_score(y_true, y_pred, pos_label=1), rel=1e-12
        )


class TestEvaluationRobustnessAndEdgeCases:
    """Testes de invariantes e casos limites."""

    def test_perfect_classifier(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_completely_inverted_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_zero_division_protection_when_no_positive_predictions(self) -> None:
        """Quando o modelo prevê apenas 0, precision e f1 têm denominador zero."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_metrics(y_true, y_pred, zero_division=0.0)
        assert metrics["accuracy"] == 0.5
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

        # Confere consistência com sklearn zero_division=0
        assert metrics["precision"] == precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        assert metrics["f1"] == f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    def test_majority_baseline(self) -> None:
        # 8 observações classe 0 e 2 classe 1 -> baseline 0.8
        y_true = np.array([0] * 8 + [1] * 2)
        assert compute_majority_baseline(y_true) == pytest.approx(0.8)

    @pytest.mark.parametrize("invalid_true, invalid_pred, match", [
        ([0, 1], [0], "Dimensões incompatíveis"),
        ([], [], "não pode ser vazio"),
        ([[0, 1]], [[0, 1]], "unidimensional"),
        ([0, 2], [0, 1], "domínio binário"),
        ([0, 1], [0, "yes"], "domínio binário"),
    ])
    def test_rejects_invalid_inputs(self, invalid_true, invalid_pred, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            compute_confusion_matrix(invalid_true, invalid_pred)

    def test_rejects_unsupported_labels_or_pos_label(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            compute_confusion_matrix([0, 1], [0, 1], labels=(1, 0))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="pos_label"):
            compute_metrics([0, 1], [0, 1], pos_label=0)

    def test_extract_components_rejects_non_2x2_matrix(self) -> None:
        with pytest.raises(ValueError, match=r"\(2, 2\)"):
            extract_confusion_components(np.zeros((3, 3)))
