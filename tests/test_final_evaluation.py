"""RFC-0007: protocolo e artefatos em fixtures sintéticas, sem avaliar o holdout."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import DataSplit
from src.evaluation import build_error_groups, evaluate_predictions
from src import run_evaluation as runner


def test_plot_preserves_counts_and_axis_convention(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from src.plotting import plot_confusion_matrix

    def inspect(figure, *args, **kwargs):
        ax = figure.axes[0]
        assert ax.get_xlabel() == "Classe predita"
        assert ax.get_ylabel() == "Classe real"
        assert "teste" in ax.get_title()
        assert [label.get_text() for label in ax.get_xticklabels()] == ["0 - não aderiu", "1 - aderiu"]
        assert [label.get_text() for label in ax.get_yticklabels()] == ["0 - não aderiu", "1 - aderiu"]
        assert {text.get_position(): text.get_text() for text in ax.texts} == {
            (0, 0): "801", (1, 0): "0", (0, 1): "103", (1, 1): "1"}

    monkeypatch.setattr(Figure, "savefig", inspect)
    plot_confusion_matrix(np.array([[801, 0], [103, 1]]), tmp_path / "cm.png")


def test_asymmetric_matrix_and_oracle():
    result = evaluate_predictions([0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 0, 1])
    assert result["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 2, "tp": 1}
    assert result["metrics"] == pytest.approx(
        {"accuracy": 4 / 7, "precision": 1 / 2, "recall": 1 / 3, "f1": 2 / 5})
    assert result["sklearn_consistency_verified"]
    assert not result["no_positive_predictions"]


@pytest.mark.parametrize("truth,prediction", [([0, 0], [0, 0]), ([1, 1], [0, 0]), ([1], [1])])
def test_one_class_and_no_positive_flag(truth, prediction):
    result = evaluate_predictions(truth, prediction)
    assert result["no_positive_predictions"] == (sum(prediction) == 0)
    assert all(0 <= value <= 1 for value in result["metrics"].values())


def test_error_groups_empty_ties_alignment_and_reference_counts():
    X = pd.DataFrame({"age": [20, 40, 60, 80], "duration": [100, 900, 500, 1000],
                      "marital": ["single", "married", "divorced", "single"]}, index=[7, 3, 9, 1])
    y = pd.Series([0, 0, 1, 1], index=X.index)
    kwargs = {"duration_boundary": 800, "long_duration_threshold": 950}
    groups = build_error_groups(X, y, [0, 0, 0, 1], **kwargs).set_index("group")
    assert list(groups.index) == ["VN", "FP", "FN", "VP"]
    assert groups.loc["VN", "age_mean"] == 30
    assert groups.loc["VN", "marital_mode"] == "married | single"
    assert groups.loc["FP", "n"] == 0
    assert pd.isna(groups.loc["FP", "duration_mean"])
    assert pd.isna(groups.loc["FP", "duration_below_boundary_fraction"])
    assert groups.loc["FN", "duration_below_boundary_n"] == 1
    assert groups.loc["VP", "duration_above_train_q95_n"] == 1
    assert groups["n"].sum() == len(X)
    with pytest.raises(ValueError, match="Índice"):
        build_error_groups(X, y.iloc[::-1], [0, 0, 0, 1], **kwargs)
    with pytest.raises(ValueError, match="Dimensões"):
        build_error_groups(X, [0], [0], **kwargs)


@pytest.fixture
def synthetic_runner(tmp_path, monkeypatch):
    """Spy com tamanhos do contrato; nenhum dado real é consultado."""
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    X = pd.DataFrame({"age": np.tile([20, 40, 60], 1507),
                      "duration": np.arange(1, 4522), "marital": "single"})
    y_train = pd.Series([0] * 3199 + [1] * 417, index=X.index[:3616])
    y_test = pd.Series([0] * 801 + [1] * 104, index=X.index[3616:])
    split = DataSplit(X.iloc[:3616], X.iloc[3616:], y_train, y_test)
    monkeypatch.setattr(runner, "load_bank_data", lambda path: X)
    monkeypatch.setattr(runner, "validate_raw_data", lambda *a, **k: None)
    monkeypatch.setattr(runner, "prepare_model_frame", lambda raw: (X, pd.concat([y_train, y_test])))
    monkeypatch.setattr(runner, "make_stratified_split", lambda *a: split)
    events = []
    params = {"priors": {"0": 3199 / 3616, "1": 417 / 3616},
              "duration": {str(c): {"shape": 2.0, "scale": 100.0} for c in (0, 1)}}

    def prerequisites():
        events.append("prerequisites")
        runner._write_json(tmp_path / runner.METRICS / "distribution_parameters.json", {
            "priors": params["priors"], "duration": {
                "class_0": params["duration"]["0"], "class_1": params["duration"]["1"],
                "map_boundaries": [800.0]}})
        return {"tests": {"output": "fixture passed"}, "univariate": {"completed_at_utc": runner._utc_now()}}

    class Spy:
        def __init__(self, **kwargs):
            assert kwargs == {"alpha": 1.0}

        def fit(self, features, target):
            assert features is split.X_train and target is split.y_train
            events.append("fit")
            return self

        def get_fitted_parameters(self):
            return dict(params)

        def predict(self, features):
            assert features is split.X_test
            assert (tmp_path / runner.CHECKLIST).is_file()
            assert (tmp_path / runner.MODEL_PARAMETERS_PATH).is_file()
            history = json.loads((tmp_path / runner.HISTORY).read_text(encoding="utf-8"))
            assert history[-1]["status"] == "started"
            events.append("predict")
            return np.array([0] * 780 + [1] * 21 + [0] * 80 + [1] * 24)

    monkeypatch.setattr(runner, "_run_prerequisites", prerequisites)
    monkeypatch.setattr(runner, "MixedNaiveBayes", Spy)
    return tmp_path, events


def test_official_order_artifacts_and_reevaluation_history(synthetic_runner):
    root, events = synthetic_runner
    path = runner.run()
    assert events == ["prerequisites", "fit", "predict"]
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["confusion_matrix"] == {"tn": 780, "fp": 21, "fn": 80, "tp": 24}
    assert result["majority_baseline_accuracy"] == pytest.approx(801 / 905)
    assert result["majority_baseline"]["no_positive_predictions"]
    assert result["majority_baseline"]["metrics"]["recall"] == 0
    assert result["majority_baseline"]["metrics"]["f1"] == 0
    assert '"recall": 0.0000' in path.read_text(encoding="utf-8")
    cm = pd.read_csv(root / runner.METRICS / "confusion_matrix.csv", index_col=0)
    np.testing.assert_array_equal(cm.to_numpy(), [[780, 21], [80, 24]])
    groups = pd.read_csv(root / runner.METRICS / "error_groups.csv")
    assert groups["n"].tolist() == [780, 21, 80, 24]
    assert (root / "reports/figures/confusion_matrix.png").stat().st_size > 0
    with pytest.raises(ValueError, match="Reavaliação"):
        runner.run()
    assert events.count("predict") == 1
    runner.run(reason="Correção de eixo comprovada em fixture.", changed_files=["src/plotting.py"])
    history = json.loads((root / runner.HISTORY).read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[0]["result"] == result
    assert history[1]["changed_files"] == ["src/plotting.py"]


def test_closed_gate_prevents_prediction(synthetic_runner, monkeypatch):
    _, events = synthetic_runner

    def fail():
        raise RuntimeError("Testes não aprovados")

    monkeypatch.setattr(runner, "_run_prerequisites", fail)
    with pytest.raises(RuntimeError, match="Testes não aprovados"):
        runner.run()
    assert events == []


def test_failed_evaluation_remains_in_history(synthetic_runner, monkeypatch):
    root, events = synthetic_runner

    def fail(*args, **kwargs):
        raise RuntimeError("Oráculo divergente")

    monkeypatch.setattr(runner, "evaluate_predictions", fail)
    with pytest.raises(RuntimeError, match="Oráculo divergente"):
        runner.run()
    history = json.loads((root / runner.HISTORY).read_text(encoding="utf-8"))
    assert history[-1]["status"] == "failed"
    assert not (root / runner.FINAL).exists()
    with pytest.raises(ValueError, match="Reavaliação"):
        runner.run()
    assert events.count("predict") == 1
