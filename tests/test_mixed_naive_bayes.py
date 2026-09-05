"""Contrato, cálculo manual e isolamento do treino — RFC-0005."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.config import EXPECTED_SHA256, MARITAL_CATEGORIES, RANDOM_STATE, TEST_SIZE
from src.mixed_naive_bayes import MixedNaiveBayes
from src import run_experiment


@pytest.fixture
def training():
    X = pd.DataFrame({
        "age": [20., 30., 40., 35., 45., 55., 65.],
        "duration": [10., 20., 40., 30., 60., 90., 150.],
        "marital": ["single", "single", "married", "married", "divorced", "single", "married"],
    }, index=[9, 3, 8, 1, 7, 2, 6])
    return X, pd.Series([0, 0, 0, 1, 1, 1, 1], index=X.index)


@pytest.fixture
def fitted(training):
    return MixedNaiveBayes().fit(*training)


def test_manual_joint_and_posterior(training, fitted):
    X, y = training
    query = X.iloc[[1, 5]]
    expected = []
    for _, row in query.iterrows():
        weights = []
        for c in (0, 1):
            group = X.loc[y == c]
            mean = sum(group.age) / len(group)
            variance = sum((value - mean) ** 2 for value in group.age) / len(group)
            normal = math.exp(-(row.age - mean) ** 2 / (2 * variance))
            normal /= math.sqrt(2 * math.pi * variance)
            gamma = fitted.duration_params_[c]
            density = row.duration ** (gamma.shape - 1) * math.exp(-row.duration / gamma.scale)
            density /= math.gamma(gamma.shape) * gamma.scale ** gamma.shape
            categorical = (sum(group.marital == row.marital) + 1) / (len(group) + 3)
            weights.append(len(group) / len(X) * normal * density * categorical)
        expected.append(weights)
    expected = np.asarray(expected)
    np.testing.assert_allclose(fitted.joint_log_likelihood(query), np.log(expected))
    posterior = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(fitted.predict_proba(query), posterior)
    np.testing.assert_allclose(fitted.predict_log_proba(query), np.log(posterior))
    np.testing.assert_array_equal(fitted.predict(query), np.argmax(posterior, axis=1))
    assert fitted.class_count_ == {0: 3, 1: 4}
    # Categoria ausente da classe 0 recebe Laplace no domínio completo K=3.
    assert np.exp(fitted.marital_log_prob_[0]["divorced"]) == pytest.approx(1 / 6)


def test_custom_hyperparameters_and_zero_variance(training):
    X, y = training
    X.loc[y == 0, "age"] = 30.
    model = MixedNaiveBayes(alpha=2., variance_floor=1e-5)
    assert model.fit(X, y) is model
    assert model.age_params_[0].variance == 1e-5
    assert np.exp(model.marital_log_prob_[0]["divorced"]) == pytest.approx(2 / 9)


@pytest.mark.parametrize("name", ["alpha", "variance_floor"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, "1", None, True])
def test_invalid_hyperparameters(name, value):
    with pytest.raises(ValueError, match=name):
        MixedNaiveBayes(**{name: value})


@pytest.mark.parametrize("method", [
    "joint_log_likelihood", "predict_log_proba", "predict_proba", "predict",
    "get_fitted_parameters",
])
def test_requires_fit(method, training):
    model = MixedNaiveBayes()
    args = () if method == "get_fitted_parameters" else (training[0],)
    with pytest.raises(RuntimeError, match="fit"):
        getattr(model, method)(*args)


@pytest.mark.parametrize("invalid", [
    lambda X: X.to_numpy(),
    lambda X: X.drop(columns="age"),
    lambda X: X.assign(extra=1),
    lambda X: X.set_axis(["age", "age", "marital"], axis=1),
    lambda X: X.assign(age=np.nan),
    lambda X: X.assign(duration=np.inf),
    lambda X: X.assign(age=-np.inf),
    lambda X: X.assign(age="30"),
    lambda X: X.assign(age=30 + 1j),
    lambda X: X.assign(age=True),
    lambda X: X.assign(duration=0),
    lambda X: X.assign(duration=-1),
    lambda X: X.assign(marital="unknown"),
    lambda X: X.assign(marital=None),
])
def test_invalid_X_rejected_at_fit_and_score(invalid, training, fitted):
    X, y = training
    bad = invalid(X)
    with pytest.raises((TypeError, ValueError)):
        MixedNaiveBayes().fit(bad, y)
    with pytest.raises((TypeError, ValueError)):
        fitted.joint_log_likelihood(bad)


@pytest.mark.parametrize("invalid", [
    lambda y: y.to_numpy(),
    lambda y: y.iloc[:-1],
    lambda y: y.iloc[::-1],
    lambda y: y * 0,
    lambda y: y.replace(1, 2),
    lambda y: y.replace(1, np.nan),
    lambda y: y.astype(str),
])
def test_invalid_target(invalid, training):
    X, y = training
    with pytest.raises((TypeError, ValueError)):
        MixedNaiveBayes().fit(X, invalid(y))


def test_empty_fit_rejected_but_empty_prediction_supported(training, fitted):
    X, y = training
    empty = X.iloc[:0]
    with pytest.raises(ValueError):
        MixedNaiveBayes().fit(empty, y.iloc[:0])
    for method in ("joint_log_likelihood", "predict_log_proba", "predict_proba"):
        assert getattr(fitted, method)(empty).shape == (0, 2)
    assert fitted.predict(empty).shape == (0,)


def test_transactional_fit_and_refit(training, fitted):
    X, y = training
    bad = X.copy()
    bad.loc[y == 1, "duration"] = 1.  # falha após ajustar a classe 0
    fresh = MixedNaiveBayes()
    with pytest.raises(ValueError, match="constantes"):
        fresh.fit(bad, y)
    assert vars(fresh) == {"alpha": 1., "variance_floor": 1e-12, "is_fitted_": False}
    previous = fitted.get_fitted_parameters()
    prediction = fitted.predict_proba(X)
    with pytest.raises(ValueError, match="constantes"):
        fitted.fit(bad, y)
    assert fitted.get_fitted_parameters() == previous
    np.testing.assert_array_equal(fitted.predict_proba(X), prediction)
    fitted.fit(X.assign(age=X.age + 10), y)
    assert fitted.age_params_[0].mean == pytest.approx(previous["age"]["0"]["mean"] + 10)


def test_reordering_and_pandas_categorical(training, fitted):
    X, y = training
    reordered = X.loc[:, ["marital", "duration", "age"]].copy()
    reordered["marital"] = pd.Categorical(reordered.marital, categories=MARITAL_CATEGORIES)
    other = MixedNaiveBayes().fit(reordered, y)
    assert other.feature_names_in_ == ("age", "duration", "marital")
    assert other.n_features_in_ == 3
    np.testing.assert_array_equal(other.predict_proba(reordered), fitted.predict_proba(X))
    np.testing.assert_array_equal(fitted.predict(X.iloc[::-1]), fitted.predict(X)[::-1])


def test_exact_ties_choose_zero_even_in_far_tail(training):
    group = training[0].iloc[:3]
    X = pd.concat([group, group], ignore_index=True)
    model = MixedNaiveBayes().fit(X, pd.Series([0, 0, 0, 1, 1, 1]))
    # O produto linear sofre underflow; scores muito negativos e iguais
    # ainda precisam produzir posteriores exatamente simétricas.
    query = group.assign(age=1e10, duration=1e8)
    scores = model.joint_log_likelihood(query)
    assert np.isfinite(scores).all()
    assert (np.exp(scores) == 0).all()
    np.testing.assert_array_equal(scores[:, 0], scores[:, 1])
    np.testing.assert_array_equal(model.predict_proba(query), np.full((3, 2), 0.5))
    np.testing.assert_array_equal(model.predict(query), np.zeros(3, dtype=int))


def test_scoring_does_not_refit_or_retain_input(training, fitted, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Predição tentou recalcular parâmetros.")

    for name in ("fit_gaussian_mle", "fit_gamma_mle", "fit_categorical", "fit_class_priors"):
        monkeypatch.setattr(f"src.mixed_naive_bayes.{name}", forbidden)
    X, _ = training
    previous = fitted.get_fitted_parameters()
    query = X.copy()
    fitted.predict(query)
    fitted.predict_log_proba(query)
    fitted.predict_proba(query)
    X.loc[:, "age"] = 999
    query.loc[:, "duration"] = 999
    assert fitted.get_fitted_parameters() == previous
    assert not any(isinstance(value, (pd.DataFrame, pd.Series)) for value in vars(fitted).values())


def test_export_is_independent_and_json_serializable(fitted):
    export = fitted.get_fitted_parameters()
    assert json.loads(json.dumps(export, allow_nan=False)) == export
    export["age"]["0"]["mean"] = -100
    export["marital"]["0"]["single"] = 0
    export["classes"].reverse()
    export["class_count"]["0"] = 0
    assert fitted.get_fitted_parameters() != export
    assert fitted.class_count_[0] == 3
    assert fitted.age_params_[0].mean == 30
    assert fitted.class_order_ == (0, 1)
    assert np.exp(fitted.marital_log_prob_[0]["single"]) == pytest.approx(0.5)


def test_frozen_split_invariants_and_scipy_reference(data_split):
    model = MixedNaiveBayes().fit(data_split.X_train, data_split.y_train)
    assert model.class_count_ == {0: 3199, 1: 417}
    X = data_split.X_test
    expected = []
    for c in (0, 1):
        age = model.age_params_[c]
        duration = model.duration_params_[c]
        expected.append(
            math.log(model.class_count_[c] / 3616)
            + stats.norm.logpdf(X.age, loc=age.mean, scale=math.sqrt(age.variance))
            + stats.gamma.logpdf(X.duration, a=duration.shape, loc=0, scale=duration.scale)
            + X.marital.map(model.marital_log_prob_[c]).to_numpy()
        )
        assert sum(np.exp(list(model.marital_log_prob_[c].values()))) == pytest.approx(1.)
    np.testing.assert_allclose(model.joint_log_likelihood(X), np.column_stack(expected))
    posterior = model.predict_proba(X)
    assert posterior.shape == (905, 2)
    assert np.isfinite(posterior).all()
    assert ((posterior >= 0) & (posterior <= 1)).all()
    np.testing.assert_allclose(posterior.sum(axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(np.exp(model.predict_log_proba(X)), posterior)
    np.testing.assert_array_equal(model.predict(X), posterior.argmax(axis=1))
    assert np.exp(list(model.class_log_prior_.values())).sum() == pytest.approx(1.)


def test_runner_uses_only_training_and_exports_reproducibly(data_split, csv_path, tmp_path, monkeypatch):
    monkeypatch.setattr(run_experiment, "DATA_PATH", csv_path)
    output = tmp_path / "model_parameters.json"
    monkeypatch.setattr(run_experiment, "MODEL_PARAMETERS_PATH", output)
    # Não oferece X_test/y_test: qualquer acesso a eles no runner falharia.
    monkeypatch.setattr(run_experiment, "make_stratified_split", lambda X, y: SimpleNamespace(
        X_train=data_split.X_train, y_train=data_split.y_train,
    ))
    assert run_experiment.run() == output
    first = output.read_bytes()
    report = json.loads(first)
    assert report["dataset"]["sha256"] == EXPECTED_SHA256
    assert report["split"] == {
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "stratify": "y", "n_train": 3616,
    }
    expected = MixedNaiveBayes().fit(data_split.X_train, data_split.y_train).get_fitted_parameters()
    assert {key: report[key] for key in expected} == expected
    run_experiment.run()
    assert output.read_bytes() == first
