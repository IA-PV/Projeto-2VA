"""RFC-0003: exemplos manuais, estabilidade, contratos e referências de treino."""

from dataclasses import FrozenInstanceError
from math import log, pi

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import digamma

from src.config import MARITAL_CATEGORIES, VARIANCE_FLOOR
from src.distributions import (
    ExponentialParams,
    GammaParams,
    GaussianParams,
    aic,
    categorical_logpmf,
    compare_duration_distributions,
    exponential_logpdf,
    fit_categorical,
    fit_class_priors,
    fit_exponential_mle,
    fit_gamma_mle,
    fit_gaussian_mle,
    gamma_logpdf,
    gaussian_logpdf,
)


def test_gaussian_manual_mle_and_logpdf():
    # Média 2; soma dos desvios quadráticos 2; divisor MLE 3 (não 2).
    params = fit_gaussian_mle(np.array([1, 2, 3]))
    assert params.mean == 2
    assert params.variance == pytest.approx(2 / 3)
    expected = np.array([-0.5 * log(4 * pi / 3), -0.5 * log(4 * pi / 3) - 0.75])
    np.testing.assert_allclose(gaussian_logpdf(np.array([2, 3]), params), expected)


def test_gaussian_variance_floor_only_for_zero():
    assert fit_gaussian_mle(np.array([5, 5])).variance == VARIANCE_FLOOR
    assert fit_gaussian_mle(np.array([5]), variance_floor=1e-10).variance == 1e-10
    assert fit_gaussian_mle(np.array([-1e-8, 1e-8])).variance == pytest.approx(1e-16, rel=1e-12, abs=0)


def test_gamma_manual_logpdf():
    # Gamma(2,3): p(3)=exp(-1)/3 e p(6)=2*exp(-2)/3; Gamma(2)=1.
    actual = gamma_logpdf(np.array([3, 6]), GammaParams(shape=2, scale=3))
    np.testing.assert_allclose(actual, [-1 - log(3), log(2) - 2 - log(3)])


def test_gamma_mle_satisfies_likelihood_equations():
    x = np.array([1, 2, 3, 6, 9], dtype=float)
    params = fit_gamma_mle(x)
    assert params.shape * params.scale == pytest.approx(x.mean())
    assert log(params.shape) - digamma(params.shape) == pytest.approx(
        log(x.mean()) - np.log(x).mean()
    )
    assert np.isfinite(gamma_logpdf(x, params)).all()


def test_exponential_manual_mle_logpdf_and_aic():
    params = fit_exponential_mle(np.array([0, 2, 4]))
    assert params.rate == 0.5
    logs = exponential_logpdf(np.array([0, 2, 4]), params)
    np.testing.assert_allclose(logs, [-log(2), -log(2) - 1, -log(2) - 2])
    assert aic(logs, q=1) == pytest.approx(8 + 6 * log(2))
    assert aic(logs, q=2) - aic(logs, q=1) == pytest.approx(2)


def test_logpdfs_against_scipy_and_underflow():
    x = np.array([1e-9, 1, 10, 1000, 1e6])
    np.testing.assert_allclose(
        gaussian_logpdf(x, GaussianParams(2, 9)), stats.norm.logpdf(x, loc=2, scale=3)
    )
    # shape alto estouraria gamma(shape); a cauda extrema anularia pdf(x).
    np.testing.assert_allclose(
        gamma_logpdf(x, GammaParams(200, 3)), stats.gamma.logpdf(x, a=200, scale=3)
    )
    np.testing.assert_allclose(
        exponential_logpdf(x, ExponentialParams(0.5)), stats.expon.logpdf(x, scale=2)
    )
    assert np.isfinite(gaussian_logpdf(np.array([1000]), GaussianParams(0, 1))).all()


@pytest.mark.parametrize("params,field", [
    (GaussianParams(0, 1), "mean"), (GammaParams(2, 3), "shape"),
    (ExponentialParams(0.5), "rate"),
])
def test_parameters_are_immutable(params, field):
    with pytest.raises(FrozenInstanceError):
        setattr(params, field, 42)


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf])
def test_invalid_parameters_and_floor(invalid):
    for create in (
        lambda: GaussianParams(0, invalid), lambda: GammaParams(invalid, 1),
        lambda: GammaParams(1, invalid), lambda: ExponentialParams(invalid),
        lambda: fit_gaussian_mle(np.array([1, 2]), variance_floor=invalid),
    ):
        with pytest.raises(ValueError, match="positivo"):
            create()


@pytest.mark.parametrize("fit", [fit_gaussian_mle, fit_gamma_mle, fit_exponential_mle])
@pytest.mark.parametrize("invalid", [[], [np.nan], [np.inf], [[1, 2]], [1 + 2j]])
def test_numeric_fits_reject_invalid_input(fit, invalid):
    with pytest.raises(ValueError):
        fit(np.array(invalid))


@pytest.mark.parametrize("invalid", [[0, 1], [-1, 2], [3, 3], [3]])
def test_gamma_rejects_nonpositive_or_constant_samples(invalid):
    with pytest.raises(ValueError, match="positivo|constantes"):
        fit_gamma_mle(np.array(invalid))


@pytest.mark.parametrize("fitted, message", [
    ((2, 1, 1), "localização"), ((np.inf, 0, 1), "shape"),
    ((2, 0, -1), "scale"), ((2, 0, 100), "média"),
])
def test_gamma_checks_solver_result(monkeypatch, fitted, message):
    def fake_fit(values, *, floc):
        assert floc == 0
        return fitted
    monkeypatch.setattr(stats.gamma, "fit", fake_fit)
    with pytest.raises(ValueError, match=message):
        fit_gamma_mle(np.array([1, 2, 3]))


def test_invalid_evaluation_and_overflow_are_explicit():
    with pytest.raises(ValueError, match="positivo"):
        gamma_logpdf(np.array([0]), GammaParams(2, 3))
    with pytest.raises(ValueError, match=">= 0"):
        exponential_logpdf(np.array([-1]), ExponentialParams(1))
    with pytest.raises(ValueError, match="positiv"):
        fit_exponential_mle(np.array([0, 0]))
    with pytest.raises(ValueError, match="não finita"):
        gaussian_logpdf(np.array([1e308]), GaussianParams(0, 1))
    with pytest.raises(ValueError, match="finit"):
        fit_gaussian_mle(np.array([-1e308, 1e308]))


def test_laplace_manual_with_absent_category():
    observed = pd.Series(["married", "married", "single"])
    probs = fit_categorical(observed, MARITAL_CATEGORIES, alpha=1)
    assert list(probs) == list(MARITAL_CATEGORIES)
    assert probs == pytest.approx({"divorced": 1 / 6, "married": 3 / 6, "single": 2 / 6})
    assert np.isclose(sum(probs.values()), 1)
    np.testing.assert_allclose(
        categorical_logpmf(pd.Series(["divorced", "single", "married"]), probs),
        [log(1 / 6), log(2 / 6), log(3 / 6)],
    )


@pytest.mark.parametrize("values", [["unknown"], [None], []])
def test_categorical_rejects_invalid_training(values):
    with pytest.raises(ValueError):
        fit_categorical(pd.Series(values))


def test_categorical_rejects_unknown_during_evaluation():
    probs = fit_categorical(pd.Series(["single"]))
    with pytest.raises(ValueError, match="Categorias desconhecidas.*unknown"):
        categorical_logpmf(pd.Series(["unknown"]), probs)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_invalid_laplace_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit_categorical(pd.Series(["single"]), alpha=alpha)


@pytest.mark.parametrize("probabilities", [
    {"single": 0}, {"single": np.nan}, {"single": 0.5}, {"single": 2}, {},
])
def test_categorical_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        categorical_logpmf(pd.Series([], dtype=str), probabilities)


def test_priors_manual_without_smoothing():
    assert fit_class_priors(pd.Series([0, 0, 0, 1])) == {0: 0.75, 1: 0.25}


@pytest.mark.parametrize("values", [[], [0, 0], [1, 1], [0, 2], [0, None]])
def test_priors_reject_missing_or_invalid_classes(values):
    with pytest.raises(ValueError):
        fit_class_priors(pd.Series(values))


@pytest.mark.parametrize("q", [0, -1, 1.5, True])
def test_aic_rejects_invalid_parameter_counts(q):
    with pytest.raises(ValueError, match="q"):
        aic(np.array([-1, -2]), q=q)


@pytest.mark.parametrize(
    "c,count,mean,std,shape,scale,counts,exp_aic,gamma_aic,exp_ks,gamma_ks", [
        (0, 3199, 40.87183, 10.06207, 1.52854, 147.44781,
         [366, 2003, 830], 41062.98, 40755.66, 0.1172, 0.0457),
        (1, 417, 42.36451, 13.06510, 2.25276, 247.81951,
         [64, 220, 133], 6110.93, 5986.05, 0.1802, 0.0525),
    ],
)
def test_rfc_training_references(
    data_split, c, count, mean, std, shape, scale, counts,
    exp_aic, gamma_aic, exp_ks, gamma_ks,
):
    # Apenas X_train/y_train: referências arredondadas são oráculos, não parâmetros.
    y = data_split.y_train
    train = data_split.X_train.loc[y == c]
    assert len(train) == count
    priors = fit_class_priors(y)
    assert priors[c] == pytest.approx(count / 3616)
    assert np.isclose(sum(priors.values()), 1)
    age = fit_gaussian_mle(train["age"].to_numpy())
    assert age.mean == pytest.approx(mean, abs=1e-5, rel=0)
    assert np.sqrt(age.variance) == pytest.approx(std, abs=1e-5, rel=0)
    duration = train["duration"].to_numpy()
    gamma = fit_gamma_mle(duration)
    assert gamma.shape == pytest.approx(shape, abs=1e-5, rel=0)
    assert gamma.scale == pytest.approx(scale, abs=1e-5, rel=0)
    assert gamma.shape * gamma.scale == pytest.approx(duration.mean())
    assert np.isfinite(gaussian_logpdf(train["age"].to_numpy(), age)).all()
    assert np.isfinite(gamma_logpdf(duration, gamma)).all()
    observed = train["marital"].value_counts().reindex(MARITAL_CATEGORIES)
    assert observed.tolist() == counts
    probs = fit_categorical(train["marital"])
    assert list(probs.values()) == pytest.approx([(n + 1) / (count + 3) for n in counts])
    assert np.isclose(sum(probs.values()), 1)
    diagnostic = compare_duration_distributions(duration).set_index("distribution")
    np.testing.assert_allclose(diagnostic["aic"], [exp_aic, gamma_aic], atol=0.02, rtol=0)
    np.testing.assert_allclose(diagnostic["ks"], [exp_ks, gamma_ks], atol=1e-4, rtol=0)
    assert diagnostic["q"].tolist() == [1, 2]
    assert diagnostic.loc["Gamma", "aic"] < diagnostic.loc["Exponencial", "aic"]
    assert diagnostic.loc["Gamma", "ks"] < diagnostic.loc["Exponencial", "ks"]
