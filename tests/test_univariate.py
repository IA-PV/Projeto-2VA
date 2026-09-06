"""Testes da análise Bayesiana univariada.

Cobre:
- Cálculos manuais de posterior
- Estabilidade logsumexp
- Smoke values da especificação
- Propriedades algébricas das posteriors
- Regra categórica
- Fronteiras contínuas
- Integração com dados reais via fixture data_split
"""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src.config import AGE_EXAMPLES, DURATION_EXAMPLES, MARITAL_CATEGORIES
from src.distributions import (
    categorical_logpmf,
    fit_categorical,
    fit_class_priors,
    fit_gamma_mle,
    fit_gaussian_mle,
    gamma_logpdf,
    gaussian_logpdf,
)
from src.univariate import (
    UnivariateResult,
    analyze_univariate,
    build_categorical_rule,
    compute_log_scores,
    compute_posteriors,
    find_continuous_boundaries,
    map_decision,
)


# ──────────────────────────────────────────────
#  Testes de compute_log_scores
# ──────────────────────────────────────────────


class TestComputeLogScores:
    def test_basic_computation(self):
        log_prior = log(0.25)
        log_liks = np.array([log(0.1), log(0.5)])
        scores = compute_log_scores(log_prior, log_liks)
        np.testing.assert_allclose(scores, [log(0.025), log(0.125)])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finitos"):
            compute_log_scores(float("-inf"), np.array([0.0]))


# ──────────────────────────────────────────────
#  Testes de compute_posteriors
# ──────────────────────────────────────────────


class TestComputePosteriors:
    def test_conceptual_prior_dominance_example(self):
        """Exemplo conceitual de dominância da prior.

        P(0)=0.8, P(1)=0.2
        p(x|0)=0.1, p(x|1)=0.3
        Lambda = 3.
        P(1|x) = (0.3*0.2) / (0.1*0.8 + 0.3*0.2) = 3/7 ≈ 0.428571.
        P(0|x) = (0.1*0.8) / (0.1*0.8 + 0.3*0.2) = 4/7 ≈ 0.571429.

        Embora Lambda > 1, a decisão é classe 0. Protege a distinção conceitual central.
        """
        priors = {0: 0.8, 1: 0.2}
        log_pdf_0 = np.array([log(0.1)])
        log_pdf_1 = np.array([log(0.3)])

        # 1. Likelihood ratio Lambda = 3
        lambda_val = np.exp(log_pdf_1[0] - log_pdf_0[0])
        assert lambda_val == pytest.approx(3.0, rel=1e-12)

        # 2. Posteriores
        s0 = log(priors[0]) + log_pdf_0
        s1 = log(priors[1]) + log_pdf_1
        p0, p1 = compute_posteriors(s0, s1)

        assert p1[0] == pytest.approx(3 / 7, rel=1e-10, abs=1e-12)
        assert p0[0] == pytest.approx(4 / 7, rel=1e-10, abs=1e-12)
        assert p0[0] + p1[0] == pytest.approx(1.0, abs=1e-12)

        # 3. Decisão: classe 0 (prior domina)
        decision = map_decision(s0, s1)
        assert decision[0] == 0

        # 4. Análise univariada completa
        res = analyze_univariate("x", np.array([1.0]), priors, log_pdf_0, log_pdf_1)
        assert res.likelihood_ratio[0] == pytest.approx(3.0, rel=1e-12)
        assert res.posterior_class_1[0] == pytest.approx(3 / 7, rel=1e-10, abs=1e-12)
        assert res.predictions[0] == 0

    def test_linear_and_logarithmic_implementations_coincide(self):
        """Implementação linear manual e logarítmica coincidem com alta precisão."""
        # Valores arbitrários em escala moderada
        p0_val, p1_val = 0.65, 0.35
        lik0_val, lik1_val = 0.04, 0.09

        # Cálculo linear manual
        num0 = p0_val * lik0_val
        num1 = p1_val * lik1_val
        denom = num0 + num1
        linear_post0 = num0 / denom
        linear_post1 = num1 / denom

        # Cálculo logarítmico via compute_posteriors
        s0 = np.array([log(p0_val) + log(lik0_val)])
        s1 = np.array([log(p1_val) + log(lik1_val)])
        log_post0, log_post1 = compute_posteriors(s0, s1)

        np.testing.assert_allclose(log_post0, [linear_post0], atol=1e-12)
        np.testing.assert_allclose(log_post1, [linear_post1], atol=1e-12)

    def test_manual_posterior(self):
        """P(Y=0|x) e P(Y=1|x) calculados manualmente."""
        # P(Y=0)=0.75, P(Y=1)=0.25
        # p(x|Y=0)=0.2, p(x|Y=1)=0.6
        # P(x) = 0.2*0.75 + 0.6*0.25 = 0.15 + 0.15 = 0.30
        # P(Y=0|x) = 0.15/0.30 = 0.5
        # P(Y=1|x) = 0.15/0.30 = 0.5
        s0 = np.array([log(0.75) + log(0.2)])
        s1 = np.array([log(0.25) + log(0.6)])
        p0, p1 = compute_posteriors(s0, s1)
        assert p0[0] == pytest.approx(0.5)
        assert p1[0] == pytest.approx(0.5)

    def test_posteriors_sum_to_one(self):
        s0 = np.array([log(0.8) + log(0.01), log(0.8) + log(0.9)])
        s1 = np.array([log(0.2) + log(0.99), log(0.2) + log(0.1)])
        p0, p1 = compute_posteriors(s0, s1)
        np.testing.assert_allclose(p0 + p1, [1.0, 1.0], atol=1e-14)

    def test_extreme_scores_stable(self):
        """Logsumexp deve manter estabilidade com scores extremos."""
        s0 = np.array([-1000.0, 1000.0])
        s1 = np.array([-1001.0, 999.0])
        p0, p1 = compute_posteriors(s0, s1)
        assert np.all(np.isfinite(p0))
        assert np.all(np.isfinite(p1))
        np.testing.assert_allclose(p0 + p1, [1.0, 1.0], atol=1e-14)
        # s0 > s1 em ambos os pontos, então p0 > p1
        assert np.all(p0 > p1)

    def test_posteriors_in_unit_interval(self):
        rng = np.random.default_rng(42)
        s0 = rng.standard_normal(100)
        s1 = rng.standard_normal(100)
        p0, p1 = compute_posteriors(s0, s1)
        assert np.all(p0 >= 0) and np.all(p0 <= 1)
        assert np.all(p1 >= 0) and np.all(p1 <= 1)


# ──────────────────────────────────────────────
#  Testes de map_decision
# ──────────────────────────────────────────────


class TestMapDecision:
    def test_selects_higher_score(self):
        s0 = np.array([10.0, 5.0, 0.0])
        s1 = np.array([5.0, 10.0, 0.0])
        preds = map_decision(s0, s1)
        np.testing.assert_array_equal(preds, [0, 1, 0])

    def test_tie_goes_to_class_0(self):
        """Em caso de empate exato, argmax com > retorna 0."""
        preds = map_decision(np.array([5.0]), np.array([5.0]))
        assert preds[0] == 0


# ──────────────────────────────────────────────
#  Testes de analyze_univariate
# ──────────────────────────────────────────────


class TestAnalyzeUnivariate:
    def test_result_is_frozen(self):
        result = analyze_univariate(
            "test", np.array([1.0]), {0: 0.5, 1: 0.5},
            np.array([-1.0]), np.array([-2.0]),
        )
        assert isinstance(result, UnivariateResult)
        with pytest.raises(AttributeError):
            result.feature = "changed"  # type: ignore[misc]

    def test_likelihood_ratio_equals_exp_diff(self):
        log_pdf_0 = np.array([-2.0, -5.0])
        log_pdf_1 = np.array([-1.0, -3.0])
        result = analyze_univariate(
            "test", np.array([10.0, 20.0]), {0: 0.5, 1: 0.5},
            log_pdf_0, log_pdf_1,
        )
        expected_ratio = np.exp(log_pdf_1 - log_pdf_0)
        np.testing.assert_allclose(result.likelihood_ratio, expected_ratio)

    def test_posteriors_consistent_with_predictions(self):
        """predictions[i] == 1 iff posterior_class_1[i] > posterior_class_0[i]."""
        rng = np.random.default_rng(123)
        log_pdf_0 = rng.standard_normal(50)
        log_pdf_1 = rng.standard_normal(50)
        result = analyze_univariate(
            "test", np.arange(50, dtype=float), {0: 0.7, 1: 0.3},
            log_pdf_0, log_pdf_1,
        )
        # Onde posterior_1 > posterior_0, predictions deve ser 1
        expected = (result.posterior_class_1 > result.posterior_class_0).astype(int)
        np.testing.assert_array_equal(result.predictions, expected)


# ──────────────────────────────────────────────
#  Testes de find_continuous_boundaries
# ──────────────────────────────────────────────


class TestFindContinuousBoundaries:
    def test_single_root(self):
        """f(x) = x - 5 tem raiz em x=5."""
        boundaries = find_continuous_boundaries(lambda x: x - 5, 0, 10)
        assert len(boundaries) == 1
        assert boundaries[0] == pytest.approx(5.0, abs=1e-10)

    def test_two_roots(self):
        """f(x) = (x-2)(x-8) tem raízes em 2 e 8."""
        boundaries = find_continuous_boundaries(lambda x: (x - 2) * (x - 8), 0, 10)
        assert len(boundaries) == 2
        assert boundaries[0] == pytest.approx(2.0, abs=1e-10)
        assert boundaries[1] == pytest.approx(8.0, abs=1e-10)

    def test_no_roots(self):
        """f(x) = 1 não tem raiz."""
        boundaries = find_continuous_boundaries(lambda x: 1.0, 0, 10)
        assert len(boundaries) == 0

    def test_boundaries_make_scores_approximately_equal(self):
        """As fronteiras retornadas realmente tornam os scores aproximadamente iguais."""
        # Função score_diff = s1(x) - s0(x)
        def score_diff(x):
            return 2.0 * x - 8.0  # raiz em x=4.0

        boundaries = find_continuous_boundaries(score_diff, 0.0, 10.0)
        assert len(boundaries) == 1
        root = boundaries[0]
        # Tolerância coerente com o solver: |score_diff(root)| < 1e-6
        assert abs(score_diff(root)) < 1e-6

    def test_intervals_between_boundaries_receive_defined_decision(self):
        """Todos os intervalos entre fronteiras recebem regra definida."""
        # Duas fronteiras em x=2 e x=8 dividem o domínio em (-inf, 2), (2, 8), (8, inf)
        def diff(x):
            return -(x - 2.0) * (x - 8.0)  # >0 para x in (2, 8), <0 fora

        boundaries = find_continuous_boundaries(diff, 0.0, 10.0)
        assert len(boundaries) == 2
        b1, b2 = boundaries

        # Testa pontos no interior de cada um dos 3 intervalos
        test_points = [
            (b1 - 1.0, 0),  # x=1.0: diff < 0 -> classe 0
            ((b1 + b2) / 2.0, 1),  # x=5.0: diff > 0 -> classe 1
            (b2 + 1.0, 0),  # x=9.0: diff < 0 -> classe 0
        ]
        for pt, expected_class in test_points:
            s1 = diff(pt)
            s0 = 0.0
            dec = map_decision(np.array([s0]), np.array([s1]))
            assert dec[0] == expected_class

    def test_rejects_invalid_domain(self):
        with pytest.raises(ValueError, match="lower"):
            find_continuous_boundaries(lambda x: x, 10, 5)

    def test_rejects_small_grid(self):
        with pytest.raises(ValueError, match="n_grid"):
            find_continuous_boundaries(lambda x: x, 0, 10, n_grid=5)


# ──────────────────────────────────────────────
#  Testes de build_categorical_rule
# ──────────────────────────────────────────────


class TestBuildCategoricalRule:
    def test_basic_rule(self):
        probs_0 = {"A": 0.8, "B": 0.1, "C": 0.1}
        probs_1 = {"A": 0.1, "B": 0.8, "C": 0.1}
        priors = {0: 0.5, 1: 0.5}  # odds = 1
        rule = build_categorical_rule(probs_0, probs_1, priors)
        assert rule["A"] == 0  # Λ = 0.1/0.8 = 0.125 < 1
        assert rule["B"] == 1  # Λ = 0.8/0.1 = 8 > 1
        assert rule["C"] == 0  # Λ = 1 = 1, empate → classe 0

    def test_prior_domination(self):
        """Com prior muito forte, toda Λ < odds → tudo classe 0."""
        probs_0 = {"A": 0.3, "B": 0.3, "C": 0.4}
        probs_1 = {"A": 0.5, "B": 0.3, "C": 0.2}
        priors = {0: 0.9, 1: 0.1}  # odds = 9
        rule = build_categorical_rule(probs_0, probs_1, priors)
        # Λ(A)=5/3≈1.67 < 9, Λ(B)=1 < 9, Λ(C)=0.5 < 9
        assert all(v == 0 for v in rule.values())

    def test_missing_category_raises(self):
        with pytest.raises(ValueError, match="ausente"):
            build_categorical_rule({"A": 0.5, "B": 0.5}, {"A": 1.0}, {0: 0.5, 1: 0.5})


# ──────────────────────────────────────────────
#  Smoke values da especificação (integração com dados reais)
# ──────────────────────────────────────────────


@pytest.mark.regression
class TestSmokeValuesAge:
    """Smoke values esperados: likelihood ≈26.95 e ≈50.44, MAP ≈72.64."""

    @pytest.fixture(autouse=True)
    def _setup(self, data_split):
        y = data_split.y_train
        X = data_split.X_train
        self.priors = fit_class_priors(y)
        self.params_0 = fit_gaussian_mle(X.loc[y == 0, "age"].to_numpy())
        self.params_1 = fit_gaussian_mle(X.loc[y == 1, "age"].to_numpy())
        self.x_range = (float(X["age"].min()) - 10, float(X["age"].max()) + 10)

    def test_likelihood_boundaries(self):
        def diff(x):
            v = np.array([x])
            return float(gaussian_logpdf(v, self.params_1)[0] - gaussian_logpdf(v, self.params_0)[0])

        boundaries = find_continuous_boundaries(diff, self.x_range[0], self.x_range[1])
        assert len(boundaries) == 2, f"Esperadas 2 fronteiras Λ=1, encontradas {len(boundaries)}"
        assert boundaries[0] == pytest.approx(26.95, abs=1.0)
        assert boundaries[1] == pytest.approx(50.44, abs=1.0)

    def test_map_boundary(self):
        def diff(x):
            v = np.array([x])
            s1 = np.log(self.priors[1]) + float(gaussian_logpdf(v, self.params_1)[0])
            s0 = np.log(self.priors[0]) + float(gaussian_logpdf(v, self.params_0)[0])
            return s1 - s0

        boundaries = find_continuous_boundaries(diff, self.x_range[0], self.x_range[1])
        assert len(boundaries) >= 1, "Esperada pelo menos 1 fronteira MAP"
        # A fronteira MAP mais próxima do domínio observado
        closest = min(boundaries, key=lambda b: abs(b - 72.64))
        assert closest == pytest.approx(72.64, abs=2.0)

    def test_examples_table(self):
        values = np.array(AGE_EXAMPLES, dtype=float)
        log_0 = gaussian_logpdf(values, self.params_0)
        log_1 = gaussian_logpdf(values, self.params_1)
        result = analyze_univariate("age", values, self.priors, log_0, log_1)
        np.testing.assert_allclose(result.posterior_class_0 + result.posterior_class_1,
                                   np.ones(len(values)), atol=1e-14)
        assert np.all(np.isfinite(result.likelihood_ratio))


@pytest.mark.regression
class TestSmokeValuesDuration:
    """Smoke values: Λ=1 ≈315s, MAP ≈808s."""

    @pytest.fixture(autouse=True)
    def _setup(self, data_split):
        y = data_split.y_train
        X = data_split.X_train
        self.priors = fit_class_priors(y)
        self.params_0 = fit_gamma_mle(X.loc[y == 0, "duration"].to_numpy())
        self.params_1 = fit_gamma_mle(X.loc[y == 1, "duration"].to_numpy())
        self.dur_max = float(X["duration"].max())

    def test_likelihood_boundary(self):
        def diff(x):
            v = np.array([x])
            return float(gamma_logpdf(v, self.params_1)[0] - gamma_logpdf(v, self.params_0)[0])

        boundaries = find_continuous_boundaries(diff, 1.0, self.dur_max * 1.5)
        assert len(boundaries) >= 1
        closest = min(boundaries, key=lambda b: abs(b - 315))
        assert closest == pytest.approx(315, abs=20)

    def test_map_boundary(self):
        def diff(x):
            v = np.array([x])
            s1 = np.log(self.priors[1]) + float(gamma_logpdf(v, self.params_1)[0])
            s0 = np.log(self.priors[0]) + float(gamma_logpdf(v, self.params_0)[0])
            return s1 - s0

        boundaries = find_continuous_boundaries(diff, 1.0, self.dur_max * 1.5)
        assert len(boundaries) >= 1
        closest = min(boundaries, key=lambda b: abs(b - 808))
        assert closest == pytest.approx(808, abs=30)

    def test_examples_table(self):
        values = np.array(DURATION_EXAMPLES, dtype=float)
        log_0 = gamma_logpdf(values, self.params_0)
        log_1 = gamma_logpdf(values, self.params_1)
        result = analyze_univariate("duration", values, self.priors, log_0, log_1)
        np.testing.assert_allclose(result.posterior_class_0 + result.posterior_class_1,
                                   np.ones(len(values)), atol=1e-14)
        # duration=100 should be class 0, duration=1000 should be class 1
        assert result.predictions[0] == 0, "duration=100 deve ser classe 0"
        assert result.predictions[3] == 1, "duration=1000 deve ser classe 1"


@pytest.mark.regression
class TestSmokeValuesMarital:
    """Nenhuma categoria vence a prior → tudo classe 0."""

    @pytest.fixture(autouse=True)
    def _setup(self, data_split):
        y = data_split.y_train
        X = data_split.X_train
        self.priors = fit_class_priors(y)
        self.probs_0 = fit_categorical(X.loc[y == 0, "marital"])
        self.probs_1 = fit_categorical(X.loc[y == 1, "marital"])

    def test_divorced_and_single_have_lr_above_one(self):
        """divorced e single devem ter Λ > 1."""
        for cat in ("divorced", "single"):
            lr = self.probs_1[cat] / self.probs_0[cat]
            assert lr > 1, f"{cat}: Λ={lr:.4f} deveria ser > 1"

    def test_married_has_lr_below_one(self):
        """married deve ter Λ < 1."""
        lr = self.probs_1["married"] / self.probs_0["married"]
        assert lr < 1, f"married: Λ={lr:.4f} deveria ser < 1"

    def test_no_category_exceeds_prior_odds(self):
        """Nenhuma Λ deve superar P(Y=0)/P(Y=1)."""
        prior_odds = self.priors[0] / self.priors[1]
        for cat in MARITAL_CATEGORIES:
            lr = self.probs_1[cat] / self.probs_0[cat]
            assert lr < prior_odds, (
                f"{cat}: Λ={lr:.4f} >= prior_odds={prior_odds:.4f}"
            )

    def test_categorical_rule_all_class_0(self):
        rule = build_categorical_rule(self.probs_0, self.probs_1, self.priors)
        for cat, dec in rule.items():
            assert dec == 0, f"{cat} deveria ser classe 0, obteve {dec}"

    def test_examples_table(self):
        cats = list(MARITAL_CATEGORIES)
        log_0 = categorical_logpmf(pd.Series(cats), self.probs_0)
        log_1 = categorical_logpmf(pd.Series(cats), self.probs_1)
        result = analyze_univariate("marital", np.array(cats), self.priors, log_0, log_1)
        np.testing.assert_allclose(
            result.posterior_class_0 + result.posterior_class_1,
            np.ones(len(cats)), atol=1e-14,
        )
        # Tudo classe 0
        assert np.all(result.predictions == 0)


# ──────────────────────────────────────────────
#  Testes de propriedades algébricas
# ──────────────────────────────────────────────


class TestAlgebraicProperties:
    """Propriedades que devem valer independente dos dados."""

    def test_uniform_prior_posterior_equals_normalized_likelihood(self):
        """Com priors iguais, posterior é proporcional à likelihood."""
        priors = {0: 0.5, 1: 0.5}
        log_pdf_0 = np.array([-3.0, -1.0])
        log_pdf_1 = np.array([-1.0, -3.0])
        result = analyze_univariate("test", np.array([1.0, 2.0]), priors, log_pdf_0, log_pdf_1)
        # Com priors iguais, P(Y=c|x) ∝ p(x|Y=c)
        # Portanto, P(Y=1|x) > P(Y=0|x) iff p(x|Y=1) > p(x|Y=0)
        assert result.predictions[0] == 1  # log_pdf_1 > log_pdf_0
        assert result.predictions[1] == 0  # log_pdf_0 > log_pdf_1

    def test_extreme_prior_dominates(self):
        """Com prior ≈1 para classe 0, posterior é ≈1 para classe 0."""
        priors = {0: 0.999, 1: 0.001}
        # Likelihood favorece classe 1, mas prior domina
        log_pdf_0 = np.array([-10.0])
        log_pdf_1 = np.array([-4.0])
        result = analyze_univariate("test", np.array([1.0]), priors, log_pdf_0, log_pdf_1)
        # Λ = exp(6) ≈ 403, mas odds = 999
        # Como Λ < odds, classe 0 vence
        assert result.predictions[0] == 0

    def test_prior_odds_threshold(self):
        """Classe 1 vence iff Λ > P(Y=0)/P(Y=1)."""
        priors = {0: 0.8, 1: 0.2}
        odds = 4.0

        # Λ = exp(2) ≈ 7.39 > 4 → classe 1
        log_pdf_0_a = np.array([-5.0])
        log_pdf_1_a = np.array([-3.0])
        result_a = analyze_univariate("test", np.array([1.0]), priors, log_pdf_0_a, log_pdf_1_a)
        assert result_a.predictions[0] == 1

        # Λ = exp(1) ≈ 2.72 < 4 → classe 0
        log_pdf_0_b = np.array([-5.0])
        log_pdf_1_b = np.array([-4.0])
        result_b = analyze_univariate("test", np.array([1.0]), priors, log_pdf_0_b, log_pdf_1_b)
        assert result_b.predictions[0] == 0


# ──────────────────────────────────────────────
#  Teste de integração completa
# ──────────────────────────────────────────────


@pytest.mark.regression
class TestIntegration:
    """Testa que a pipeline completa roda sem erros nos dados reais."""

    def test_full_pipeline_runs(self, data_split):
        y = data_split.y_train
        X = data_split.X_train
        priors = fit_class_priors(y)

        # Age
        age_p0 = fit_gaussian_mle(X.loc[y == 0, "age"].to_numpy())
        age_p1 = fit_gaussian_mle(X.loc[y == 1, "age"].to_numpy())
        age_vals = np.array(AGE_EXAMPLES, dtype=float)
        result_age = analyze_univariate(
            "age", age_vals, priors,
            gaussian_logpdf(age_vals, age_p0),
            gaussian_logpdf(age_vals, age_p1),
        )
        assert result_age.feature == "age"
        assert len(result_age.values) == 4

        # Duration
        dur_p0 = fit_gamma_mle(X.loc[y == 0, "duration"].to_numpy())
        dur_p1 = fit_gamma_mle(X.loc[y == 1, "duration"].to_numpy())
        dur_vals = np.array(DURATION_EXAMPLES, dtype=float)
        result_dur = analyze_univariate(
            "duration", dur_vals, priors,
            gamma_logpdf(dur_vals, dur_p0),
            gamma_logpdf(dur_vals, dur_p1),
        )
        assert result_dur.feature == "duration"

        # Marital
        probs_0 = fit_categorical(X.loc[y == 0, "marital"])
        probs_1 = fit_categorical(X.loc[y == 1, "marital"])
        cats = list(MARITAL_CATEGORIES)
        result_mar = analyze_univariate(
            "marital", np.array(cats), priors,
            categorical_logpmf(pd.Series(cats), probs_0),
            categorical_logpmf(pd.Series(cats), probs_1),
        )
        assert result_mar.feature == "marital"
        assert len(result_mar.values) == 3
