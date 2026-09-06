"""Naive Bayes próprio para age, campaign e loan — Classificador Misto.

A independência condicional é uma aproximação operacional: age e loan
permanecem relacionados dentro das classes. Os ajustes usam apenas o treino;
scores e posteriores reutilizam os parâmetros aprendidos, sem novo ajuste.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy.special import logsumexp

from src.config import (
    CLASS_ORDER,
    FEATURE_COLUMNS,
    LAPLACE_ALPHA,
    LOAN_CATEGORIES,
    VARIANCE_FLOOR,
)
from src.distributions import (
    GammaParams,
    GaussianParams,
    fit_categorical,
    fit_class_priors,
    fit_gamma_mle,
    fit_gaussian_mle,
    gamma_logpdf,
    gaussian_logpdf,
)


class MixedNaiveBayes:
    """Normal para age, Gamma para campaign e categórica com Laplace.

    A API exige DataFrame com exatamente as três colunas do contrato, que
    são reordenadas explicitamente. As colunas de saída seguem (0, 1).
    Um fit malsucedido preserva integralmente o estado anterior: um modelo
    novo continua não ajustado e um refit não substitui o ajuste válido.
    """

    def __init__(
        self,
        *,
        alpha: float = LAPLACE_ALPHA,
        variance_floor: float = VARIANCE_FLOOR,
    ) -> None:
        self.alpha = alpha
        self.variance_floor = variance_floor
        self._validate_hyperparameters()
        self.is_fitted_ = False

    def _validate_hyperparameters(self) -> None:
        for name in ("alpha", "variance_floor"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} deve ser finito e positivo.")

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um DataFrame com nomes de colunas.")
        if (
            not X.columns.is_unique
            or len(X.columns) != len(FEATURE_COLUMNS)
            or set(X.columns) != set(FEATURE_COLUMNS)
        ):
            raise ValueError(f"X deve conter exatamente as colunas {FEATURE_COLUMNS}.")
        frame = X.loc[:, FEATURE_COLUMNS]
        if frame.isna().any().any():
            raise ValueError("X não pode conter nulos.")
        for feature in ("age", "campaign"):
            dtype = frame[feature].dtype
            if (
                not is_numeric_dtype(dtype)
                or is_bool_dtype(dtype)
                or is_complex_dtype(dtype)
            ):
                raise ValueError(f"{feature} deve conter números reais finitos.")
            if not np.isfinite(frame[feature].to_numpy(dtype=float)).all():
                raise ValueError(f"{feature} deve conter somente valores finitos.")
        if (frame["campaign"] <= 0).any():
            raise ValueError("campaign deve ser estritamente positivo.")
        if not frame["loan"].isin(LOAN_CATEGORIES).all():
            raise ValueError(f"loan deve pertencer às categorias {LOAN_CATEGORIES}.")
        return frame

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Modelo não ajustado; execute fit antes desta operação.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MixedNaiveBayes:
        """Estima todo o estado localmente e o publica somente após validação."""
        self._validate_hyperparameters()
        frame = self._validate_X(X)
        if not isinstance(y, pd.Series):
            raise TypeError("y deve ser uma Series com o mesmo índice de X.")
        if len(y) != len(frame) or not y.index.equals(frame.index):
            raise ValueError("X e y devem ter o mesmo índice e número de linhas.")
        priors = fit_class_priors(y)
        if not np.isclose(sum(priors.values()), 1.0):
            raise ValueError("A soma das priors deve ser 1.")

        class_count: dict[int, int] = {}
        class_log_prior: dict[int, float] = {}
        age_params: dict[int, GaussianParams] = {}
        campaign_params: dict[int, GammaParams] = {}
        loan_log_prob: dict[int, dict[str, float]] = {}
        for c in CLASS_ORDER:
            subset = frame.loc[y == c]
            class_count[c] = len(subset)
            class_log_prior[c] = float(np.log(priors[c]))
            age_params[c] = fit_gaussian_mle(
                subset["age"].to_numpy(dtype=float), variance_floor=self.variance_floor
            )
            campaign_params[c] = fit_gamma_mle(subset["campaign"].to_numpy(dtype=float))
            probabilities = fit_categorical(
                subset["loan"], LOAN_CATEGORIES, self.alpha
            )
            loan_log_prob[c] = {
                category: float(np.log(probabilities[category]))
                for category in LOAN_CATEGORIES
            }
            logs = [class_log_prior[c], *loan_log_prob[c].values()]
            if not np.isfinite(logs).all():
                raise ValueError("Priors e probabilidades categóricas devem ter logs finitos.")

        # As dataclasses e os ajustes de modelagem probabilística validam parâmetros, somas
        # categóricas e log-densidades. Nenhuma referência a X ou y é guardada.
        self.class_order_ = tuple(CLASS_ORDER)
        self.class_count_ = class_count
        self.class_log_prior_ = class_log_prior
        self.age_params_ = age_params
        self.campaign_params_ = campaign_params
        self.loan_log_prob_ = loan_log_prob
        self.feature_names_in_ = tuple(FEATURE_COLUMNS)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.is_fitted_ = True
        return self

    def joint_log_likelihood(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna log prior + log likelihoods, shape (n_samples, 2)."""
        self._check_fitted()
        frame = self._validate_X(X)
        age = frame["age"].to_numpy(dtype=float)
        campaign = frame["campaign"].to_numpy(dtype=float)
        scores = np.empty((len(frame), len(self.class_order_)), dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            for column, c in enumerate(self.class_order_):
                scores[:, column] = (
                    self.class_log_prior_[c]
                    + gaussian_logpdf(age, self.age_params_[c])
                    + gamma_logpdf(campaign, self.campaign_params_[c])
                    + frame["loan"].map(self.loan_log_prob_[c]).to_numpy(dtype=float)
                )
        if not np.isfinite(scores).all():
            raise ValueError("Score conjunto não finito; verifique os valores de X.")
        return scores

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Normaliza em log; a evidência é comum às classes no argmax."""
        scores = self.joint_log_likelihood(X)
        # Remover o máximo primeiro evita perda de precisão na subtração
        # quando ambas as classes têm scores muito negativos e próximos.
        scores = scores - np.max(scores, axis=1, keepdims=True)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna posteriores normalizadas, shape (n_samples, 2)."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Decisão MAP; empate exato favorece deterministicamente a classe 0."""
        columns = np.argmax(self.joint_log_likelihood(X), axis=1)
        return np.asarray(self.class_order_, dtype=int)[columns]

    def get_fitted_parameters(self) -> dict:
        """Cópia JSON-serializável para inspeção; metadados cabem ao runner."""
        self._check_fitted()
        return {
            "classes": list(self.class_order_),
            "class_count": {str(c): self.class_count_[c] for c in self.class_order_},
            "priors": {
                str(c): float(np.exp(self.class_log_prior_[c])) for c in self.class_order_
            },
            "age": {str(c): asdict(self.age_params_[c]) for c in self.class_order_},
            "campaign": {
                str(c): asdict(self.campaign_params_[c]) for c in self.class_order_
            },
            "loan": {
                str(c): {category: float(np.exp(log_prob)) for category, log_prob
                         in self.loan_log_prob_[c].items()}
                for c in self.class_order_
            },
            "alpha": float(self.alpha),
            "variance_floor": float(self.variance_floor),
            "feature_order": list(self.feature_names_in_),
        }
