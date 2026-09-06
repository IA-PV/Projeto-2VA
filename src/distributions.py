"""Estimação e log-densidades explícitas — Modelagem Probabilística.

Os ajustes recebem exclusivamente observações de treino (por classe para os
atributos). Este módulo não lê dados nem realiza splits ou classificação.
SciPy resolve apenas o MLE Gamma com localização fixa em zero, fornece gammaln
e as CDFs/estatística KS de diagnóstico. As log-densidades são implementadas
aqui; combinação de evidências e decisão pertencem aos módulos de classificação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from src.config import CLASS_ORDER, LAPLACE_ALPHA, LOAN_CATEGORIES, VARIANCE_FLOOR


def _positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} deve ser finito e positivo; recebido: {value}.")


def _numeric_values(values: np.ndarray, *, fitting: bool = False) -> np.ndarray:
    """Valida vetores reais finitos; ajustes exigem ao menos uma observação."""
    if np.iscomplexobj(values):
        raise ValueError("values deve conter números reais, não complexos.")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values deve conter números reais finitos.") from exc
    if result.ndim != 1:
        raise ValueError("values deve ser um vetor unidimensional.")
    if fitting and result.size == 0:
        raise ValueError("O ajuste exige um vetor não vazio.")
    if not np.isfinite(result).all():
        raise ValueError("values deve conter somente valores finitos.")
    return result


def _finite_logpdf(result: np.ndarray) -> np.ndarray:
    if not np.isfinite(result).all():
        raise ValueError("Log-densidade não finita; verifique valores e parâmetros.")
    return result


@dataclass(frozen=True)
class GaussianParams:
    """Média e variância (não desvio-padrão) da Normal."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("mean deve ser finita.")
        _positive_finite(self.variance, "variance")


@dataclass(frozen=True)
class GammaParams:
    """Forma k e escala theta da Gamma; localização sempre zero."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        _positive_finite(self.shape, "shape")
        _positive_finite(self.scale, "scale")


@dataclass(frozen=True)
class ExponentialParams:
    """Taxa lambda da hipótese comparativa; localização sempre zero."""

    rate: float

    def __post_init__(self) -> None:
        _positive_finite(self.rate, "rate")


def fit_gaussian_mle(
    values: np.ndarray, *, variance_floor: float = VARIANCE_FLOOR
) -> GaussianParams:
    """Estima média e variância MLE (ddof=0) de um vetor de treino.

    O piso positivo só substitui variância igual a zero na precisão de float;
    variâncias positivas, mesmo menores que o piso, são preservadas.
    """
    x = _numeric_values(values, fitting=True)
    _positive_finite(variance_floor, "variance_floor")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(x))
        variance = float(np.var(x, ddof=0))
    params = GaussianParams(mean, variance_floor if variance == 0 else variance)
    gaussian_logpdf(x, params)
    return params


def gaussian_logpdf(values: np.ndarray, params: GaussianParams) -> np.ndarray:
    """Retorna log p(x|c) explicitamente, preservando a ordem do vetor.

    Entradas e resultados não finitos geram ValueError, sem clipping de caudas.
    """
    x = _numeric_values(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = (x - params.mean) / np.sqrt(params.variance)
        result = -0.5 * (np.log(2 * np.pi) + np.log(params.variance))
        result = result - 0.5 * standardized**2
    return _finite_logpdf(result)


def fit_gamma_mle(values: np.ndarray) -> GammaParams:
    """Ajusta Gamma no treino usando stats.gamma.fit(values, floc=0).

    Exige valores estritamente positivos e não constantes: uma amostra
    constante não admite MLE Gamma finito. Valida localização, parâmetros,
    média k*theta e log-densidades de todas as observações usadas no ajuste.
    """
    x = _numeric_values(values, fitting=True)
    if np.any(x <= 0):
        raise ValueError("Gamma exige duration estritamente positivo (values > 0).")
    if np.all(x == x[0]):
        raise ValueError("Gamma não possui MLE finito para valores constantes.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            shape, location, scale = stats.gamma.fit(x, floc=0)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        raise ValueError("Falha no ajuste MLE Gamma com localização zero.") from exc
    if location != 0:
        raise ValueError("A localização da Gamma deve ser exatamente zero.")
    params = GammaParams(float(shape), float(scale))
    fitted_mean = params.shape * params.scale
    if not np.isfinite(fitted_mean) or not np.isclose(
        fitted_mean, np.mean(x), rtol=1e-7, atol=0
    ):
        raise ValueError("A média Gamma (shape * scale) diverge da média empírica.")
    gamma_logpdf(x, params)
    return params


def gamma_logpdf(values: np.ndarray, params: GammaParams) -> np.ndarray:
    """Calcula a log-densidade Gamma via gammaln, para valores > 0.

    Zero e negativos violam o contrato de duration e geram erro descritivo.
    Não calcula gamma(k) nem log(pdf(x)).
    """
    x = _numeric_values(values)
    if np.any(x <= 0):
        raise ValueError("Gamma exige duration estritamente positivo (values > 0).")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = (
            (params.shape - 1) * np.log(x)
            - x / params.scale
            - params.shape * np.log(params.scale)
            - gammaln(params.shape)
        )
    return _finite_logpdf(result)


def fit_exponential_mle(values: np.ndarray) -> ExponentialParams:
    """Estima lambda=1/média no treino, somente para comparação com Gamma.

    Aceita x >= 0; a média deve ser positiva para existir uma taxa finita.
    """
    x = _numeric_values(values, fitting=True)
    if np.any(x < 0):
        raise ValueError("Exponencial exige values >= 0.")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(x))
    _positive_finite(mean, "Média da Exponencial")
    params = ExponentialParams(1 / mean)
    exponential_logpdf(x, params)
    return params


def exponential_logpdf(values: np.ndarray, params: ExponentialParams) -> np.ndarray:
    """Log-densidade comparativa: log(lambda) - lambda*x, para x >= 0."""
    x = _numeric_values(values)
    if np.any(x < 0):
        raise ValueError("Exponencial exige values >= 0.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.log(params.rate) - params.rate * x
    return _finite_logpdf(result)


def _categorical_values(values: pd.Series, categories: tuple[str, ...]) -> pd.Series:
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("categories deve ser não vazio e sem categorias duplicadas.")
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Valores categóricos não podem conter nulos.")
    unknown = series[~series.isin(categories)].unique().tolist()
    if unknown:
        raise ValueError(
            f"Categorias desconhecidas: {unknown}. Categorias válidas: {categories}."
        )
    return series


def fit_categorical(
    values: pd.Series,
    categories: tuple[str, ...] = LOAN_CATEGORIES,
    alpha: float = LAPLACE_ALPHA,
) -> dict[str, float]:
    """Estima (N_k + alpha)/(N + alpha*K), incluindo categorias ausentes.

    Recebe somente a série de treino de uma classe; alpha deve ser positivo.
    O modelo principal usa o domínio congelado de loan e alpha=1.
    """
    series = _categorical_values(values, categories)
    if series.empty:
        raise ValueError("O ajuste categórico exige uma série não vazia.")
    _positive_finite(alpha, "alpha")
    counts = series.value_counts().reindex(categories, fill_value=0)
    probabilities = (counts + alpha) / (len(series) + alpha * len(categories))
    result = {category: float(probabilities[category]) for category in categories}
    categorical_logpmf(series, result)
    return result


def categorical_logpmf(values: pd.Series, probabilities: dict[str, float]) -> np.ndarray:
    """Retorna log-probabilidades categóricas; rejeita categorias desconhecidas.

    O dicionário deve ser uma distribuição estritamente positiva e normalizada,
    como a retornada por fit_categorical. Densidade contínua não se aplica aqui.
    """
    series = _categorical_values(values, tuple(probabilities))
    probs = np.asarray(list(probabilities.values()), dtype=float)
    if not np.isfinite(probs).all() or np.any(probs <= 0) or np.any(probs > 1):
        raise ValueError("Probabilidades categóricas devem ser finitas e positivas, até 1.")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("A soma das probabilidades categóricas deve ser 1.")
    return np.log(series.map(probabilities).to_numpy(dtype=float))


def fit_class_priors(y_train: pd.Series) -> dict[int, float]:
    """Estima N_c/N nas classes congeladas, sem suavização ou balanceamento.

    Ambas as classes devem estar presentes no alvo de treinamento.
    """
    y = pd.Series(y_train)
    if y.empty or y.isna().any() or not y.isin(CLASS_ORDER).all():
        raise ValueError(f"y_train deve ser não vazio, sem nulos e conter classes {CLASS_ORDER}.")
    counts = y.value_counts().reindex(CLASS_ORDER, fill_value=0)
    if (counts == 0).any():
        raise ValueError("Todas as classes devem estar presentes no treino.")
    return {c: float(counts[c] / len(y)) for c in CLASS_ORDER}


def aic(log_likelihoods: np.ndarray, q: int) -> float:
    """AIC=2*q-2*sum(log p(x_i)); usar no mesmo atributo/classe de treino.

    q conta parâmetros livres: Normal=2, Gamma=2 e Exponencial=1.
    A localização fixada em zero não conta como parâmetro livre.
    """
    logs = _numeric_values(log_likelihoods, fitting=True)
    if not isinstance(q, (int, np.integer)) or isinstance(q, bool) or q < 1:
        raise ValueError("q deve ser um inteiro positivo de parâmetros livres.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(2 * q - 2 * np.sum(logs))
    if not np.isfinite(result):
        raise ValueError("AIC não finito; verifique as log-verossimilhanças.")
    return result


def compare_campaign_distributions(values: np.ndarray) -> pd.DataFrame:
    """Compara Exponencial e Gamma na mesma amostra de campaign de treino.

    KS é apenas diagnóstico relativo com parâmetros estimados; seu p-valor
    não é reportado como evidência de aderência. Não escolhe um classificador.
    """
    x = _numeric_values(values, fitting=True)
    exponential = fit_exponential_mle(x)
    gamma = fit_gamma_mle(x)
    candidates = (
        ("Exponencial", 1, exponential_logpdf(x, exponential),
         stats.expon(scale=1 / exponential.rate).cdf),
        ("Gamma", 2, gamma_logpdf(x, gamma),
         stats.gamma(a=gamma.shape, loc=0, scale=gamma.scale).cdf),
    )
    rows = []
    for name, q, logs, cdf in candidates:
        criterion = aic(logs, q)
        ks = float(stats.kstest(x, cdf).statistic)
        if not np.isfinite(ks):
            raise ValueError("Estatística KS não finita.")
        rows.append({
            "distribution": name, "q": q,
            "log_likelihood": float(np.sum(logs)), "aic": criterion, "ks": ks,
        })
    return pd.DataFrame(rows)
