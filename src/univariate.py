"""Análise Bayesiana univariada — RFC-0004.

Responsabilidades
-----------------
- Calcular log-scores, posteriores normalizadas e decisão MAP para um
  único atributo.
- Localizar fronteiras de decisão contínuas (likelihood equality e MAP).
- Construir regra categórica comparando Λ com o odds-ratio das priors.

Não deve
--------
- Ajustar distribuições (isso é RFC-0003 / distributions.py).
- Ler dados ou gerar figuras (isso é data.py / plotting.py).
- Usar o conjunto de teste.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


# ──────────────────────────────────────────────
#  Dataclass de resultado
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class UnivariateResult:
    """Resultado completo da análise univariada para um atributo.

    Attributes
    ----------
    feature : str
        Nome do atributo analisado.
    values : np.ndarray
        Pontos avaliados (numéricos ou índices de categorias).
    likelihood_class_0 : np.ndarray
        p(x|Y=0) em escala original para cada ponto.
    likelihood_class_1 : np.ndarray
        p(x|Y=1) em escala original para cada ponto.
    likelihood_ratio : np.ndarray
        Λ(x) = p(x|Y=1) / p(x|Y=0) para cada ponto.
    posterior_class_0 : np.ndarray
        P(Y=0|x) normalizada para cada ponto.
    posterior_class_1 : np.ndarray
        P(Y=1|x) normalizada para cada ponto.
    predictions : np.ndarray
        h_j(x) ∈ {0, 1} — decisão MAP para cada ponto.
    """

    feature: str
    values: np.ndarray
    likelihood_class_0: np.ndarray
    likelihood_class_1: np.ndarray
    likelihood_ratio: np.ndarray
    posterior_class_0: np.ndarray
    posterior_class_1: np.ndarray
    predictions: np.ndarray


# ──────────────────────────────────────────────
#  Funções de score e posterior (log-space)
# ──────────────────────────────────────────────


def compute_log_scores(
    log_prior: float,
    log_likelihoods: np.ndarray,
) -> np.ndarray:
    """Calcula s_c(x) = log P(Y=c) + log p(x|Y=c) para cada ponto.

    Parameters
    ----------
    log_prior : float
        log P(Y=c) — log-prior da classe.
    log_likelihoods : np.ndarray
        Vetor de log-densidades p(x|Y=c) para cada ponto avaliado.

    Returns
    -------
    np.ndarray
        Vetor de log-scores s_c(x).
    """
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)
    scores = log_prior + log_likelihoods
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Log-scores não finitos; verifique priors e likelihoods."
        )
    return scores


def compute_posteriors(
    log_scores_0: np.ndarray,
    log_scores_1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula posteriors normalizadas via logsumexp.

    P(Y=c|x) = exp(s_c(x) - logsumexp(s_0(x), s_1(x)))

    Parameters
    ----------
    log_scores_0 : np.ndarray
        s_0(x) para cada ponto.
    log_scores_1 : np.ndarray
        s_1(x) para cada ponto.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (P(Y=0|x), P(Y=1|x)) ambas ∈ [0, 1] e somando 1.
    """
    log_scores_0 = np.asarray(log_scores_0, dtype=float)
    log_scores_1 = np.asarray(log_scores_1, dtype=float)

    # Stack para logsumexp ao longo do eixo 0
    stacked = np.vstack([log_scores_0, log_scores_1])
    log_norm = logsumexp(stacked, axis=0)

    posterior_0 = np.exp(log_scores_0 - log_norm)
    posterior_1 = np.exp(log_scores_1 - log_norm)

    if not np.all(np.isfinite(posterior_0)) or not np.all(
        np.isfinite(posterior_1)
    ):
        raise ValueError("Posteriors não finitas após normalização.")

    return posterior_0, posterior_1


def map_decision(
    log_scores_0: np.ndarray,
    log_scores_1: np.ndarray,
) -> np.ndarray:
    """Aplica a regra MAP: h(x) = argmax_c s_c(x).

    Parameters
    ----------
    log_scores_0 : np.ndarray
        s_0(x) para cada ponto.
    log_scores_1 : np.ndarray
        s_1(x) para cada ponto.

    Returns
    -------
    np.ndarray
        Vetor de decisões ∈ {0, 1}.
    """
    return (np.asarray(log_scores_1) > np.asarray(log_scores_0)).astype(int)


# ──────────────────────────────────────────────
#  Análise univariada completa
# ──────────────────────────────────────────────


def analyze_univariate(
    feature: str,
    values: np.ndarray,
    priors: dict[int, float],
    log_pdf_0: np.ndarray,
    log_pdf_1: np.ndarray,
) -> UnivariateResult:
    """Executa a análise univariada completa para um atributo.

    Opera inteiramente em log-space para estabilidade. Exponencia
    likelihoods e ratios somente para exibição na tabela.

    Parameters
    ----------
    feature : str
        Nome do atributo ('age', 'duration', 'marital').
    values : np.ndarray
        Pontos de avaliação.
    priors : dict[int, float]
        {0: P(Y=0), 1: P(Y=1)} do treino.
    log_pdf_0 : np.ndarray
        log p(x|Y=0) para cada valor.
    log_pdf_1 : np.ndarray
        log p(x|Y=1) para cada valor.

    Returns
    -------
    UnivariateResult
        Resultado completo com likelihoods, Λ, posteriors e decisão.
    """
    values = np.asarray(values)
    log_pdf_0 = np.asarray(log_pdf_0, dtype=float)
    log_pdf_1 = np.asarray(log_pdf_1, dtype=float)

    log_prior_0 = np.log(priors[0])
    log_prior_1 = np.log(priors[1])

    # Log-scores: s_c(x) = log P(Y=c) + log p(x|Y=c)
    log_s0 = compute_log_scores(log_prior_0, log_pdf_0)
    log_s1 = compute_log_scores(log_prior_1, log_pdf_1)

    # Posteriors normalizadas via logsumexp
    posterior_0, posterior_1 = compute_posteriors(log_s0, log_s1)

    # Decisão MAP
    predictions = map_decision(log_s0, log_s1)

    # Likelihoods em escala original (para exibição)
    likelihood_0 = np.exp(log_pdf_0)
    likelihood_1 = np.exp(log_pdf_1)

    # Razão de verossimilhanças: Λ(x) = p(x|Y=1) / p(x|Y=0)
    # Calculada em log para estabilidade, depois exponenciada
    log_ratio = log_pdf_1 - log_pdf_0
    likelihood_ratio = np.exp(log_ratio)

    return UnivariateResult(
        feature=feature,
        values=values,
        likelihood_class_0=likelihood_0,
        likelihood_class_1=likelihood_1,
        likelihood_ratio=likelihood_ratio,
        posterior_class_0=posterior_0,
        posterior_class_1=posterior_1,
        predictions=predictions,
    )


# ──────────────────────────────────────────────
#  Fronteiras contínuas
# ──────────────────────────────────────────────


def find_continuous_boundaries(
    log_score_diff_fn: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    n_grid: int = 5_000,
) -> list[float]:
    """Encontra todas as fronteiras onde log_score_diff muda de sinal.

    Procedimento (conforme RFC-0004):
    1. Avaliar a diferença em grade densa no intervalo [lower, upper].
    2. Detectar trocas de sinal.
    3. Refinar cada raiz com brentq.
    4. Não assumir previamente quantas raízes existem.

    Parameters
    ----------
    log_score_diff_fn : callable
        Função f(x) = s_1(x) - s_0(x) (ou log L1 - log L0 para equality).
        Raízes são onde f(x) = 0.
    lower : float
        Limite inferior do domínio de busca.
    upper : float
        Limite superior do domínio de busca.
    n_grid : int
        Número de pontos na grade inicial.

    Returns
    -------
    list[float]
        Fronteiras ordenadas, com precisão de brentq (≈1e-12).
    """
    if lower >= upper:
        raise ValueError(f"lower ({lower}) deve ser menor que upper ({upper}).")
    if n_grid < 10:
        raise ValueError(f"n_grid ({n_grid}) deve ser >= 10.")

    grid = np.linspace(lower, upper, n_grid)
    diffs = np.array([log_score_diff_fn(x) for x in grid])

    # Detectar trocas de sinal
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]

    boundaries: list[float] = []
    for idx in sign_changes:
        a, b = float(grid[idx]), float(grid[idx + 1])
        try:
            root = brentq(log_score_diff_fn, a, b, xtol=1e-12, rtol=1e-14)
            boundaries.append(root)
        except ValueError:
            # Caso extremo: sign change mas sem raiz estrita (ponto tangente)
            continue

    return sorted(boundaries)


# ──────────────────────────────────────────────
#  Regra categórica
# ──────────────────────────────────────────────


def build_categorical_rule(
    probabilities_0: dict[str, float],
    probabilities_1: dict[str, float],
    priors: dict[int, float],
) -> dict[str, int]:
    """Constrói a regra de decisão MAP para cada categoria.

    Para cada a_k, classe 1 é escolhida sse:
        Λ(a_k) = P(a_k|Y=1) / P(a_k|Y=0) > P(Y=0) / P(Y=1)

    Parameters
    ----------
    probabilities_0 : dict[str, float]
        P(a_k|Y=0) para cada categoria.
    probabilities_1 : dict[str, float]
        P(a_k|Y=1) para cada categoria.
    priors : dict[int, float]
        {0: P(Y=0), 1: P(Y=1)}.

    Returns
    -------
    dict[str, int]
        {categoria: decisão} com decisão ∈ {0, 1}.
    """
    prior_odds = priors[0] / priors[1]
    rule: dict[str, int] = {}

    for category in probabilities_0:
        if category not in probabilities_1:
            raise ValueError(
                f"Categoria '{category}' ausente em probabilities_1."
            )
        lr = probabilities_1[category] / probabilities_0[category]
        rule[category] = 1 if lr > prior_odds else 0

    return rule
