"""Visualizações da análise Bayesiana univariada.

Responsabilidades
-----------------
- Gerar figuras reprodutíveis para age, duration e marital.
- Seguir o padrão visual definido na especificação.

Não deve
--------
- Calcular likelihoods, posteriors ou fronteiras (isso é univariate.py).
- Ler ou escrever dados (isso é data.py / run_univariate.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para reprodutibilidade

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.distributions import (  # noqa: E402
    GammaParams,
    GaussianParams,
    gamma_logpdf,
    gaussian_logpdf,
)


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """Salva a matriz do teste: linhas reais, colunas preditas, contagens."""
    from src.evaluation import extract_confusion_components

    extract_confusion_components(cm)
    labels = ["0 - não aderiu", "1 - aderiu"]
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, int(cm.max())))
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Classe predita", ylabel="Classe real",
           title="Matriz de confusão — conjunto de teste")
    ax.grid(False)
    for (row, column), count in np.ndenumerate(cm):
        # Caixa branca mantém até células pequenas legíveis em qualquer cor.
        ax.text(column, row, str(int(count)), ha="center", va="center", color="black",
                fontsize=16, bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

# ──────────────────────────────────────────────
#  Estilo global
# ──────────────────────────────────────────────

_COLORS = {
    0: "#2563eb",   # azul — classe 0 (no)
    1: "#dc2626",   # vermelho — classe 1 (yes)
}
_LABELS = {0: "Classe 0 (no)", 1: "Classe 1 (yes)"}
_BOUNDARY_STYLES = {
    "likelihood": {"color": "#9333ea", "linestyle": "--", "linewidth": 1.5},
    "map": {"color": "#059669", "linestyle": "-", "linewidth": 2.0},
}


def _setup_style() -> None:
    """Configura estilo global do matplotlib."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


# ──────────────────────────────────────────────
#  Figura: age
# ──────────────────────────────────────────────


def plot_age_analysis(
    params_0: GaussianParams,
    params_1: GaussianParams,
    priors: dict[int, float],
    train_age_0: np.ndarray,
    train_age_1: np.ndarray,
    likelihood_boundaries: list[float],
    map_boundaries: list[float],
    save_path: Path,
    *,
    n_points: int = 1_000,
) -> Path:
    """Gera a figura de análise de age.

    Painel superior: histogramas do treino e densidades condicionais p(x|Y=c).
    Painel inferior: curvas ponderadas p(x|Y=c)·P(Y=c) com regiões de decisão.
    Ambos com fronteiras marcadas.

    Parameters
    ----------
    params_0, params_1 : GaussianParams
        Parâmetros Gaussianos por classe.
    priors : dict[int, float]
        Priors do treino.
    train_age_0, train_age_1 : np.ndarray
        Idades de treino por classe, usadas somente nos histogramas empíricos.
    likelihood_boundaries : list[float]
        Fronteiras onde Λ(x) = 1.
    map_boundaries : list[float]
        Fronteiras de decisão MAP.
    save_path : Path
        Caminho para salvar a figura PNG.
    n_points : int
        Pontos na grade de plotagem.

    Returns
    -------
    Path
        Caminho absoluto da figura salva.
    """
    _setup_style()

    train_age_0 = np.asarray(train_age_0, dtype=float)
    train_age_1 = np.asarray(train_age_1, dtype=float)
    if train_age_0.ndim != 1 or train_age_1.ndim != 1:
        raise ValueError("As idades de treino devem ser vetores unidimensionais.")
    if train_age_0.size == 0 or train_age_1.size == 0:
        raise ValueError("Cada classe deve possuir idades de treino para o histograma.")
    if not np.isfinite(np.concatenate([train_age_0, train_age_1])).all():
        raise ValueError("As idades de treino devem ser finitas.")

    observed_min = float(min(train_age_0.min(), train_age_1.min()))
    observed_max = float(max(train_age_0.max(), train_age_1.max()))
    x_min = observed_min - 2
    x_max = observed_max + 2
    x = np.linspace(x_min, x_max, n_points)

    pdf_0 = np.exp(gaussian_logpdf(x, params_0))
    pdf_1 = np.exp(gaussian_logpdf(x, params_1))
    weighted_0 = pdf_0 * priors[0]
    weighted_1 = pdf_1 * priors[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Análise Bayesiana Univariada — age", fontweight="bold")

    # ── Painel superior: comportamento empírico + densidades condicionais ──
    bins = np.arange(np.floor(observed_min) - 0.5, np.ceil(observed_max) + 1.5, 3)
    ax1.hist(train_age_0, bins=bins, density=True, alpha=0.22,
             color=_COLORS[0], label=f"{_LABELS[0]} (hist)")
    ax1.hist(train_age_1, bins=bins, density=True, alpha=0.22,
             color=_COLORS[1], label=f"{_LABELS[1]} (hist)")
    ax1.plot(x, pdf_0, color=_COLORS[0], linewidth=2, label=f"{_LABELS[0]} (Normal)")
    ax1.plot(x, pdf_1, color=_COLORS[1], linewidth=2, label=f"{_LABELS[1]} (Normal)")

    for b in likelihood_boundaries:
        ax1.axvline(b, label=f"Λ=1 ({b:.2f})", **_BOUNDARY_STYLES["likelihood"])

    ax1.set_ylabel("Densidade $p(x \\mid Y=c)$")
    ax1.set_title("Comportamento Empírico e Densidades Condicionais (Normal)")
    ax1.legend(loc="upper right", fontsize=8)

    # ── Painel inferior: curvas ponderadas + regiões ──
    ax2.plot(x, weighted_0, color=_COLORS[0], linewidth=2, label=f"{_LABELS[0]} · P(Y=0)")
    ax2.plot(x, weighted_1, color=_COLORS[1], linewidth=2, label=f"{_LABELS[1]} · P(Y=1)")

    # Regiões de decisão
    decision = (weighted_1 > weighted_0).astype(int)
    ax2.fill_between(
        x, 0, np.maximum(weighted_0, weighted_1),
        where=(decision == 0), color=_COLORS[0], alpha=0.08,
    )
    ax2.fill_between(
        x, 0, np.maximum(weighted_0, weighted_1),
        where=(decision == 1), color=_COLORS[1], alpha=0.08,
    )

    for b in map_boundaries:
        ax2.axvline(b, label=f"MAP ({b:.2f})", **_BOUNDARY_STYLES["map"])

    ax2.set_xlabel("age (anos)")
    ax2.set_ylabel("$p(x \\mid Y=c) \\cdot P(Y=c)$")
    ax2.set_title("Curvas Ponderadas e Regiões de Decisão MAP")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path.resolve()


# ──────────────────────────────────────────────
#  Figura: duration
# ──────────────────────────────────────────────


def plot_duration_analysis(
    params_0: GammaParams,
    params_1: GammaParams,
    priors: dict[int, float],
    train_duration_0: np.ndarray,
    train_duration_1: np.ndarray,
    likelihood_boundaries: list[float],
    map_boundaries: list[float],
    save_path: Path,
    *,
    n_points: int = 1_000,
) -> Path:
    """Gera a figura de análise de duration.

    Painel superior: histogramas normalizados + densidades Gamma.
    Painel inferior: curvas ponderadas + regiões de decisão.
    Limitado a P99 com nota do intervalo total.

    Parameters
    ----------
    params_0, params_1 : GammaParams
        Parâmetros Gamma por classe.
    priors : dict[int, float]
        Priors do treino.
    train_duration_0, train_duration_1 : np.ndarray
        Dados de treino por classe (para histogramas).
    likelihood_boundaries : list[float]
        Fronteiras onde Λ(x) = 1.
    map_boundaries : list[float]
        Fronteiras de decisão MAP.
    save_path : Path
        Caminho para salvar a figura PNG.
    n_points : int
        Pontos na grade de plotagem.

    Returns
    -------
    Path
        Caminho absoluto da figura salva.
    """
    _setup_style()

    all_durations = np.concatenate([train_duration_0, train_duration_1])
    observed_min = float(all_durations.min())
    p99 = float(np.percentile(all_durations, 99))
    total_max = float(all_durations.max())

    x_min = observed_min
    x_max_plot = p99 * 1.15
    x = np.linspace(x_min, x_max_plot, n_points)

    pdf_0 = np.exp(gamma_logpdf(x, params_0))
    pdf_1 = np.exp(gamma_logpdf(x, params_1))
    weighted_0 = pdf_0 * priors[0]
    weighted_1 = pdf_1 * priors[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Análise Bayesiana Univariada — duration", fontweight="bold")

    # ── Painel superior: histogramas + densidades ──
    bins = np.linspace(x_min, x_max_plot, 60)
    ax1.hist(train_duration_0, bins=bins, density=True, alpha=0.25,
             color=_COLORS[0], label=f"{_LABELS[0]} (hist)")
    ax1.hist(train_duration_1, bins=bins, density=True, alpha=0.25,
             color=_COLORS[1], label=f"{_LABELS[1]} (hist)")
    ax1.plot(x, pdf_0, color=_COLORS[0], linewidth=2, label=f"{_LABELS[0]} (Gamma)")
    ax1.plot(x, pdf_1, color=_COLORS[1], linewidth=2, label=f"{_LABELS[1]} (Gamma)")

    for b in likelihood_boundaries:
        if b <= x_max_plot:
            ax1.axvline(b, label=f"Λ=1 ({b:.1f}s)", **_BOUNDARY_STYLES["likelihood"])

    ax1.set_ylabel("Densidade")
    ax1.set_title("Densidades Condicionais (Gamma) e Histogramas")
    ax1.legend(loc="upper right", fontsize=8)

    # Nota do intervalo observado e do recorte visual da cauda
    ax1.annotate(
        f"Intervalo observado: [{observed_min:.0f}, {total_max:.0f}]s — "
        f"Exibindo até 1,15 × P99 ≈ {x_max_plot:.0f}s",
        xy=(0.02, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
    )

    # ── Painel inferior: curvas ponderadas + regiões ──
    ax2.plot(x, weighted_0, color=_COLORS[0], linewidth=2,
             label=f"{_LABELS[0]} · P(Y=0)")
    ax2.plot(x, weighted_1, color=_COLORS[1], linewidth=2,
             label=f"{_LABELS[1]} · P(Y=1)")

    decision = (weighted_1 > weighted_0).astype(int)
    ax2.fill_between(
        x, 0, np.maximum(weighted_0, weighted_1),
        where=(decision == 0), color=_COLORS[0], alpha=0.08,
    )
    ax2.fill_between(
        x, 0, np.maximum(weighted_0, weighted_1),
        where=(decision == 1), color=_COLORS[1], alpha=0.08,
    )

    for b in map_boundaries:
        if b <= x_max_plot:
            ax2.axvline(b, label=f"MAP ({b:.1f}s)", **_BOUNDARY_STYLES["map"])

    ax2.set_xlabel("duration (segundos)")
    ax2.set_ylabel("$p(x \\mid Y=c) \\cdot P(Y=c)$")
    ax2.set_title("Curvas Ponderadas e Regiões de Decisão MAP")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path.resolve()


# ──────────────────────────────────────────────
#  Figura: marital
# ──────────────────────────────────────────────


def plot_marital_analysis(
    probs_0: dict[str, float],
    probs_1: dict[str, float],
    priors: dict[int, float],
    save_path: Path,
) -> Path:
    """Gera a figura de análise de marital.

    Painel superior: barras agrupadas P(a_k|Y=c) por categoria e classe.
    Painel inferior: log-Λ por categoria com limiar log(P(Y=0)/P(Y=1)).

    Parameters
    ----------
    probs_0 : dict[str, float]
        P(a_k|Y=0) para cada categoria.
    probs_1 : dict[str, float]
        P(a_k|Y=1) para cada categoria.
    priors : dict[int, float]
        Priors do treino.
    save_path : Path
        Caminho para salvar a figura PNG.

    Returns
    -------
    Path
        Caminho absoluto da figura salva.
    """
    _setup_style()

    categories = list(probs_0.keys())
    p0 = np.array([probs_0[c] for c in categories])
    p1 = np.array([probs_1[c] for c in categories])
    log_lr = np.log(p1 / p0)
    log_threshold = np.log(priors[0] / priors[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    fig.suptitle("Análise Bayesiana Univariada — marital", fontweight="bold")

    # ── Painel superior: barras agrupadas ──
    x_pos = np.arange(len(categories))
    width = 0.35

    bars_0 = ax1.bar(x_pos - width / 2, p0, width, color=_COLORS[0],
                     label=_LABELS[0], edgecolor="white", linewidth=0.5)
    bars_1 = ax1.bar(x_pos + width / 2, p1, width, color=_COLORS[1],
                     label=_LABELS[1], edgecolor="white", linewidth=0.5)

    # Valores sobre as barras
    for bars in [bars_0, bars_1]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                     f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel("$P(a_k \\mid Y=c)$")
    ax1.set_title("Probabilidades Condicionais por Categoria")
    ax1.legend()

    # ── Painel inferior: log-Λ ──
    bar_colors = ["#059669" if ll > 0 else "#dc2626" for ll in log_lr]
    ax2.bar(x_pos, log_lr, 0.5, color=bar_colors, edgecolor="white",
            linewidth=0.5)
    ax2.axhline(log_threshold, color="#9333ea", linestyle="--", linewidth=2,
                label=f"log(P(Y=0)/P(Y=1)) = {log_threshold:.3f}")
    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    for i, ll in enumerate(log_lr):
        ax2.text(i, ll + 0.03 * np.sign(ll), f"{ll:.3f}",
                 ha="center", va="bottom" if ll >= 0 else "top", fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("$\\log \\Lambda(a_k)$")
    ax2.set_title("Log-Razão de Verossimilhanças vs. Limiar MAP")
    ax2.legend(loc="upper right")

    # Legenda explicativa
    legend_elements = [
        Patch(facecolor="#059669", label="log Λ > 0 (evidência p/ Y=1)"),
        Patch(facecolor="#dc2626", label="log Λ < 0 (evidência p/ Y=0)"),
    ]
    ax2.legend(handles=legend_elements + ax2.get_legend_handles_labels()[0][:1],
               loc="upper right", fontsize=8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path.resolve()
