"""Runner da análise Bayesiana univariada.

Execução: python -m src.run_univariate

Este script:
1. Consome o DataSplit via data.py.
2. Ajusta modelos por classe via distributions.py.
3. Calcula priors do treino.
4. Executa a análise univariada para age, duration e marital.
5. Encontra fronteiras (likelihood equality e MAP).
6. Gera tabelas CSV, JSON de parâmetros e figuras PNG.
7. Imprime interpretação e comparação qualitativa.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AGE_EXAMPLES,
    DATA_PATH,
    DURATION_EXAMPLES,
    MARITAL_CATEGORIES,
    UNIVARIATE_FIGURES_DIR,
    UNIVARIATE_METRICS_DIR,
)
from src.data import (
    load_bank_data,
    make_stratified_split,
    prepare_model_frame,
    validate_raw_data,
)
from src.distributions import (
    categorical_logpmf,
    fit_categorical,
    fit_class_priors,
    fit_gamma_mle,
    fit_gaussian_mle,
    gamma_logpdf,
    gaussian_logpdf,
)
from src.plotting import (
    plot_age_analysis,
    plot_duration_analysis,
    plot_marital_analysis,
)
from src.univariate import (
    analyze_univariate,
    build_categorical_rule,
    find_continuous_boundaries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Funções auxiliares de I/O
# ──────────────────────────────────────────────


def _save_result_csv(result, save_path: Path) -> Path:
    """Salva tabela de resultados univariados em CSV."""
    df = pd.DataFrame({
        result.feature: result.values,
        "p(x|Y=0)": result.likelihood_class_0,
        "p(x|Y=1)": result.likelihood_class_1,
        "Λ(x)": result.likelihood_ratio,
        "P(Y=0|x)": result.posterior_class_0,
        "P(Y=1|x)": result.posterior_class_1,
        "decisão": result.predictions,
    })
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format="%.8f")
    logger.info("Tabela salva: %s", save_path)
    return save_path.resolve()


def _print_result_table(result) -> None:
    """Imprime tabela formatada no console."""
    print(f"\n{'='*80}")
    print(f"  Análise Univariada: {result.feature}")
    print(f"{'='*80}")
    header = f"{'Valor':>10} | {'p(x|Y=0)':>12} | {'p(x|Y=1)':>12} | {'Λ(x)':>10} | {'P(0|x)':>8} | {'P(1|x)':>8} | {'h(x)':>4}"
    print(header)
    print("-" * len(header))
    for i in range(len(result.values)):
        v = result.values[i]
        v_str = f"{v}" if isinstance(v, str) else f"{v:>10.1f}"
        print(
            f"{v_str:>10} | {result.likelihood_class_0[i]:>12.8f} | "
            f"{result.likelihood_class_1[i]:>12.8f} | "
            f"{result.likelihood_ratio[i]:>10.5f} | "
            f"{result.posterior_class_0[i]:>8.5f} | "
            f"{result.posterior_class_1[i]:>8.5f} | "
            f"{result.predictions[i]:>4d}"
        )
    print()


def _print_boundaries(name: str, boundaries: list[float], kind: str) -> None:
    """Imprime fronteiras encontradas."""
    if boundaries:
        vals = ", ".join(f"{b:.4f}" for b in boundaries)
        print(f"  {kind} boundaries ({name}): {vals}")
    else:
        print(f"  Nenhuma {kind} boundary encontrada para {name}.")


# ──────────────────────────────────────────────
#  Pipeline principal
# ──────────────────────────────────────────────


def run() -> None:
    """Executa o pipeline completo da análise univariada."""
    # ── 1. Carregar e validar dados ──
    logger.info("Carregando dados...")
    df = load_bank_data(DATA_PATH)
    validate_raw_data(df, path=DATA_PATH)
    X, y = prepare_model_frame(df)
    split = make_stratified_split(X, y)

    X_train = split.X_train
    y_train = split.y_train

    # ── 2. Priors do treino ──
    priors = fit_class_priors(y_train)
    prior_odds = priors[0] / priors[1]
    logger.info("Priors: P(Y=0)=%.6f, P(Y=1)=%.6f", priors[0], priors[1])
    logger.info("Limiar MAP (odds ratio): %.5f", prior_odds)

    print(f"\n{'#'*80}")
    print(f"  Experimentos Bayesianos Univariados")
    print(f"{'#'*80}")
    print(f"\n  Priors do treino:")
    print(f"    P(Y=0) = {priors[0]:.6f}  ({(y_train == 0).sum()} amostras)")
    print(f"    P(Y=1) = {priors[1]:.6f}  ({(y_train == 1).sum()} amostras)")
    print(f"    Limiar MAP: Λ > {prior_odds:.5f}")

    # ── 3. Ajustar modelos por classe ──
    mask_0 = y_train == 0
    mask_1 = y_train == 1

    # Age — Gaussiana
    age_params_0 = fit_gaussian_mle(X_train.loc[mask_0, "age"].to_numpy())
    age_params_1 = fit_gaussian_mle(X_train.loc[mask_1, "age"].to_numpy())
    logger.info("Age class 0: μ=%.5f, σ²=%.5f", age_params_0.mean, age_params_0.variance)
    logger.info("Age class 1: μ=%.5f, σ²=%.5f", age_params_1.mean, age_params_1.variance)

    # Duration — Gamma
    dur_params_0 = fit_gamma_mle(X_train.loc[mask_0, "duration"].to_numpy())
    dur_params_1 = fit_gamma_mle(X_train.loc[mask_1, "duration"].to_numpy())
    logger.info("Duration class 0: k=%.5f, θ=%.5f", dur_params_0.shape, dur_params_0.scale)
    logger.info("Duration class 1: k=%.5f, θ=%.5f", dur_params_1.shape, dur_params_1.scale)

    # Marital — Categórica
    marital_probs_0 = fit_categorical(X_train.loc[mask_0, "marital"])
    marital_probs_1 = fit_categorical(X_train.loc[mask_1, "marital"])

    # ── 4. Análise de AGE ──
    print(f"\n{'─'*80}")
    print("  ANÁLISE: age (Normal por classe)")
    print(f"{'─'*80}")
    print(f"  Classe 0: μ={age_params_0.mean:.5f}, σ={np.sqrt(age_params_0.variance):.5f}")
    print(f"  Classe 1: μ={age_params_1.mean:.5f}, σ={np.sqrt(age_params_1.variance):.5f}")

    age_values = np.array(AGE_EXAMPLES, dtype=float)
    age_log_pdf_0 = gaussian_logpdf(age_values, age_params_0)
    age_log_pdf_1 = gaussian_logpdf(age_values, age_params_1)
    age_result = analyze_univariate("age", age_values, priors, age_log_pdf_0, age_log_pdf_1)
    _print_result_table(age_result)

    # Fronteiras age — likelihood equality
    age_range = (float(X_train["age"].min()), float(X_train["age"].max()))
    search_lower = max(age_range[0] - 10, 0)
    search_upper = age_range[1] + 10

    def age_likelihood_diff(x: float) -> float:
        v = np.array([x])
        return float(gaussian_logpdf(v, age_params_1)[0] - gaussian_logpdf(v, age_params_0)[0])

    def age_map_diff(x: float) -> float:
        v = np.array([x])
        s1 = np.log(priors[1]) + float(gaussian_logpdf(v, age_params_1)[0])
        s0 = np.log(priors[0]) + float(gaussian_logpdf(v, age_params_0)[0])
        return s1 - s0

    age_lr_boundaries = find_continuous_boundaries(age_likelihood_diff, search_lower, search_upper)
    age_map_boundaries = find_continuous_boundaries(age_map_diff, search_lower, search_upper)

    _print_boundaries("age", age_lr_boundaries, "Λ=1 (likelihood)")
    _print_boundaries("age", age_map_boundaries, "MAP")

    print("\n  Interpretação:")
    print("    • As duas Gaussianas têm médias próximas (~41 e ~42 anos)")
    print("    • A classe 1 tem variância maior, favorecendo caudas")
    print("    • Λ=1 em ~27 e ~50 anos mostra que likelihood favorece classe 1 nas caudas")
    print("    • Mas a prior forte (~88.4% classe 0) elimina a região intermediária")
    print("    • Fronteira MAP existe apenas nas caudas extremas (~73 anos)")
    print("    • Poder discriminativo limitado: age sozinha classifica quase tudo como 0")

    # ── 5. Análise de DURATION ──
    print(f"\n{'─'*80}")
    print("  ANÁLISE: duration (Gamma por classe)")
    print(f"{'─'*80}")
    print(f"  Classe 0: k={dur_params_0.shape:.5f}, θ={dur_params_0.scale:.5f}")
    print(f"  Classe 1: k={dur_params_1.shape:.5f}, θ={dur_params_1.scale:.5f}")

    dur_values = np.array(DURATION_EXAMPLES, dtype=float)
    dur_log_pdf_0 = gamma_logpdf(dur_values, dur_params_0)
    dur_log_pdf_1 = gamma_logpdf(dur_values, dur_params_1)
    dur_result = analyze_univariate("duration", dur_values, priors, dur_log_pdf_0, dur_log_pdf_1)
    _print_result_table(dur_result)

    # Fronteiras duration
    dur_search_lower = 1.0  # Gamma > 0
    dur_search_upper = float(X_train["duration"].max()) * 1.5

    def dur_likelihood_diff(x: float) -> float:
        v = np.array([x])
        return float(gamma_logpdf(v, dur_params_1)[0] - gamma_logpdf(v, dur_params_0)[0])

    def dur_map_diff(x: float) -> float:
        v = np.array([x])
        s1 = np.log(priors[1]) + float(gamma_logpdf(v, dur_params_1)[0])
        s0 = np.log(priors[0]) + float(gamma_logpdf(v, dur_params_0)[0])
        return s1 - s0

    dur_lr_boundaries = find_continuous_boundaries(dur_likelihood_diff, dur_search_lower, dur_search_upper)
    dur_map_boundaries = find_continuous_boundaries(dur_map_diff, dur_search_lower, dur_search_upper)

    _print_boundaries("duration", dur_lr_boundaries, "Λ=1 (likelihood)")
    _print_boundaries("duration", dur_map_boundaries, "MAP")

    print("\n  Interpretação:")
    print("    • Durações curtas são mais compatíveis com 'no'")
    print("    • Durações longas aumentam a evidência para 'yes'")
    print("    • A prior negativa (~88.4%) exige evidência forte antes da decisão positiva")
    print("    • Λ=1 perto de ~315s, mas MAP boundary só em ~808s")
    print("    • duration é a evidência isolada mais discriminativa")
    print("    • NOTA: duration é conhecida somente após a chamada (viés de seleção)")

    # ── 6. Análise de MARITAL ──
    print(f"\n{'─'*80}")
    print("  ANÁLISE: marital (Categórica com Laplace)")
    print(f"{'─'*80}")

    cats = list(MARITAL_CATEGORIES)
    cat_values = np.array(cats)
    cat_log_pdf_0 = categorical_logpmf(pd.Series(cats), marital_probs_0)
    cat_log_pdf_1 = categorical_logpmf(pd.Series(cats), marital_probs_1)
    marital_result = analyze_univariate("marital", cat_values, priors, cat_log_pdf_0, cat_log_pdf_1)
    _print_result_table(marital_result)

    # Regra categórica
    marital_rule = build_categorical_rule(marital_probs_0, marital_probs_1, priors)
    print(f"  Regra categórica h₃(marital):")
    for cat, dec in marital_rule.items():
        lr = marital_probs_1[cat] / marital_probs_0[cat]
        print(f"    {cat}: Λ={lr:.5f}, decisão={dec} ({'yes' if dec == 1 else 'no'})")

    print(f"\n  Limiar MAP = {prior_odds:.5f}")
    print("  Nenhuma categoria supera o limiar → h₃ decide classe 0 para todas")
    print("\n  Interpretação:")
    print("    • 'divorced' e 'single' têm Λ>1 (likelihood favorece classe 1)")
    print("    • 'married' tem Λ<1 (likelihood favorece classe 0)")
    print("    • Mas nenhuma Λ supera ~7.67, então a prior domina")
    print("    • Exemplo principal da diferença entre likelihood e posterior")

    # ── 7. Comparação qualitativa ──
    print(f"\n{'─'*80}")
    print("  COMPARAÇÃO QUALITATIVA DOS TRÊS ATRIBUTOS")
    print(f"{'─'*80}")
    print("  Critérios (evidências do treino, sem teste):")
    print()
    print("  1. Separação visual das distribuições:")
    print("     • duration: melhor separação (Gamma com formas e escalas distintas)")
    print("     • age: sobreposição quase total (médias ~41 vs ~42)")
    print("     • marital: diferenças proporcionais pequenas")
    print()
    print("  2. Amplitude do log-likelihood ratio:")
    print("     • duration: maior amplitude (cresce com x)")
    print("     • age: amplitude limitada (caudas apenas)")
    print("     • marital: log-Λ confinado a uma faixa estreita")
    print()
    print("  3. Existência e posição de regiões de decisão:")
    print("     • duration: região de decisão classe 1 a partir de ~808s")
    print("     • age: região de decisão classe 1 apenas em caudas extremas (~73+)")
    print("     • marital: nenhuma região de decisão classe 1")
    print()
    print("  4. Mudança das posteriors em relação às priors:")
    print("     • duration: para valores altos, P(1|x) pode superar 0.5")
    print("     • age: posterior quase inalterada em relação à prior")
    print("     • marital: posterior muda pouco, nunca inverte a prior")
    print()
    print("  5. Limitações da hipótese:")
    print("     • age: Gaussiana razoável mas sem poder discriminativo")
    print("     • duration: Gamma é superior à Exponencial (AIC), boa aderência")
    print("     • marital: Laplace smoothing necessário para robustez")
    print()
    print("  Conclusão: duration >> age > marital em poder discriminativo univariado")

    # ── Classificadores nomeados ──
    print(f"\n{'─'*80}")
    print("  CLASSIFICADORES UNIVARIADOS")
    print(f"{'─'*80}")
    if age_map_boundaries:
        boundaries_str = " e ".join(f"{b:.2f}" for b in age_map_boundaries)
        print(f"  h₁(age): classe 1 se age > {boundaries_str}; caso contrário classe 0")
    else:
        print("  h₁(age): classe 0 para todos os valores (nenhuma fronteira MAP)")
    if dur_map_boundaries:
        boundaries_str = " e ".join(f"{b:.2f}" for b in dur_map_boundaries)
        print(f"  h₂(duration): classe 1 se duration > {boundaries_str}; caso contrário classe 0")
    else:
        print("  h₂(duration): classe 0 para todos os valores")
    print(f"  h₃(marital): classe 0 para todas as categorias ({marital_rule})")

    # ── 8. Salvar artefatos ──
    logger.info("Salvando artefatos...")

    # CSVs
    _save_result_csv(age_result, UNIVARIATE_METRICS_DIR / "age_univariate_examples.csv")
    _save_result_csv(dur_result, UNIVARIATE_METRICS_DIR / "duration_univariate_examples.csv")

    # Marital CSV (valores string)
    marital_df = pd.DataFrame({
        "Categoria": marital_result.values,
        "P(a_k|Y=0)": marital_result.likelihood_class_0,
        "P(a_k|Y=1)": marital_result.likelihood_class_1,
        "Λ": marital_result.likelihood_ratio,
        "P(Y=0|a_k)": marital_result.posterior_class_0,
        "P(Y=1|a_k)": marital_result.posterior_class_1,
        "regra": marital_result.predictions,
    })
    marital_csv_path = UNIVARIATE_METRICS_DIR / "marital_univariate_examples.csv"
    marital_csv_path.parent.mkdir(parents=True, exist_ok=True)
    marital_df.to_csv(marital_csv_path, index=False, float_format="%.8f")
    logger.info("Tabela salva: %s", marital_csv_path)

    # JSON de parâmetros
    params_json = {
        "priors": {str(k): v for k, v in priors.items()},
        "prior_odds_ratio": prior_odds,
        "age": {
            "distribution": "Normal",
            "class_0": {"mean": age_params_0.mean, "variance": age_params_0.variance,
                        "std": float(np.sqrt(age_params_0.variance))},
            "class_1": {"mean": age_params_1.mean, "variance": age_params_1.variance,
                        "std": float(np.sqrt(age_params_1.variance))},
            "likelihood_boundaries": age_lr_boundaries,
            "map_boundaries": age_map_boundaries,
        },
        "duration": {
            "distribution": "Gamma",
            "class_0": {"shape": dur_params_0.shape, "scale": dur_params_0.scale,
                        "mean": dur_params_0.shape * dur_params_0.scale},
            "class_1": {"shape": dur_params_1.shape, "scale": dur_params_1.scale,
                        "mean": dur_params_1.shape * dur_params_1.scale},
            "likelihood_boundaries": dur_lr_boundaries,
            "map_boundaries": dur_map_boundaries,
        },
        "marital": {
            "distribution": "Categorical (Laplace α=1)",
            "class_0": marital_probs_0,
            "class_1": marital_probs_1,
            "categorical_rule": marital_rule,
        },
    }
    params_path = UNIVARIATE_METRICS_DIR / "distribution_parameters.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params_json, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    logger.info("Parâmetros salvos: %s", params_path)

    # Figuras
    plot_age_analysis(
        age_params_0, age_params_1, priors,
        X_train.loc[mask_0, "age"].to_numpy(),
        X_train.loc[mask_1, "age"].to_numpy(),
        age_lr_boundaries, age_map_boundaries,
        UNIVARIATE_FIGURES_DIR / "age_conditional_and_decision.png",
    )
    logger.info("Figura age salva.")

    plot_duration_analysis(
        dur_params_0, dur_params_1, priors,
        X_train.loc[mask_0, "duration"].to_numpy(),
        X_train.loc[mask_1, "duration"].to_numpy(),
        dur_lr_boundaries, dur_map_boundaries,
        UNIVARIATE_FIGURES_DIR / "duration_conditional_and_decision.png",
    )
    logger.info("Figura duration salva.")

    plot_marital_analysis(
        marital_probs_0, marital_probs_1, priors,
        UNIVARIATE_FIGURES_DIR / "marital_conditional_probabilities.png",
    )
    logger.info("Figura marital salva.")

    print(f"\n{'#'*80}")
    print("  Artefatos gerados com sucesso!")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    run()
