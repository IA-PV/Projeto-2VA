"""Runner da análise Bayesiana univariada.

Execução: python -m src.run_univariate

Este script:
1. Consome o DataSplit via data.py.
2. Ajusta modelos por classe via distributions.py.
3. Calcula priors do treino.
4. Executa a análise univariada para age, campaign e loan.
5. Encontra fronteiras (likelihood equality e MAP).
6. Quantifica a associação age × loan dentro de cada classe.
7. Gera tabelas CSV, JSON de parâmetros e figuras PNG.
8. Imprime interpretação e comparação qualitativa.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    AGE_EXAMPLES,
    CAMPAIGN_EXAMPLES,
    DATA_PATH,
    LOAN_CATEGORIES,
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
from src.evaluation import summarize_age_by_loan
from src.plotting import (
    plot_age_analysis,
    plot_campaign_analysis,
    plot_loan_analysis,
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
    header = f"{'Valor':>10} | {'p(x|Y=0)':>12} | {'p(x|Y=1)':>12} | {'Lambda(x)':>10} | {'P(0|x)':>8} | {'P(1|x)':>8} | {'h(x)':>4}"
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
    print(f"    Limiar MAP: Lambda > {prior_odds:.5f}")

    # ── 3. Ajustar modelos por classe ──
    mask_0 = y_train == 0
    mask_1 = y_train == 1

    # Age — Gaussiana
    age_params_0 = fit_gaussian_mle(X_train.loc[mask_0, "age"].to_numpy())
    age_params_1 = fit_gaussian_mle(X_train.loc[mask_1, "age"].to_numpy())
    logger.info("Age class 0: μ=%.5f, σ²=%.5f", age_params_0.mean, age_params_0.variance)
    logger.info("Age class 1: μ=%.5f, σ²=%.5f", age_params_1.mean, age_params_1.variance)

    # Campaign — Gamma
    camp_params_0 = fit_gamma_mle(X_train.loc[mask_0, "campaign"].to_numpy(dtype=float))
    camp_params_1 = fit_gamma_mle(X_train.loc[mask_1, "campaign"].to_numpy(dtype=float))
    logger.info("Campaign class 0: k=%.5f, θ=%.5f", camp_params_0.shape, camp_params_0.scale)
    logger.info("Campaign class 1: k=%.5f, θ=%.5f", camp_params_1.shape, camp_params_1.scale)

    # Loan — Categórica
    loan_probs_0 = fit_categorical(X_train.loc[mask_0, "loan"])
    loan_probs_1 = fit_categorical(X_train.loc[mask_1, "loan"])

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

    _print_boundaries("age", age_lr_boundaries, "Lambda=1 (likelihood)")
    _print_boundaries("age", age_map_boundaries, "MAP")

    print("\n  Interpretação:")
    print("    • As duas Gaussianas têm médias próximas (~41 e ~42 anos)")
    print("    • A classe 1 tem variância maior, favorecendo caudas")
    print("    • Fronteiras de likelihood nas caudas; prior forte elimina a região intermediária")
    print("    • Poder discriminativo limitado: age sozinha classifica quase tudo como 0")

    # ── 5. Análise de CAMPAIGN ──
    print(f"\n{'─'*80}")
    print("  ANÁLISE: campaign (Gamma por classe)")
    print(f"{'─'*80}")
    print(f"  Classe 0: k={camp_params_0.shape:.5f}, θ={camp_params_0.scale:.5f}")
    print(f"  Classe 1: k={camp_params_1.shape:.5f}, θ={camp_params_1.scale:.5f}")

    camp_values = np.array(CAMPAIGN_EXAMPLES, dtype=float)
    camp_log_pdf_0 = gamma_logpdf(camp_values, camp_params_0)
    camp_log_pdf_1 = gamma_logpdf(camp_values, camp_params_1)
    camp_result = analyze_univariate("campaign", camp_values, priors, camp_log_pdf_0, camp_log_pdf_1)
    _print_result_table(camp_result)

    # Fronteiras campaign
    camp_search_lower = 0.5  # Gamma > 0
    camp_search_upper = float(X_train["campaign"].max()) * 1.5

    def camp_likelihood_diff(x: float) -> float:
        v = np.array([x])
        return float(gamma_logpdf(v, camp_params_1)[0] - gamma_logpdf(v, camp_params_0)[0])

    def camp_map_diff(x: float) -> float:
        v = np.array([x])
        s1 = np.log(priors[1]) + float(gamma_logpdf(v, camp_params_1)[0])
        s0 = np.log(priors[0]) + float(gamma_logpdf(v, camp_params_0)[0])
        return s1 - s0

    camp_lr_boundaries = find_continuous_boundaries(camp_likelihood_diff, camp_search_lower, camp_search_upper)
    camp_map_boundaries = find_continuous_boundaries(camp_map_diff, camp_search_lower, camp_search_upper)

    _print_boundaries("campaign", camp_lr_boundaries, "Lambda=1 (likelihood)")
    _print_boundaries("campaign", camp_map_boundaries, "MAP")

    print("\n  Interpretação:")
    print("    • campaign conta os contatos nesta campanha (sempre ≥ 1)")
    print("    • Distribuição assimétrica à direita, com cauda longa")
    print("    • A prior negativa (~88.4%) exige evidência forte antes da decisão positiva")
    print("    • NOTA: campaign é uma variável disponível durante/após a campanha")

    # ── 6. Análise de LOAN ──
    print(f"\n{'─'*80}")
    print("  ANÁLISE: loan (Categórica com Laplace)")
    print(f"{'─'*80}")

    cats = list(LOAN_CATEGORIES)
    cat_values = np.array(cats)
    cat_log_pdf_0 = categorical_logpmf(pd.Series(cats), loan_probs_0)
    cat_log_pdf_1 = categorical_logpmf(pd.Series(cats), loan_probs_1)
    loan_result = analyze_univariate("loan", cat_values, priors, cat_log_pdf_0, cat_log_pdf_1)
    _print_result_table(loan_result)

    # Regra categórica
    loan_rule = build_categorical_rule(loan_probs_0, loan_probs_1, priors)
    print(f"  Regra categórica h3(loan):")
    for cat, dec in loan_rule.items():
        lr = loan_probs_1[cat] / loan_probs_0[cat]
        print(f"    {cat}: Lambda={lr:.5f}, decisão={dec} ({'yes' if dec == 1 else 'no'})")

    print(f"\n  Limiar MAP = {prior_odds:.5f}")
    print("\\n  Interpretação:")
    print("    • loan é uma variável binária (no/yes)")
    print("    • Compara a proporção de empréstimos pessoais entre aderentes e não aderentes")
    print("    • A prior forte (~88.4% classe 0) pode dominar a evidência de loan")

    # ── 7. Comparação qualitativa ──
    print(f"\n{'─'*80}")
    print("  COMPARAÇÃO QUALITATIVA DOS TRÊS ATRIBUTOS")
    print(f"{'─'*80}")
    print("  Critérios (evidências do treino, sem teste):")
    print()
    print("  1. Separação visual das distribuições:")
    print("     • campaign: Gamma com formas e escalas por classe")
    print("     • age: sobreposição quase total (médias ~41 vs ~42)")
    print("     • loan: diferenças proporcionais entre as duas categorias")
    print()
    print("  2. Amplitude do log-likelihood ratio:")
    print("     • campaign: varia com o número de contatos")
    print("     • age: amplitude limitada (caudas apenas)")
    print("     • loan: log-Λ confinado a uma faixa estreita (apenas 2 categorias)")
    print()
    print("  Conclusão: ranking de poder discriminativo univariado a ser verificado")

    # ── Classificadores nomeados ──
    print(f"\n{'─'*80}")
    print("  CLASSIFICADORES UNIVARIADOS")
    print(f"{'─'*80}")
    if age_map_boundaries:
        boundaries_str = " e ".join(f"{b:.2f}" for b in age_map_boundaries)
        observed_age_min, observed_age_max = age_range
        print(
            f"  h₁(age), no domínio observado [{observed_age_min:.0f}, "
            f"{observed_age_max:.0f}]: classe 1 se age > {boundaries_str}; "
            "caso contrário classe 0"
        )
    else:
        print("  h₁(age): classe 0 para todos os valores (nenhuma fronteira MAP)")
    if camp_map_boundaries:
        boundaries_str = " e ".join(f"{b:.2f}" for b in camp_map_boundaries)
        print(f"  h₂(campaign): classe 1 se campaign > {boundaries_str}; caso contrário classe 0")
    else:
        print("  h₂(campaign): classe 0 para todos os valores")
    print(f"  h₃(loan): regra categórica ({loan_rule})")

    # ── 8. Salvar artefatos ──
    logger.info("Salvando artefatos...")

    # CSVs
    _save_result_csv(age_result, UNIVARIATE_METRICS_DIR / "age_univariate_examples.csv")
    _save_result_csv(camp_result, UNIVARIATE_METRICS_DIR / "campaign_univariate_examples.csv")

    # Evidência quantitativa, somente do treino, para a limitação da hipótese
    # de independência condicional entre age e loan.
    age_by_loan = summarize_age_by_loan(X_train, y_train)
    age_by_loan_path = UNIVARIATE_METRICS_DIR / "age_by_loan_within_class.csv"
    age_by_loan.to_csv(age_by_loan_path, index=False, float_format="%.8f")
    logger.info("Diagnóstico de independência salvo: %s", age_by_loan_path)

    # Loan CSV (valores string)
    loan_df = pd.DataFrame({
        "Categoria": loan_result.values,
        "P(a_k|Y=0)": loan_result.likelihood_class_0,
        "P(a_k|Y=1)": loan_result.likelihood_class_1,
        "Λ": loan_result.likelihood_ratio,
        "P(Y=0|a_k)": loan_result.posterior_class_0,
        "P(Y=1|a_k)": loan_result.posterior_class_1,
        "regra": loan_result.predictions,
    })
    loan_csv_path = UNIVARIATE_METRICS_DIR / "loan_univariate_examples.csv"
    loan_csv_path.parent.mkdir(parents=True, exist_ok=True)
    loan_df.to_csv(loan_csv_path, index=False, float_format="%.8f")
    logger.info("Tabela salva: %s", loan_csv_path)

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
        "campaign": {
            "distribution": "Gamma",
            "class_0": {"shape": camp_params_0.shape, "scale": camp_params_0.scale,
                        "mean": camp_params_0.shape * camp_params_0.scale},
            "class_1": {"shape": camp_params_1.shape, "scale": camp_params_1.scale,
                        "mean": camp_params_1.shape * camp_params_1.scale},
            "likelihood_boundaries": camp_lr_boundaries,
            "map_boundaries": camp_map_boundaries,
        },
        "loan": {
            "distribution": "Categorical (Laplace α=1)",
            "class_0": loan_probs_0,
            "class_1": loan_probs_1,
            "categorical_rule": loan_rule,
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

    plot_campaign_analysis(
        camp_params_0, camp_params_1, priors,
        X_train.loc[mask_0, "campaign"].to_numpy(dtype=float),
        X_train.loc[mask_1, "campaign"].to_numpy(dtype=float),
        camp_lr_boundaries, camp_map_boundaries,
        UNIVARIATE_FIGURES_DIR / "campaign_conditional_and_decision.png",
    )
    logger.info("Figura campaign salva.")

    plot_loan_analysis(
        loan_probs_0, loan_probs_1, priors,
        UNIVARIATE_FIGURES_DIR / "loan_conditional_probabilities.png",
    )
    logger.info("Figura loan salva.")

    print(f"\n{'#'*80}")
    print("  Artefatos gerados com sucesso!")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    run()
