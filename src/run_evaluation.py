"""Avaliação oficial congelada. Execute python -m src.run_evaluation.

A checklist é persistida antes de predict; cada tentativa fica no histórico,
inclusive falhas. Reavaliações exigem motivo e arquivos alterados explícitos.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH, EXPECTED_SHA256, LAPLACE_ALPHA, MODEL_PARAMETERS_PATH,
    RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
)
from src.data import load_bank_data, make_stratified_split, prepare_model_frame, validate_raw_data
from src.evaluation import build_error_groups, evaluate_predictions
from src.mixed_naive_bayes import MixedNaiveBayes
from src.plotting import plot_confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
METRICS = Path("reports/metrics")
HISTORY = METRICS / "evaluation_history.json"
CHECKLIST = METRICS / "freeze_checklist.json"
FINAL = METRICS / "final_metrics.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict | list) -> None:
    """Escreve JSON estrito; floats mantêm precisão, inclusive zeros 0.0000."""
    import re

    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    # Somente tokens numéricos de valores (não strings); ao menos 4 decimais.
    text = re.sub(r'(?<=: )(\d+\.\d{1,3})(?=,?\n)',
                  lambda m: m.group(1) + "0" * (4 - len(m.group(1).split(".")[1])), text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _run_prerequisites() -> dict:
    """Verifica requisitos e conclui análises exclusivamente no treino.

    Exclui apenas o teste legado que faz inferência em X_test. Ele já existia
    antes desta etapa; não é necessário consultar o holdout para abrir o portão.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    commands = {
        "tests": [sys.executable, "-m", "pytest", "-q", "-k",
                  "not test_frozen_split_invariants_and_scipy_reference"],
        "univariate": [sys.executable, "-m", "src.run_univariate"],
    }
    evidence = {}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", check=True)
        evidence[name] = {"command": "python " + " ".join(command[1:]),
                          "completed_at_utc": _utc_now(), "output": result.stdout.strip()}
    return evidence


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def run(*, reason: str | None = None, changed_files: list[str] | None = None) -> Path:
    """Ajusta no treino, prediz teste uma vez e exporta uma avaliação auditada."""
    history_path = ROOT / HISTORY
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if history or (ROOT / FINAL).exists():
        if not reason or not reason.strip() or not changed_files:
            raise ValueError("Reavaliação exige --reason e --changed-file; consulte evaluation_history.json.")
    if (TEST_SIZE, RANDOM_STATE, LAPLACE_ALPHA) != (0.20, 42, 1.0):
        raise ValueError("Configuração diverge da especificação congelada.")

    evidence = _run_prerequisites()
    raw = load_bank_data(ROOT / DATA_PATH)
    validate_raw_data(raw, path=ROOT / DATA_PATH)
    X, y = prepare_model_frame(raw)
    split = make_stratified_split(X, y)
    if (len(split.y_train), len(split.y_test)) != (3616, 905):
        raise ValueError("O split congelado exige 3616 observações de treino e 905 de teste.")
    if split.X_train.index.intersection(split.X_test.index).size:
        raise ValueError("Treino e teste possuem índices sobrepostos.")
    if split.y_test.value_counts().to_dict() != {0: 801, 1: 104}:
        raise ValueError("Composição do teste difere das 801 negativas e 104 positivas.")
    model = MixedNaiveBayes(alpha=LAPLACE_ALPHA).fit(split.X_train, split.y_train)
    parameters = model.get_fitted_parameters()
    parameters["dataset"] = {"path": DATA_PATH.as_posix(), "sha256": EXPECTED_SHA256}
    parameters["split"] = {"random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                           "stratify": TARGET_COLUMN, "n_train": len(split.y_train)}
    _write_json(ROOT / MODEL_PARAMETERS_PATH, parameters)

    univariate_path = ROOT / METRICS / "distribution_parameters.json"
    univariate = json.loads(univariate_path.read_text(encoding="utf-8"))
    for c in (0, 1):
        np.testing.assert_allclose(univariate["priors"][str(c)], parameters["priors"][str(c)])
        for key in ("shape", "scale"):
            np.testing.assert_allclose(univariate["campaign"][f"class_{c}"][key],
                                       parameters["campaign"][str(c)][key])
    boundaries = univariate["campaign"]["map_boundaries"]
    if not boundaries:
        boundaries = univariate["campaign"]["likelihood_boundaries"]
    if len(boundaries) < 1:
        raise ValueError("A análise congelada deve fornecer pelo menos uma fronteira para campaign.")
    references = {
        "campaign_map_boundary_train": float(boundaries[-1]),
        "campaign_q95_train": float(split.X_train["campaign"].quantile(0.95)),
        "interpretation": "Referências descritivas do treino; não alteram a regra MAP conjunta.",
    }
    checklist = {
        "recorded_at_utc": _utc_now(),
        "data_contract": "Contrato verificado por validate_raw_data e testes de dados.",
        "probabilistic_modeling": "Normal / Gamma / Categórica; priors empíricas; parâmetros do treino.",
        "univariate_analysis": "Três análises concluídas por src.run_univariate antes da avaliação.",
        "model_fit": "MixedNaiveBayes ajustado exclusivamente em 3616 observações de treino.",
        "automated_tests": evidence["tests"],
        "univariate_completed_at_utc": evidence["univariate"]["completed_at_utc"],
        "configuration": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "alpha": LAPLACE_ALPHA},
        "source_sha256": _source_hashes(),
        "dataset_sha256": EXPECTED_SHA256,
        "model_parameters_sha256": hashlib.sha256((ROOT / MODEL_PARAMETERS_PATH).read_bytes()).hexdigest(),
        "split_indices_sha256": {
            name: hashlib.sha256(json.dumps(frame.index.tolist()).encode()).hexdigest()
            for name, frame in (("train", split.X_train), ("test", split.X_test))
        },
        "prior_test_access": (
            "O teste legado test_frozen_split_invariants_and_scipy_reference já faz inferência "
            "em X_test, sem métricas de desempenho. Foi executado na inspeção inicial; "
            "é excluído da verificação do portão. Nenhuma métrica final anterior foi encontrada. "
            "Resultados anteriores ao portão são exploratórios e não compõem esta entrega. "
            "Não se afirma que este seja o primeiro acesso intocado ao teste."
        ),
    }
    _write_json(ROOT / CHECKLIST, checklist)
    entry = {"run_id": len(history) + 1, "started_at_utc": _utc_now(), "status": "started",
             "reason": reason or "Avaliação oficial após verificação das dependências.",
             "changed_files": changed_files or [], "checklist": checklist,
             "model_parameters": parameters}
    history.append(entry)
    _write_json(history_path, history)
    try:
        prediction = model.predict(split.X_test)  # Única inferência oficial desta execução.
        result = evaluate_predictions(split.y_test, prediction)
        if sum(result["confusion_matrix"].values()) != 905:
            raise ValueError("A soma da matriz deve ser 905.")
        baseline = evaluate_predictions(split.y_test, np.zeros(len(split.y_test), dtype=int))
        result.update({"majority_baseline_accuracy": baseline["metrics"]["accuracy"],
                       "majority_baseline": baseline, "run_id": entry["run_id"],
                       "evaluated_at_utc": entry["started_at_utc"],
                       "error_analysis_references": references})
        components = result["confusion_matrix"]
        cm = np.array([[components["tn"], components["fp"]],
                       [components["fn"], components["tp"]]])
        groups = build_error_groups(split.X_test, split.y_test, prediction,
                                    campaign_boundary=references["campaign_map_boundary_train"],
                                    long_campaign_threshold=references["campaign_q95_train"])
        pd.DataFrame(cm, index=[0, 1], columns=["predicted_0", "predicted_1"]).to_csv(
            ROOT / METRICS / "confusion_matrix.csv", index_label="actual_class")
        groups.to_csv(ROOT / METRICS / "error_groups.csv", index=False, float_format="%.10f")
        plot_confusion_matrix(cm, ROOT / "reports/figures/confusion_matrix.png")
        _write_json(ROOT / FINAL, result)
        entry.update(status="completed", completed_at_utc=_utc_now(), result=result)
    except Exception as exc:
        entry.update(status="failed", completed_at_utc=_utc_now(), error=str(exc))
        raise
    finally:
        _write_json(history_path, history)
    return ROOT / FINAL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", help="Motivo documentado para reavaliação legítima.")
    parser.add_argument("--changed-file", action="append", dest="changed_files",
                        help="Arquivo alterado após avaliação; pode ser repetido.")
    args = parser.parse_args(argv)
    path = run(reason=args.reason, changed_files=args.changed_files)
    print(f"Avaliação oficial registrada em {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
