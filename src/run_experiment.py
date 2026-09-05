"""Ajuste e auditoria do classificador misto — RFC-0005.

Execução na raiz: python -m src.run_experiment
O teste permanece reservado à avaliação final; este runner ajusta o modelo
no treino e salva os parâmetros e a identificação da fonte/split.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import (
    DATA_PATH,
    EXPECTED_SHA256,
    MODEL_PARAMETERS_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import (
    load_bank_data,
    make_stratified_split,
    prepare_model_frame,
    validate_raw_data,
)
from src.mixed_naive_bayes import MixedNaiveBayes


def run() -> Path:
    """Reproduz o ajuste congelado e exporta model_parameters.json."""
    raw = load_bank_data(DATA_PATH)
    validate_raw_data(raw, path=DATA_PATH)
    X, y = prepare_model_frame(raw)
    split = make_stratified_split(X, y)
    model = MixedNaiveBayes().fit(split.X_train, split.y_train)
    parameters = model.get_fitted_parameters()
    parameters["dataset"] = {
        "path": DATA_PATH.as_posix(),
        "sha256": EXPECTED_SHA256,  # verificado por validate_raw_data
    }
    parameters["split"] = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stratify": TARGET_COLUMN,
        "n_train": len(split.y_train),
    }
    payload = json.dumps(parameters, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    MODEL_PARAMETERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PARAMETERS_PATH.write_text(payload, encoding="utf-8")
    return MODEL_PARAMETERS_PATH.resolve()


if __name__ == "__main__":
    print(f"Parâmetros do modelo salvos em: {run()}")
