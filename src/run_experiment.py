"""Ajuste e auditoria do classificador misto — RFC-0005.

Execução na raiz: python -m src.run_experiment
O teste permanece reservado à avaliação final; este runner ajusta o modelo
no treino e salva os parâmetros e a identificação da fonte/split.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def validate_pipeline() -> bool:
    """Valida integridade de dados, divisão estratificada e ajuste do modelo em memória.

    Não escreve nem altera artefatos de saída (pureza para testes e auditoria).
    """
    raw = load_bank_data(DATA_PATH)
    validate_raw_data(raw, path=DATA_PATH)
    X, y = prepare_model_frame(raw)
    split = make_stratified_split(X, y)

    # Invariantes do split
    if len(split.X_train) != 3616 or len(split.X_test) != 905:
        raise ValueError(f"Dimensões do split inválidas: treino={len(split.X_train)}, teste={len(split.X_test)}")

    overlap = set(split.X_train.index) & set(split.X_test.index)
    if overlap:
        raise ValueError(f"Sobreposição detectada entre treino e teste: {len(overlap)} índices compartilhados")

    union_indices = set(split.X_train.index) | set(split.X_test.index)
    if union_indices != set(raw.index):
        raise ValueError("A união dos índices de treino e teste não cobre o dataset completo.")

    # Ajuste transacional do modelo em memória
    model = MixedNaiveBayes().fit(split.X_train, split.y_train)
    params = model.get_fitted_parameters()
    if not params or "age" not in params or "duration" not in params or "marital" not in params:
        raise ValueError("Parâmetros do modelo vazios ou incompletos após ajuste.")

    return True


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


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada CLI para execução ou validação rápida."""
    parser = argparse.ArgumentParser(
        description="Ajuste e auditoria do classificador misto — RFC-0005 e RFC-0006."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Executa validações formais de integridade, split e modelo sem gravar arquivos.",
    )
    args = parser.parse_args(argv)

    if args.validate_only:
        validate_pipeline()
        print("[OK] Validação de dados, split e modelo concluída com sucesso (sem alterações em relatórios).")
        return 0

    saved_path = run()
    print(f"Parâmetros do modelo salvos em: {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

