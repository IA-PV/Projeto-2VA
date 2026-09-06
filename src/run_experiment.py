"""Orquestrador do estudo científico e auditoria.

Execução na raiz:
    Validação de dados e ambiente (somente leitura):
        python -m src.run_experiment --validate-only

    Execução do estudo completo (reprodução oficial de ponta a ponta):
        python -m src.run_experiment
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import sys

from src.config import (
    DATA_PATH,
    DISTRIBUTIONS,
    EXPECTED_SHA256,
    ExperimentConfig,
    MODEL_PARAMETERS_PATH,
    RANDOM_STATE,
    RUN_MANIFEST_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.data import (
    _compute_sha256,
    load_bank_data,
    make_stratified_split,
    prepare_model_frame,
    save_split_report,
    validate_raw_data,
)
from src.mixed_naive_bayes import MixedNaiveBayes

logger = logging.getLogger("src.run_experiment")


def validate_pipeline(config: ExperimentConfig | None = None) -> bool:
    """Valida integridade de dados, divisão estratificada e ajuste do modelo em memória.

    Não escreve nem altera artefatos de saída (pureza para testes e auditoria).
    """
    cfg = config or ExperimentConfig()
    data_path = DATA_PATH if DATA_PATH != Path("data/raw/bank.csv") else cfg.data_path
    raw = load_bank_data(data_path)
    validate_raw_data(raw, path=data_path)
    X, y = prepare_model_frame(raw)
    split = make_stratified_split(X, y)

    # Invariantes do split
    if len(split.X_train) != 3616 or len(split.X_test) != 905:
        raise ValueError(
            f"Dimensões do split inválidas: treino={len(split.X_train)}, teste={len(split.X_test)}"
        )

    overlap = set(split.X_train.index) & set(split.X_test.index)
    if overlap:
        raise ValueError(
            f"Sobreposição detectada entre treino e teste: {len(overlap)} índices compartilhados"
        )

    union_indices = set(split.X_train.index) | set(split.X_test.index)
    if union_indices != set(raw.index):
        raise ValueError("A união dos índices de treino e teste não cobre o dataset completo.")

    # Ajuste transacional do modelo em memória
    model = MixedNaiveBayes(alpha=cfg.laplace_alpha).fit(split.X_train, split.y_train)
    params = model.get_fitted_parameters()
    if not params or "age" not in params or "duration" not in params or "marital" not in params:
        raise ValueError("Parâmetros do modelo vazios ou incompletos após ajuste.")

    return True


def generate_run_manifest(
    config: ExperimentConfig | None = None,
    dataset_sha256: str | None = None,
    path: Path | None = None,
) -> Path:
    """Gera o manifesto formal de execução.

    Registra metadados de execução, ambiente operacional, versão da linguagem,
    identidade da base, hiperparâmetros e famílias de distribuição.
    """
    cfg = config or ExperimentConfig()
    manifest_path = Path(path) if path is not None else cfg.run_manifest_path

    if dataset_sha256 is None:
        raw_path = DATA_PATH if DATA_PATH != Path("data/raw/bank.csv") else cfg.data_path
        dataset_sha256 = _compute_sha256(Path(raw_path))

    manifest_data = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_sha256": dataset_sha256,
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "features": list(cfg.feature_columns),
        "target_mapping": dict(TARGET_MAPPING),
        "laplace_alpha": cfg.laplace_alpha,
        "distributions": dict(DISTRIBUTIONS),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n"
    manifest_path.write_text(payload, encoding="utf-8")
    return manifest_path.resolve()


def run(
    config: ExperimentConfig | None = None,
    *,
    full_pipeline: bool = False,
    reason: str | None = None,
    changed_files: list[str] | None = None,
    force_reproduce: bool = False,
) -> Path | dict[str, Path]:
    """Ajusta o modelo ou executa o pipeline científico completo.

    Se full_pipeline=False (padrão para testes unitários de ajuste), ajusta no
    treino e exporta model_parameters.json, retornando seu caminho.
    Se full_pipeline=True, regenera parâmetros, tabelas, figuras e manifesto. A
    avaliação auditada existente é preservada por padrão; ``force_reproduce=True``
    ou um motivo documentado permitem reproduzi-la explicitamente.
    """
    cfg = config or ExperimentConfig()
    data_path = DATA_PATH if DATA_PATH != Path("data/raw/bank.csv") else cfg.data_path
    output_params_path = (
        MODEL_PARAMETERS_PATH
        if MODEL_PARAMETERS_PATH != Path("reports/metrics/model_parameters.json")
        else cfg.model_parameters_path
    )

    if not full_pipeline:
        # Modo de ajuste: isolado ao ajuste de treino e exportação de model_parameters.json
        raw = load_bank_data(data_path)
        validate_raw_data(raw, path=data_path)
        X, y = prepare_model_frame(raw)
        split = make_stratified_split(X, y)
        model = MixedNaiveBayes(alpha=cfg.laplace_alpha).fit(split.X_train, split.y_train)
        parameters = model.get_fitted_parameters()
        parameters["dataset"] = {
            "path": Path(data_path).as_posix(),
            "sha256": EXPECTED_SHA256,
        }
        parameters["split"] = {
            "random_state": cfg.random_state,
            "test_size": cfg.test_size,
            "stratify": cfg.target_column,
            "n_train": len(split.y_train),
        }
        payload = json.dumps(parameters, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        output_params_path.parent.mkdir(parents=True, exist_ok=True)
        output_params_path.write_text(payload, encoding="utf-8")
        return output_params_path.resolve()

    # ─────────────────────────────────────────────────────────────
    #  Execução do estudo completo
    # ─────────────────────────────────────────────────────────────
    logger.info("Iniciando execução do estudo científico completo...")

    # 1. Carregamento e validação de dados
    raw = load_bank_data(data_path)
    validate_raw_data(raw, path=data_path)
    dataset_sha256 = _compute_sha256(Path(data_path))
    logger.info("Dados carregados e validados: shape=%s, sha256=%s", raw.shape, dataset_sha256)

    # 2. Divisão estratificada e relatório de split
    X, y = prepare_model_frame(raw)
    split = make_stratified_split(X, y)
    split_report_path = save_split_report(split, sha256=dataset_sha256, path=cfg.split_report_path)
    logger.info(
        "Divisão estratificada realizada: treino=%d (0=%d, 1=%d), teste=%d (0=%d, 1=%d)",
        len(split.y_train),
        (split.y_train == 0).sum(),
        (split.y_train == 1).sum(),
        len(split.y_test),
        (split.y_test == 0).sum(),
        (split.y_test == 1).sum(),
    )

    # 3. Análise univariada bayesiana
    from src import run_univariate

    logger.info("Executando análises univariadas...")
    run_univariate.run()
    logger.info("Conclusão dos ajustes univariados e figuras.")

    # 4. Ajuste do classificador misto e auditoria de parâmetros
    model = MixedNaiveBayes(alpha=cfg.laplace_alpha).fit(split.X_train, split.y_train)
    parameters = model.get_fitted_parameters()
    parameters["dataset"] = {
        "path": Path(data_path).as_posix(),
        "sha256": dataset_sha256,
    }
    parameters["split"] = {
        "random_state": cfg.random_state,
        "test_size": cfg.test_size,
        "stratify": cfg.target_column,
        "n_train": len(split.y_train),
    }
    payload = json.dumps(parameters, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_params_path.parent.mkdir(parents=True, exist_ok=True)
    output_params_path.write_text(payload, encoding="utf-8")
    logger.info("Conclusão dos ajustes do classificador misto: %s", output_params_path)

    # 5. Avaliação oficial no teste
    from src import run_evaluation

    final_metrics_path = cfg.final_metrics_path
    if final_metrics_path.exists() and not force_reproduce and not reason:
        logger.info(
            "Conclusão da avaliação: final_metrics.json existente preservado. "
            "Para reavaliação auditada, use --reason e --changed-file ou --force-reproduce."
        )
    else:
        logger.info("Executando avaliação oficial congelada no holdout...")
        eval_reason = (
            reason
            if reason
            else (
                "Reprodução oficial em clone limpo"
                if force_reproduce
                else "Execução oficial do pipeline completo"
            )
        )
        eval_changed = changed_files if changed_files is not None else ["reproducibility-pipeline"]
        final_metrics_path = run_evaluation.run(reason=eval_reason, changed_files=eval_changed)
        logger.info("Conclusão da avaliação: %s", final_metrics_path)

    # 6. Geração do manifesto de execução
    manifest_path = generate_run_manifest(config=cfg, dataset_sha256=dataset_sha256)
    logger.info("Manifesto de execução gerado: %s", manifest_path)

    artifacts: dict[str, Path] = {
        "split_report": split_report_path,
        "distribution_parameters": cfg.distribution_parameters_path,
        "model_parameters": output_params_path.resolve(),
        "final_metrics": final_metrics_path,
        "run_manifest": manifest_path,
    }

    logger.info("Caminhos dos artefatos gerados:")
    for name, art_path in artifacts.items():
        logger.info("  • %s: %s", name, art_path)

    return artifacts


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada CLI para execução completa ou validação rápida."""
    parser = argparse.ArgumentParser(
        description="Orquestrador de reprodução e auditoria do estudo científico."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Executa validações formais de integridade, split e modelo sem gravar arquivos.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/bank.csv"),
        help="Caminho para o CSV de dados brutos (padrão: data/raw/bank.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Diretório base de saída para artefatos (padrão: reports).",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default=None,
        help="Motivo documentado para reavaliação oficial se métricas já existirem.",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        dest="changed_files",
        default=None,
        help="Arquivo alterado antes da reavaliação (pode ser repetido).",
    )
    parser.add_argument(
        "--force-reproduce",
        action="store_true",
        default=False,
        help="Força reexecução da avaliação oficial registrando motivo de reprodutibilidade.",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    config = ExperimentConfig(data_path=args.data_path, output_dir=args.output_dir)

    if args.validate_only:
        validate_pipeline(config)
        print(
            "[OK] Validação de dados, split e modelo concluída com sucesso (sem alterações em relatórios)."
        )
        return 0

    artifacts = run(
        config=config,
        full_pipeline=True,
        reason=args.reason,
        changed_files=args.changed_files,
        force_reproduce=args.force_reproduce,
    )

    print("\n" + "=" * 80)
    print("  ESTUDO COMPLETO EXECUTADO COM SUCESSO")
    print("=" * 80)
    if isinstance(artifacts, dict):
        for name, art_path in artifacts.items():
            print(f"  • {name:<25}: {art_path}")
    else:
        print(f"  • Artefato gerado: {artifacts}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
