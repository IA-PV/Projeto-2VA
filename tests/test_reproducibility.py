"""Testes de conformidade e reprodutibilidade científica.

Valida:
- Imutabilidade e contratos da dataclass ExperimentConfig.
- Comportamento de somente-leitura do modo --validate-only.
- Estrutura estrita, completude e tipos do manifesto de execução (run_manifest.json).
- Fixação exata de versões diretas em requirements.txt (sem marcadores de rascunho).
- Governança do .gitignore garantindo o versionamento de relatórios e figuras.
- Ausência de caminhos absolutos locais no código e notebooks.
- Determinismo estrito na exportação de parâmetros.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import json
from pathlib import Path
import re

import pytest

from src.config import (
    DISTRIBUTIONS,
    EXPECTED_SHA256,
    ExperimentConfig,
    FEATURE_COLUMNS,
    LAPLACE_ALPHA,
    RANDOM_STATE,
    RUN_MANIFEST_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src import run_experiment

ROOT = Path(__file__).resolve().parents[1]


class TestExperimentConfig:
    """Valida o contrato da configuração científica imutável."""

    def test_is_frozen_dataclass(self) -> None:
        assert is_dataclass(ExperimentConfig)
        cfg = ExperimentConfig()
        with pytest.raises((FrozenInstanceError, TypeError)):
            cfg.random_state = 99  # type: ignore[misc]

    def test_default_values_match_congelados(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.data_path == Path("data/raw/bank.csv")
        assert cfg.output_dir == Path("reports")
        assert cfg.feature_columns == ("age", "campaign", "loan")
        assert cfg.target_column == TARGET_COLUMN
        assert cfg.test_size == pytest.approx(TEST_SIZE)
        assert cfg.random_state == RANDOM_STATE
        assert cfg.laplace_alpha == pytest.approx(LAPLACE_ALPHA)
        assert cfg.class_order == (0, 1)

    def test_derived_paths_respect_output_dir(self, tmp_path: Path) -> None:
        custom_cfg = ExperimentConfig(output_dir=tmp_path / "custom_reports")
        assert custom_cfg.metrics_dir == tmp_path / "custom_reports" / "metrics"
        assert custom_cfg.figures_dir == tmp_path / "custom_reports" / "figures"
        assert custom_cfg.split_report_path == custom_cfg.metrics_dir / "data_split.json"
        assert custom_cfg.model_parameters_path == custom_cfg.metrics_dir / "model_parameters.json"
        assert custom_cfg.run_manifest_path == custom_cfg.metrics_dir / "run_manifest.json"


class TestValidateOnlyReadOnly:
    """Valida que o modo --validate-only não altera nem cria arquivos em disco."""

    def test_validate_only_leaves_filesystem_untouched(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "reports_probe"
        cfg = ExperimentConfig(output_dir=output_dir)

        assert not output_dir.exists()
        success = run_experiment.validate_pipeline(cfg)
        assert success is True
        assert not output_dir.exists(), "O modo --validate-only não deve criar diretórios de saída."


class TestRunManifest:
    """Valida a estrutura formal do manifesto de execução."""

    def test_manifest_structure_and_types(self, tmp_path: Path) -> None:
        manifest_file = tmp_path / "run_manifest.json"
        saved = run_experiment.generate_run_manifest(path=manifest_file)
        assert saved == manifest_file.resolve()
        assert manifest_file.is_file()

        data = json.loads(manifest_file.read_text(encoding="utf-8"))

        # 1. Chaves mandatórias da especificação de reprodutibilidade
        expected_keys = {
            "executed_at_utc",
            "python_version",
            "platform",
            "dataset_sha256",
            "random_state",
            "test_size",
            "features",
            "target_mapping",
            "laplace_alpha",
            "distributions",
        }
        assert set(data.keys()) == expected_keys

        # 2. Nenhum campo pode permanecer nulo
        for k, v in data.items():
            assert v is not None, f"O campo {k} no manifesto não foi preenchido."

        # 3. Tipos e valores canônicos
        assert isinstance(data["executed_at_utc"], str)
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", data["executed_at_utc"])
        assert data["dataset_sha256"] == EXPECTED_SHA256
        assert data["random_state"] == RANDOM_STATE
        assert data["test_size"] == TEST_SIZE
        assert data["features"] == list(FEATURE_COLUMNS)
        assert data["target_mapping"] == TARGET_MAPPING
        assert data["laplace_alpha"] == LAPLACE_ALPHA
        assert data["distributions"] == DISTRIBUTIONS


class TestDependenciesAndGovernance:
    """Valida a governança de requisitos, gitignore e integridade do repositório."""

    def test_requirements_contains_exact_pinned_versions(self) -> None:
        req_file = ROOT / "requirements.txt"
        assert req_file.is_file()
        content = req_file.read_text(encoding="utf-8")

        assert "VERSAO_VALIDADA" not in content, "Marcações residuais de rascunho em requirements.txt"
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

        expected_packages = {"pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "pytest"}
        found_packages = set()

        for line in lines:
            assert "==" in line, f"Dependência '{line}' deve conter versão exata pinada com '=='"
            assert not any(op in line for op in (">=", "<=", "~=", ">", "<")), (
                f"Operador flexível encontrado na dependência '{line}'"
            )
            pkg_name = line.split("==")[0].strip()
            found_packages.add(pkg_name)

        assert expected_packages.issubset(found_packages), (
            f"Pacotes essenciais ausentes em requirements.txt: {expected_packages - found_packages}"
        )

    def test_gitignore_preserves_reports_and_figures(self) -> None:
        gitignore_file = ROOT / ".gitignore"
        assert gitignore_file.is_file()
        content = gitignore_file.read_text(encoding="utf-8")

        # Não pode ignorar os relatórios ou figuras que compõem a entrega
        assert "reports/metrics/*.json" not in content
        assert "reports/figures/*.png" not in content
        assert ".venv/" in content
        assert "__pycache__/" in content

    def test_no_absolute_user_paths_in_codebase(self) -> None:
        """Verifica que nenhum arquivo de código ou notebook contém caminhos absolutos do desenvolvedor."""
        forbidden_patterns = [
            re.compile(r"[A-Za-z]:[\\/](?:Users|home|root)[\\/]", re.IGNORECASE),
        ]

        targets = list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
        for nb in (ROOT / "notebooks").glob("*.ipynb"):
            targets.append(nb)

        violations = []
        for file_path in targets:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            for pattern in forbidden_patterns:
                matches = pattern.findall(text)
                if matches:
                    violations.append(f"{file_path.relative_to(ROOT)}: {matches}")

        assert not violations, f"Caminhos absolutos locais encontrados no código:\n" + "\n".join(violations)


class TestDeterministicModelExport:
    """Valida que múltiplas execuções reproduzem saídas idênticas bit-a-bit."""

    def test_deterministic_model_parameters(self, tmp_path: Path) -> None:
        cfg1 = ExperimentConfig(output_dir=tmp_path / "run1")
        cfg2 = ExperimentConfig(output_dir=tmp_path / "run2")

        run_experiment.run(cfg1, full_pipeline=False)
        run_experiment.run(cfg2, full_pipeline=False)

        out1 = cfg1.model_parameters_path
        out2 = cfg2.model_parameters_path

        assert out1.is_file()
        assert out2.is_file()
        assert out1.read_bytes() == out2.read_bytes(), "Ajuste do modelo gerou arquivos divergentes."
