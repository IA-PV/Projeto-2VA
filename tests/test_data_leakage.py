"""Teste de não-vazamento de dados (Data Leakage).

Prova formalmente que:
1. O método `fit` recebe exclusivamente o conjunto de treino (X_train, y_train).
2. Os parâmetros do modelo ajustado coincidem exatamente com o cálculo direto no treino.
3. Os parâmetros do modelo ajustado diferem estritamente dos parâmetros que seriam
   obtidos se a base completa (treino + teste) fosse utilizada.
4. O artefato serializado de parâmetros registra apenas contagens e dados do treino.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import EXPECTED_SHA256, MARITAL_CATEGORIES, MODEL_PARAMETERS_PATH
from src.data import DataSplit
from src.distributions import (
    fit_categorical,
    fit_class_priors,
    fit_gamma_mle,
    fit_gaussian_mle,
)
from src.mixed_naive_bayes import MixedNaiveBayes
from src import run_experiment


@pytest.mark.regression
class TestDataLeakagePrevention:
    """Garantias formais de isolamento do conjunto de teste e ausência de contaminação."""

    def test_fit_parameters_match_training_slice_exactly(self, data_split: DataSplit) -> None:
        """Parâmetros do modelo são estritamente iguais ao cálculo direto em X_train."""
        X_train = data_split.X_train
        y_train = data_split.y_train

        model = MixedNaiveBayes().fit(X_train, y_train)

        # 1. Priors e contagens
        expected_priors = fit_class_priors(y_train)
        for c in (0, 1):
            assert np.exp(model.class_log_prior_[c]) == pytest.approx(expected_priors[c], abs=1e-12)
            assert model.class_count_[c] == int((y_train == c).sum())

        assert model.class_count_ == {0: 3199, 1: 417}

        # 2. Gaussian MLE para 'age'
        for c in (0, 1):
            age_direct = fit_gaussian_mle(X_train.loc[y_train == c, "age"].to_numpy())
            assert model.age_params_[c].mean == pytest.approx(age_direct.mean, rel=1e-12)
            assert model.age_params_[c].variance == pytest.approx(age_direct.variance, rel=1e-12)

        # 3. Gamma MLE para 'duration'
        for c in (0, 1):
            dur_direct = fit_gamma_mle(X_train.loc[y_train == c, "duration"].to_numpy())
            assert model.duration_params_[c].shape == pytest.approx(dur_direct.shape, rel=1e-12)
            assert model.duration_params_[c].scale == pytest.approx(dur_direct.scale, rel=1e-12)

        # 4. Categórico para 'marital'
        for c in (0, 1):
            marital_direct = fit_categorical(
                X_train.loc[y_train == c, "marital"],
                categories=MARITAL_CATEGORIES,
                alpha=1.0,
            )
            for cat in MARITAL_CATEGORIES:
                prob_model = np.exp(model.marital_log_prob_[c][cat])
                assert prob_model == pytest.approx(marital_direct[cat], rel=1e-12)

    def test_fit_parameters_differ_from_full_dataset(
        self, data_split: DataSplit, model_xy: tuple
    ) -> None:
        """Prova que os parâmetros do treino diferem dos parâmetros da base completa."""
        X_full, y_full = model_xy
        X_train = data_split.X_train
        y_train = data_split.y_train

        model_train = MixedNaiveBayes().fit(X_train, y_train)
        model_full = MixedNaiveBayes().fit(X_full, y_full)

        # 1. Total de amostras difere: 3616 no treino vs 4521 na base completa
        assert sum(model_train.class_count_.values()) == 3616
        assert sum(model_full.class_count_.values()) == 4521
        assert model_train.class_count_ != model_full.class_count_

        # 2. Média de idade difere entre treino e base completa
        for c in (0, 1):
            mean_train = model_train.age_params_[c].mean
            mean_full = model_full.age_params_[c].mean
            assert mean_train != mean_full, (
                f"Média de idade da classe {c} no treino ({mean_train}) "
                f"coincidiu com a base completa ({mean_full}), indicando possível vazamento."
            )

        # 3. Variância de idade difere
        for c in (0, 1):
            var_train = model_train.age_params_[c].variance
            var_full = model_full.age_params_[c].variance
            assert var_train != var_full

        # 4. Parâmetros Gamma de duration diferem
        for c in (0, 1):
            scale_train = model_train.duration_params_[c].scale
            scale_full = model_full.duration_params_[c].scale
            assert scale_train != scale_full

    def test_serialized_parameters_contain_only_train_information(self) -> None:
        """Verifica se o artefato serializado oficial registra estritamente n_train=3616."""
        if not MODEL_PARAMETERS_PATH.is_file():
            run_experiment.run()

        data = json.loads(MODEL_PARAMETERS_PATH.read_text(encoding="utf-8"))

        assert data["split"]["n_train"] == 3616
        assert data["split"]["test_size"] == 0.20
        assert data["class_count"] == {"0": 3199, "1": 417}
        assert data["dataset"]["sha256"] == EXPECTED_SHA256

        # Garante que não existem chaves de teste ou avaliação vazadas nos parâmetros
        forbidden_keys = {"n_test", "y_test", "test_metrics", "test_counts"}
        assert forbidden_keys.isdisjoint(set(data.keys()))
        assert forbidden_keys.isdisjoint(set(data.get("split", {}).keys()))
