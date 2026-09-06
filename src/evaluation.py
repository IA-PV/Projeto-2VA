"""Métricas de avaliação e matriz de confusão — Avaliação Supervisionada.

Responsabilidades
-----------------
- Implementar cálculo manual e explícito da matriz de confusão binária.
- Extrair os quatro componentes fundamentais: VN, FP, FN e VP.
- Calcular acurácia, precisão, recall e F1 de forma transparente.
- Proteger contra divisão por zero de forma explícita.
- Servir como base de cálculo pura e auditável contra oráculos externos.

Não deve
--------
- Alterar o modelo ou realizar previsões (responsabilidade de MixedNaiveBayes).
- Usar scikit-learn para cálculos principais (é somente oráculo de consistência).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import MARITAL_CATEGORIES


def _to_binary_array(values: Any, name: str) -> np.ndarray:
    """Converte e valida arrays binários contendo valores {0, 1}."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError(f"'{name}' deve ser unidimensional, obtido ndim={arr.ndim}.")

    if len(arr) == 0:
        raise ValueError(f"'{name}' não pode ser vazio.")

    # Converte tipos numéricos/booleanos e valida valores permitidos
    unique_vals = set(np.unique(arr))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"'{name}' deve conter apenas valores no domínio binário {{0, 1}}. "
            f"Valores encontrados: {unique_vals}"
        )

    return arr.astype(int)


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Calcula a matriz de confusão 2x2 com convenção explícita.

    Convenção:
    - Linhas: Valores reais (classe 0, classe 1).
    - Colunas: Valores preditos (classe 0, classe 1).

    Estrutura retornada:
        [[VN, FP],
         [FN, VP]]

    Parameters
    ----------
    y_true : Array-like
        Rótulos reais binários {0, 1}.
    y_pred : Array-like
        Previsões binárias {0, 1}.
    labels : tuple[int, int], default=(0, 1)
        Ordem explícita das classes.

    Returns
    -------
    np.ndarray
        Matriz 2x2 de inteiros.
    """
    if labels != (0, 1):
        raise ValueError("Apenas labels=(0, 1) são suportadas neste projeto.")

    y_t = _to_binary_array(y_true, "y_true")
    y_p = _to_binary_array(y_pred, "y_pred")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Dimensões incompatíveis: y_true tem {len(y_t)} elementos, "
            f"mas y_pred tem {len(y_p)} elementos."
        )

    vn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    vp = int(np.sum((y_t == 1) & (y_p == 1)))

    return np.array([[vn, fp], [fn, vp]], dtype=int)


def extract_confusion_components(cm: np.ndarray) -> dict[str, int]:
    """Extrai os quatro componentes da matriz de confusão.

    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusão 2x2.

    Returns
    -------
    dict[str, int]
        Dicionário com chaves 'tn', 'fp', 'fn', 'tp'.
    """
    if not isinstance(cm, np.ndarray) or cm.shape != (2, 2):
        raise ValueError(f"A matriz de confusão deve ter shape (2, 2), obtido: {getattr(cm, 'shape', None)}")

    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> dict[str, float]:
    """Calcula acurácia, precisão, recall e F1 manualmente.

    Parameters
    ----------
    y_true : Array-like
        Valores reais {0, 1}.
    y_pred : Array-like
        Previsões {0, 1}.
    pos_label : int, default=1
        Classe considerada positiva.
    zero_division : float, default=0.0
        Valor de proteção quando denominador for zero.

    Returns
    -------
    dict[str, float]
        Dicionário com 'accuracy', 'precision', 'recall', 'f1'.
    """
    if pos_label != 1:
        raise ValueError("A convenção do projeto exige estritamente pos_label=1.")

    cm = compute_confusion_matrix(y_true, y_pred, labels=(0, 1))
    comps = extract_confusion_components(cm)
    vn, fp, fn, vp = comps["tn"], comps["fp"], comps["fn"], comps["tp"]
    total = vn + fp + fn + vp

    if total == 0:
        raise ValueError("O número total de amostras avaliadas não pode ser zero.")

    # Acurácia = (VP + VN) / Total
    accuracy = float((vp + vn) / total)

    # Precisão = VP / (VP + FP)
    precision_denom = vp + fp
    precision = float(vp / precision_denom) if precision_denom > 0 else float(zero_division)

    # Recall = VP / (VP + FN)
    recall_denom = vp + fn
    recall = float(vp / recall_denom) if recall_denom > 0 else float(zero_division)

    # F1 = 2 * (Precisão * Recall) / (Precisão + Recall)
    f1_denom = precision + recall
    f1 = float(2.0 * (precision * recall) / f1_denom) if f1_denom > 0 else float(zero_division)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_majority_baseline(y_true: Any) -> float:
    """Calcula a acurácia do classificador baseline majoritário (sempre classe 0).

    Parameters
    ----------
    y_true : Array-like
        Rótulos reais.

    Returns
    -------
    float
        Acurácia prevendo sempre classe 0.
    """
    y_t = _to_binary_array(y_true, "y_true")
    return float(np.sum(y_t == 0) / len(y_t))


def evaluate_predictions(y_true: Any, y_pred: Any) -> dict:
    """Calcula manualmente e confere matriz/métricas com scikit-learn.

    Aceita vetores binários, inclusive com uma única classe. Registra
    explicitamente a ausência de previsões positivas antes da proteção zero.
    """
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    )

    cm = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(y_true, y_pred, zero_division=0)
    np.testing.assert_array_equal(cm, confusion_matrix(y_true, y_pred, labels=[0, 1]))
    oracle = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    for name, value in metrics.items():
        np.testing.assert_allclose(value, oracle[name], rtol=1e-12, atol=1e-12)
    return {
        "positive_class": 1,
        "test_size": int(cm.sum()),
        "confusion_matrix": extract_confusion_components(cm),
        "metrics": metrics,
        "no_positive_predictions": bool(cm[:, 1].sum() == 0),
        "sklearn_consistency_verified": True,
    }


def build_error_groups(
    X: pd.DataFrame, y_true: Any, y_pred: Any, *,
    duration_boundary: float, long_duration_threshold: float,
) -> pd.DataFrame:
    """Resume VN/FP/FN/VP, preservando grupos vazios e empates na moda.

    Fronteira MAP univariada e quantil de duração longa devem vir somente
    do treino. São referências descritivas, nunca novos limiares de decisão.
    Series devem ter o mesmo índice/ordem que X; arrays seguem a ordem de X.
    Proporções usam o tamanho de cada grupo como denominador.
    """
    truth = _to_binary_array(y_true, "y_true")
    prediction = _to_binary_array(y_pred, "y_pred")
    if len(X) != len(truth) or len(truth) != len(prediction):
        raise ValueError("Dimensões incompatíveis entre X, y_true e y_pred.")
    for values in (y_true, y_pred):
        if isinstance(values, pd.Series) and not values.index.equals(X.index):
            raise ValueError("Índice e ordem dos rótulos devem coincidir com X.")
    for threshold in (duration_boundary, long_duration_threshold):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Referências de duração devem ser positivas e finitas.")
    rows = []
    for name, real, predicted in (("VN", 0, 0), ("FP", 0, 1), ("FN", 1, 0), ("VP", 1, 1)):
        group = X.loc[(truth == real) & (prediction == predicted)]
        row = {"group": name, "n": len(group)}
        for feature in ("age", "duration"):
            for statistic in ("mean", "median"):
                row[f"{feature}_{statistic}"] = getattr(group[feature], statistic)()
        row["marital_mode"] = " | ".join(sorted(group["marital"].mode().astype(str)))
        for category in MARITAL_CATEGORIES:
            row[f"marital_{category}_n"] = int((group["marital"] == category).sum())
        for label, mask in (
            ("duration_below_boundary", group["duration"] < duration_boundary),
            ("duration_above_train_q95", group["duration"] > long_duration_threshold),
        ):
            row[f"{label}_n"] = int(mask.sum())
            row[f"{label}_fraction"] = float(mask.mean()) if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_age_by_marital(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Evidência descritiva de associação age/marital dentro de cada classe."""
    if not X_train.index.equals(y_train.index):
        raise ValueError("Índice e ordem de y_train devem coincidir com X_train.")
    frame = X_train.assign(actual_class=_to_binary_array(y_train, "y_train"))
    return (frame.groupby(["actual_class", "marital"], observed=True)["age"]
            .agg(n="size", age_mean="mean", age_median="median").reset_index())
