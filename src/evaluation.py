"""Métricas de avaliação e matriz de confusão — RFC-0006 e RFC-0007.

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
- Usar scikit-learn internamente para cálculos principais (scikit-learn é apenas oráculo de teste).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


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
