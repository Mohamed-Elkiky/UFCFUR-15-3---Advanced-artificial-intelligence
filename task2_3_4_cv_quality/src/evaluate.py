"""Evaluation utilities for the CV quality task.

This module runs a trained classifier over a :class:`~torch.utils.data.DataLoader`
and turns the raw predictions into the headline metrics used throughout
the rest of the task: accuracy, per-class precision / recall / F1, and
the macro- and weighted-averages. Results are returned as dataclasses
and can be pretty-printed or converted to a :class:`pandas.DataFrame`.

Typical usage::

    from task2_3_4_cv_quality.src.train import create_dataloaders
    from task2_3_4_cv_quality.src.evaluate import (
        evaluate_model,
        compute_metrics,
        print_results_table,
    )

    bundle = create_dataloaders(verbose=False)
    eval_result = evaluate_model(model, bundle.test_loader, bundle.classes, device)
    metrics = compute_metrics(eval_result.y_true, eval_result.y_pred, bundle.classes)
    print_results_table(metrics)

Public API
----------
* :func:`evaluate_model`      -- run a model over a dataloader inside a
                                 ``torch.no_grad`` loop.
* :func:`compute_metrics`     -- compute accuracy, per-class P/R/F1 and
                                 macro / weighted averages.
* :func:`confusion_matrix_df` -- labelled confusion matrix as a
                                 :class:`pandas.DataFrame`.
* :func:`print_results_table` -- pretty-printed results table.
* :class:`EvaluationResult`   -- container returned by :func:`evaluate_model`.
* :class:`MetricsResult`      -- container returned by :func:`compute_metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Raw outputs collected from a single evaluation pass.

    Attributes
    ----------
    y_true : np.ndarray
        Ground-truth class indices, shape ``(n_samples,)``.
    y_pred : np.ndarray
        Predicted class indices (``argmax`` of the logits),
        shape ``(n_samples,)``.
    y_probs : np.ndarray
        Softmax probabilities, shape ``(n_samples, n_classes)``. Retained
        so downstream code can compute top-k accuracy, ROC curves or
        feed the XAI / transparency views consumed by the main admin
        dashboard.
    class_names : list[str]
        Ordered class names so ``class_names[i]`` corresponds to the
        integer label ``i``.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_probs: np.ndarray
    class_names: List[str]


@dataclass
class MetricsResult:
    """Structured metrics returned by :func:`compute_metrics`.

    Attributes
    ----------
    accuracy : float
        Overall classification accuracy.
    per_class : dict[str, dict[str, float]]
        Mapping ``class_name -> {"precision", "recall", "f1", "support"}``.
    macro : dict[str, float]
        Macro-averaged ``precision``, ``recall`` and ``f1``.
    weighted : dict[str, float]
        Support-weighted ``precision``, ``recall`` and ``f1``.
    class_names : list[str]
        Ordered class names (kept on the object for downstream use).
    n_samples : int
        Total number of samples evaluated.
    """

    accuracy: float
    per_class: Dict[str, Dict[str, float]]
    macro: Dict[str, float]
    weighted: Dict[str, float]
    class_names: List[str] = field(default_factory=list)
    n_samples: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame: one row per class + ``macro/weighted`` avg rows."""
        rows: List[Dict[str, Any]] = []
        for name, m in self.per_class.items():
            rows.append(
                {
                    "class": name,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "support": m["support"],
                }
            )
        total_support = int(sum(m["support"] for m in self.per_class.values()))
        rows.append(
            {
                "class": "macro avg",
                "precision": self.macro["precision"],
                "recall": self.macro["recall"],
                "f1": self.macro["f1"],
                "support": total_support,
            }
        )
        rows.append(
            {
                "class": "weighted avg",
                "precision": self.weighted["precision"],
                "recall": self.weighted["recall"],
                "f1": self.weighted["f1"],
                "support": total_support,
            }
        )
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, JSON-serialisable representation."""
        return {
            "accuracy": float(self.accuracy),
            "per_class": {
                name: {
                    "precision": float(m["precision"]),
                    "recall": float(m["recall"]),
                    "f1": float(m["f1"]),
                    "support": int(m["support"]),
                }
                for name, m in self.per_class.items()
            },
            "macro": {k: float(v) for k, v in self.macro.items()},
            "weighted": {k: float(v) for k, v in self.weighted.items()},
            "class_names": list(self.class_names),
            "n_samples": int(self.n_samples),
        }


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` from a dataloader batch.

    The :class:`torchvision.datasets.ImageFolder`-backed loaders used
    elsewhere in this task yield ``(images, labels)`` tuples; this
    helper additionally accepts dict-style batches for
    forward-compatibility with HuggingFace-style datasets.
    """
    if isinstance(batch, Mapping):
        if "image" in batch and "label" in batch:
            return batch["image"], batch["label"]
        if "pixel_values" in batch and "labels" in batch:
            return batch["pixel_values"], batch["labels"]
        raise KeyError(
            "Mapping batch must contain 'image'/'label' or "
            "'pixel_values'/'labels' keys."
        )
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch).__name__}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    device: Union[str, torch.device],
) -> EvaluationResult:
    """Run ``model`` over ``dataloader`` and collect predictions.

    The model is switched to ``eval`` mode and inference is wrapped in
    ``torch.no_grad`` so gradients are not tracked -- this reduces
    memory use and speeds up evaluation. The model's raw outputs are
    treated as logits and passed through ``softmax`` to produce the
    probabilities stored on the returned :class:`EvaluationResult`.

    Parameters
    ----------
    model : nn.Module
        Trained classifier. Expected to return logits of shape
        ``(batch_size, n_classes)`` -- this matches the models built by
        :func:`task2_3_4_cv_quality.src.model.get_model`.
    dataloader : DataLoader
        Dataloader yielding ``(images, labels)`` batches (typically the
        ``test_loader`` on a
        :class:`~task2_3_4_cv_quality.src.train.DataLoaderBundle`).
    class_names : Sequence[str]
        Ordered class names. ``class_names[i]`` must correspond to the
        integer label ``i``.
    device : str | torch.device
        Device used for inference, e.g. ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    EvaluationResult
        Dataclass with ``y_true``, ``y_pred``, ``y_probs`` and
        ``class_names``.

    Raises
    ------
    ValueError
        If ``class_names`` is empty.
    RuntimeError
        If ``dataloader`` yields no batches.

    Examples
    --------
    >>> from task2_3_4_cv_quality.src.train import create_dataloaders
    >>> from task2_3_4_cv_quality.src.model import get_model
    >>> bundle = create_dataloaders(verbose=False)
    >>> model = get_model(
    ...     num_classes=len(bundle.classes), model_type="resnet50"
    ... )
    >>> result = evaluate_model(
    ...     model, bundle.test_loader, bundle.classes, "cpu"
    ... )
    >>> result.y_pred.shape == result.y_true.shape
    True
    """
    if len(class_names) == 0:
        raise ValueError("`class_names` must be non-empty.")

    device = torch.device(device)
    model.to(device)
    model.eval()

    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    if not all_labels:
        raise RuntimeError("Dataloader yielded no batches; nothing to evaluate.")

    return EvaluationResult(
        y_true=np.concatenate(all_labels).astype(np.int64),
        y_pred=np.concatenate(all_preds).astype(np.int64),
        y_probs=np.concatenate(all_probs).astype(np.float32),
        class_names=list(class_names),
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred: Union[Sequence[int], np.ndarray],
    class_names: Sequence[str],
) -> MetricsResult:
    """Compute accuracy, per-class P/R/F1 and macro / weighted averages.

    ``zero_division=0`` is passed through to scikit-learn so that
    classes which receive no predicted samples contribute a 0 rather
    than raising a warning. This matters here because the raw dataset
    is heavily imbalanced -- minority classes can genuinely end up
    with zero predictions on a small test split.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth class indices.
    y_pred : array-like of int
        Predicted class indices.
    class_names : Sequence[str]
        Ordered class names. ``class_names[i]`` must correspond to the
        integer label ``i``.

    Returns
    -------
    MetricsResult
        Dataclass holding accuracy, per-class metrics, and macro /
        weighted averages.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different shapes, or if
        ``class_names`` is empty.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}."
        )
    if len(class_names) == 0:
        raise ValueError("`class_names` must be non-empty.")

    labels = list(range(len(class_names)))

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        zero_division=0,
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    return MetricsResult(
        accuracy=accuracy,
        per_class=per_class,
        macro={
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        weighted={
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        class_names=list(class_names),
        n_samples=int(y_true_arr.shape[0]),
    )


# ---------------------------------------------------------------------------
# Helpers: confusion matrix and pretty-printing
# ---------------------------------------------------------------------------

def confusion_matrix_df(
    result: EvaluationResult,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Return a labelled confusion matrix as a DataFrame.

    Parameters
    ----------
    result : EvaluationResult
        The result returned by :func:`evaluate_model`.
    normalize : {'true', 'pred', 'all', None}, optional
        Passed through to :func:`sklearn.metrics.confusion_matrix`.
        ``'true'`` normalises rows (share of each true class that was
        predicted as each class), ``'pred'`` normalises columns, and
        ``'all'`` normalises over the whole matrix. ``None`` (default)
        leaves raw counts.

    Returns
    -------
    pandas.DataFrame
        Square DataFrame indexed by true class with columns for each
        predicted class.
    """
    labels = list(range(len(result.class_names)))
    cm = confusion_matrix(
        result.y_true,
        result.y_pred,
        labels=labels,
        normalize=normalize,
    )
    return pd.DataFrame(
        cm,
        index=pd.Index(result.class_names, name="true"),
        columns=pd.Index(result.class_names, name="pred"),
    )


def print_results_table(
    metrics: MetricsResult,
    title: str = "Evaluation Results",
    float_format: str = "{:.4f}",
) -> None:
    """Pretty-print the metrics as a table to ``stdout``.

    The output mirrors :func:`sklearn.metrics.classification_report`
    but is rendered via :meth:`pandas.DataFrame.to_string`, which keeps
    the alignment clean when notebooks pipe the output through their
    usual formatters.

    Parameters
    ----------
    metrics : MetricsResult
        Metrics produced by :func:`compute_metrics`.
    title : str, optional
        Heading printed above the table.
    float_format : str, optional
        Format spec applied to the ``precision``, ``recall`` and ``f1``
        columns (default: 4 decimal places).
    """
    bar = "=" * 80
    print(bar)
    print(title)
    print(bar)
    print(f"Samples evaluated : {metrics.n_samples}")
    print(f"Overall accuracy  : {metrics.accuracy:.4f}")
    print("-" * 80)

    df = metrics.to_dataframe()
    formatters = {
        "precision": float_format.format,
        "recall": float_format.format,
        "f1": float_format.format,
    }
    print(df.to_string(index=False, formatters=formatters))
    print(bar)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# AA-34: Plotting — confusion matrix heatmap & per-class F1 bar chart
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    result: "EvaluationResult",
    normalize: str = "true",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """Seaborn heatmap of the confusion matrix, optionally saved to file."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(range(len(result.class_names)))
    cm = confusion_matrix(result.y_true, result.y_pred, labels=labels, normalize=normalize)

    n = len(result.class_names)
    if figsize is None:
        figsize = (max(10, n * 0.55), max(8, n * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=result.class_names,
        yticklabels=result.class_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_metrics(
    metrics: "MetricsResult",
    metric: str = "f1",
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of a per-class metric (default: F1)."""
    import matplotlib.pyplot as plt

    names = list(metrics.per_class.keys())
    values = [metrics.per_class[n][metric] for n in names]

    sorted_pairs = sorted(zip(values, names))
    values, names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.3)))

    colors = ["#e74c3c" if v < 0.9 else "#f39c12" if v < 0.95 else "#2ecc71" for v in values]
    ax.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Per-Class {metric.upper()} Score", fontsize=14, fontweight="bold")
    ax.set_xlim(min(values) * 0.95, 1.0)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# AA-35: Failure case analysis
# ---------------------------------------------------------------------------


def analyse_failures(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    device: Union[str, torch.device],
    max_per_class: int = 5,
) -> Dict[str, List[Dict]]:
    """Collect misclassified images grouped by true class.

    Returns dict mapping true class name -> list of dicts with keys:
    image (Tensor), true_label, pred_label, confidence.
    """
    model.eval()
    failures: Dict[str, List[Dict]] = {name: [] for name in class_names}

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for images, labels in dataloader:
            images_dev, labels_dev = images.to(device), labels.to(device)
            outputs = model(images_dev)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            wrong = preds != labels_dev
            if not wrong.any():
                continue

            for idx in wrong.nonzero(as_tuple=True)[0]:
                true_cls = class_names[labels[idx].item()]
                if len(failures[true_cls]) >= max_per_class:
                    continue

                img = images[idx].cpu() * std + mean
                img = img.clamp(0, 1)

                failures[true_cls].append({
                    "image": img,
                    "true_label": true_cls,
                    "pred_label": class_names[preds[idx].item()],
                    "confidence": confs[idx].item(),
                })

    # Remove classes with no failures
    return {k: v for k, v in failures.items() if v}


if __name__ == "__main__":
    # Small self-contained demo using synthetic labels so the module can
    # be smoke-tested without needing a trained checkpoint.
    print("Running evaluate.py self-test with synthetic predictions...")
    rng = np.random.default_rng(seed=42)

    demo_class_names = ["Apple__Healthy", "Apple__Rotten", "Banana__Healthy"]
    n_classes = len(demo_class_names)
    n_samples = 300

    y_true_demo = rng.integers(low=0, high=n_classes, size=n_samples)
    # Create predictions that agree with y_true 80% of the time.
    y_pred_demo = y_true_demo.copy()
    flip_mask = rng.random(n_samples) < 0.20
    y_pred_demo[flip_mask] = rng.integers(
        low=0, high=n_classes, size=int(flip_mask.sum())
    )

    demo_metrics = compute_metrics(y_true_demo, y_pred_demo, demo_class_names)
    print_results_table(demo_metrics, title="Synthetic Demo (80% agreement)")

    demo_result = EvaluationResult(
        y_true=y_true_demo,
        y_pred=y_pred_demo,
        y_probs=np.zeros((n_samples, n_classes), dtype=np.float32),
        class_names=demo_class_names,
    )
    print("\nConfusion matrix (counts):")
    print(confusion_matrix_df(demo_result).to_string())

    print("\n[OK] evaluate.py self-test passed.")