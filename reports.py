from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_auc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_col, y_prob_col))


def _safe_auprc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    return float(average_precision_score(y_true_col, y_prob_col))


def plot_history_simple(results_dir: Path, history: pd.DataFrame) -> None:
    plots_dir = results_dir / "plots"
    ensure_dir(plots_dir)
    if history.empty:
        return

    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_loss_total"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss_total"], label="val_loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["train_main_f1_macro"], label="train_f1_macro")
    plt.plot(history["epoch"], history["val_main_f1_macro"], label="val_f1_macro")
    plt.xlabel("Época")
    plt.ylabel("F1 Macro")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "f1_curve.png", dpi=130)
    plt.close(fig)


def build_metrics_json(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
    output_path: Path,
    extra: Dict[str, Any] | None = None,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    payload: Dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(y_true.shape[0]),
        "classes": list(class_names),
        "per_class": {},
    }
    for i, cls in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prob = y_prob[:, i]
        payload["per_class"][cls] = {
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auroc": _safe_auc(yt, prob),
            "auprc": _safe_auprc(yt, prob),
        }
    if extra:
        payload["extra"] = extra
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], threshold: float, output_path: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
    for i, cls in enumerate(classes):
        ax = axes[i]
        cm = mcm[i]
        ax.imshow(cm)
        ax.set_title(cls)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(int(cm[r, c])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _plot_roc(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    for i, cls in enumerate(classes):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, y_prob[:, i])
        auc = roc_auc_score(yt, y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{cls} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":", label="chance")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _plot_pr(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    for i, cls in enumerate(classes):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            continue
        p, r, _ = precision_recall_curve(yt, y_prob[:, i])
        ap = average_precision_score(yt, y_prob[:, i])
        plt.plot(r, p, label=f"{cls} AP={ap:.3f}")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _table(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], threshold: float) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    rows = []
    for i, cls in enumerate(classes):
        yt, yp, prob = y_true[:, i], y_pred[:, i], y_prob[:, i]
        rows.append({
            "class": cls,
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "auroc": _safe_auc(yt, prob),
            "auprc": _safe_auprc(yt, prob),
        })
    return pd.DataFrame(rows)


def generate_split_reports(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    test_bases: Sequence[str],
    class_names: Sequence[str],
    threshold: float,
    output_root: Path,
) -> None:
    ensure_dir(output_root)
    groups = {"global": np.arange(len(test_bases))}
    for base in sorted(set(test_bases)):
        idx = np.where(np.asarray(test_bases) == base)[0]
        groups[base] = idx

    for group_name, idx in groups.items():
        group_dir = output_root / group_name
        ensure_dir(group_dir)
        yt = y_true[idx]
        yp = y_prob[idx]
        _plot_confusion(yt, yp, class_names, threshold, group_dir / "confusion_superclass.png")
        _plot_pr(yt, yp, class_names, group_dir / "pr_curves.png")
        _plot_roc(yt, yp, class_names, group_dir / "roc_curves.png")
        _table(yt, yp, class_names, threshold).to_csv(group_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")


def save_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(6, 6))
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if len(np.unique(yt)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(yt, y_prob[:, i], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1)
    plt.plot([0, 1], [0, 1], linestyle=":", color="black")
    plt.xlabel("Prob prevista")
    plt.ylabel("Fração positiva")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def export_base_generalization_table(y_true: np.ndarray, y_prob: np.ndarray, test_bases: Sequence[str], classes: Sequence[str], threshold: float, output_path: Path) -> None:
    rows = []
    tb = np.asarray(test_bases)
    for base in sorted(set(test_bases)):
        idx = np.where(tb == base)[0]
        yt, yp = y_true[idx], y_prob[idx]
        ypred = (yp >= threshold).astype(int)
        f1_macro = float(f1_score(yt, ypred, average="macro", zero_division=0))
        try: auroc = float(roc_auc_score(yt, yp, average="macro"))
        except Exception: auroc = float("nan")
        try: auprc = float(average_precision_score(yt, yp, average="macro"))
        except Exception: auprc = float("nan")
        class_f1 = [f1_score(yt[:,i], ypred[:,i], zero_division=0) for i in range(len(classes))]
        pior = classes[int(np.argmin(class_f1))]
        rows.append({"Test base": base, "F1 macro": f1_macro, "AUROC macro": auroc, "AUPRC macro": auprc, "pior classe": pior})
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding='utf-8-sig')
