from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


IMG_DIR = Path(__file__).resolve().parent.parent / "img"


# История всех эпох остаётся доступна после завершения программы.
def save_loss_plot(run_name, train_losses, validation_losses, best_epoch):
    img_dir = IMG_DIR / "bce"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    epoch_numbers = range(1, len(train_losses) + 1)
    ax.plot(epoch_numbers, train_losses, label="Train")
    ax.plot(epoch_numbers, validation_losses, label="Validation")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="BCE loss", title="Training and validation BCE loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(img_dir / f"{run_name}_train_validation_loss.png", dpi=150)
    plt.close(fig)


# Лекция 1: true — по строкам, predicted — по столбцам, класс 1 первым.
def save_classification_plots(run_name, labels, probabilities, split, folder="bce"):
    img_dir = IMG_DIR / folder
    model_name = "Neural network" if folder == "bce" else "Logistic regression"
    img_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        labels, predictions, labels=[1, 0], display_labels=["Signal (1)", "Background (0)"],
        ax=ax, colorbar=False, cmap="Blues", values_format="d",
    )
    names = (("TP", "FN"), ("FP", "TN"))
    for row in range(2):
        for column in range(2):
            count = display.confusion_matrix[row, column]
            display.text_[row, column].set_text(f"{names[row][column]}\n{count:,}")
    ax.set_title(f"{model_name}: {split.lower()} confusion matrix\nThreshold = 0.5")
    fig.tight_layout()
    fig.savefig(img_dir / f"{run_name}_{split.lower()}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    if len(np.unique(labels)) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(labels, probabilities, ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"{model_name}: {split.lower()} ROC curve")
        fig.tight_layout()
        fig.savefig(img_dir / f"{run_name}_{split.lower()}_roc_curve.png", dpi=150)
        plt.close(fig)
    print("Plots saved:", img_dir)


# Итоговые значения BCE logistic regression после fit().
def save_final_bce_plot(run_name, train_loss, validation_loss):
    img_dir = IMG_DIR / "logistic_regression"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Train", "Validation"], [train_loss, validation_loss], color=["tab:blue", "tab:orange"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylabel="BCE loss", title="Logistic regression: final BCE after fit")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(img_dir / f"{run_name}_final_train_validation_bce.png", dpi=150)
    plt.close(fig)
    print("Plots saved:", img_dir)
