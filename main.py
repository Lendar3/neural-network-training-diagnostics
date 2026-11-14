from pathlib import Path
import csv

import numpy as np
from sklearn.model_selection import train_test_split

from src.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from src.neural_network import train_neural_network, evaluate_neural_network, WEIGHTS_DIR



RESULTS_PATH = Path(__file__).resolve().parent / "results" / "metrics_slices.csv"


def save_result(weights_path, model_name, split, metrics=None):
    with np.load(weights_path, allow_pickle=False) as saved:
        if metrics is None:
            metrics = {
                "bce": float(saved["validation_bce"]),
                "accuracy": float(saved["validation_accuracy"]),
                "precision": float(saved["validation_precision"]),
                "recall": float(saved["validation_recall"]),
                "f1": float(saved["validation_f1"]),
                "roc_auc": float(saved["validation_roc_auc"]),
            }
        row = {
            "run": Path(weights_path).stem,
            "model": model_name,
            "split": split,
            "seed": int(saved["seed"]),
            "threshold": 0.5,
            "training_rows": int(saved["training_rows"]),
            "validation_rows": int(saved["validation_rows"]),
            "architecture": str(saved["architecture"]),
            "optimizer": str(saved["optimizer"]),
            "batch_size": str(saved.get("batch_size", "")),
            "learning_rate": str(saved.get("learning_rate", "")),
            "epochs": str(saved.get("epochs", "")),
            "best_epoch": str(saved.get("best_epoch", "")),
            "training_seconds": float(saved["training_seconds"]),
            "bce": metrics["bce"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

    print(model_name, split)
    print("BCE:", round(metrics["bce"], 4))
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))


def select_data(full_data, rows):
    X_train, Y_train, X_validation, Y_validation, X_test, Y_test = full_data
    fraction = rows / len(X_train)
    validation_rows = round(len(X_validation) * fraction)
    test_rows = round(len(X_test) * fraction)
    return (
        X_train[:rows],
        Y_train[:rows],
        X_validation[:validation_rows],
        Y_validation[:validation_rows],
        X_test[:test_rows],
        Y_test[:test_rows],
    )


def load_data():
    PROJECT_ROOT = Path(__file__).resolve().parent
    data_path = PROJECT_ROOT / "data" / "processed" / "higgs.npz"

    with np.load(data_path) as data:
        X = data["X"]
        Y = data["Y"]


    X_train, X_rest, Y_train, Y_rest = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=12,
        stratify=Y,
    )
    X_validation, X_test, Y_validation, Y_test = train_test_split(
        X_rest,
        Y_rest,
        test_size=0.50,
        random_state=12,
        stratify=Y_rest,
    )

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test



def choose_file(paths, title):
    print(title)
    for number, path in enumerate(paths, 1):
        print(f"{number}. {path.name}")
    selected = input("Number (0 to cancel): ").strip()
    if selected == "0":
        return None
    return paths[int(selected) - 1]


def matching_models(pattern, rows=None):
    paths = []
    for path in sorted(WEIGHTS_DIR.glob(pattern)):
        with np.load(path, allow_pickle=False) as saved:
            if rows is None or int(saved["training_rows"]) == rows:
                paths.append(path)
    return paths


def main():
    full_data = None
    selected_data = None
    while True:
        print("1. Load data")
        print("2. Train baseline")
        print("3. Train neural network")
        print("4. Evaluate on test")
        print("0. Exit")
        choice = input("Choose an option: ").strip()
        try:
            if choice == "0":
                break
            if choice == "1":
                full_data = load_data()
                rows = int(input("Training rows: "))
                selected_data = select_data(full_data, rows)
            elif choice in ("2", "3"):
                if selected_data is None:
                    print("Choose option 1 first.")
                    continue
                X_train, Y_train, X_val, Y_val, X_test, Y_test = selected_data
                if choice == "2":
                    path = train_logistic_regression(X_train, Y_train, X_val, Y_val)
                    save_result(path, "logistic_regression", "validation")
                else:
                    lr = float(input("Learning rate (e.g. 0.001): "))
                    epochs = int(input("Epochs: "))
                    path = train_neural_network(X_train, Y_train, X_val, Y_val, lr, epochs)
                    save_result(path, "neural_network", "validation")
            elif choice == "4":
                if full_data is None:
                    print("Choose option 1 first.")
                    continue
                path = choose_file(matching_models("bce_slice_*.npz"), "Choose neural network.")
                if path is None:
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    rows = int(saved["training_rows"])
                baseline = choose_file(matching_models("logistic_slice_*.npz", rows),"Choose baseline.")
                if baseline is None:
                    continue
                test_data = select_data(full_data, rows)
                network_metrics = evaluate_neural_network(path, test_data[4], test_data[5])
                save_result(path, "neural_network", "test", network_metrics)
                baseline_metrics = evaluate_logistic_regression(baseline, test_data[4], test_data[5])
                save_result(baseline, "logistic_regression", "test", baseline_metrics)
            else:
                print("Choose 0, 1, 2, 3 or 4.")
        except ValueError as error:
            print("Invalid input or configuration:", error)


if __name__ == "__main__":
    main()
