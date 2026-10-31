from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.logistic_regression import train_logistic_regression, evaluate_logistic_regression
from src.experiment import SEED, select_data
from src.neural_network import train_neural_network, evaluate_neural_network, WEIGHTS_DIR



def load_data():
    PROJECT_ROOT = Path(__file__).resolve().parent
    data_path = PROJECT_ROOT / "data" / "processed" / "higgs.npz"

    with np.load(data_path) as data:
        X = data["X"]
        Y = data["Y"]
    print(X.shape, Y.shape)


    X_train, X_rest, Y_train, Y_rest = train_test_split(
        X,
        Y,
        test_size=0.30,
        random_state=SEED,
        stratify=Y,
    )
    X_validation, X_test, Y_validation, Y_test = train_test_split(
        X_rest,
        Y_rest,
        test_size=0.50,
        random_state=SEED,
        stratify=Y_rest,
    )

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test



def choose_file(paths, title):
    if not paths:
        print("No matching saved models. Train the model first.")
        return None
    print(title)
    for number, path in enumerate(paths, 1):
        print(f"{number}. {path.name}")
    selected = input("Number (0 to cancel): ").strip()
    if selected == "0":
        return None
    if not selected.isdigit() or not 1 <= int(selected) <= len(paths):
        raise ValueError("Choose a number from the list.")
    return paths[int(selected) - 1]


def matching_models(pattern, split_id=None):
    paths = []
    for path in sorted(WEIGHTS_DIR.glob(pattern)):
        with np.load(path, allow_pickle=False) as saved:
            # Legacy runs used a different subset protocol; keep them as history.
            if "split_version" not in saved or str(saved["split_version"]) != "proportional-v1":
                continue
            if split_id is None or str(saved["split_id"]) == split_id:
                paths.append(path)
    return paths


def main():
    full_data = None
    selected_data = None
    metadata = None
    while True:
        print("\nSeed = 12 (fixed)")
        print("1. Load data / choose training rows (validation and test scale too)")
        print("2. Train and save logistic regression baseline")
        print("3. Train neural network: learning rate and epochs")
        print("4. Final test comparison: saved neural network and baseline")
        print("0. Exit")
        choice = input("Choose an option: ").strip()
        try:
            if choice == "0":
                break
            if choice == "1":
                if full_data is None:
                    full_data = load_data()
                rows = int(input(f"Training rows (10-{len(full_data[0])}): "))
                new_data, new_metadata = select_data(full_data, rows)
                selected_data, metadata = new_data, new_metadata
                total = metadata["source_rows"]
                for name in ("training_rows", "validation_rows", "test_rows"):
                    print(f"{name}: {metadata[name]:,} ({100 * metadata[name] / total:.4f}% of original data)")
            elif choice in ("2", "3"):
                if selected_data is None:
                    print("Choose option 1 first.")
                    continue
                X_train, Y_train, X_val, Y_val, _, _ = selected_data
                if choice == "2":
                    train_logistic_regression(X_train, Y_train, X_val, Y_val, metadata)
                else:
                    lr = float(input("Learning rate (e.g. 0.001): "))
                    epochs = int(input("Epochs: "))
                    train_neural_network(X_train, Y_train, X_val, Y_val, lr, epochs, metadata)
            elif choice == "4":
                if full_data is None:
                    print("Choose option 1 first.")
                    continue
                path = choose_file(matching_models("bce_*.npz"), "Select the network chosen using validation.")
                if path is None:
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    rows = int(saved["training_rows"])
                    split_id = str(saved["split_id"])
                    if int(saved["source_rows"]) != sum(len(full_data[i]) for i in (0, 2, 4)):
                        raise ValueError("Source dataset size differs from the saved run.")
                baseline = choose_file(matching_models("logistic_*.npz", split_id),
                                       "Select baseline trained on the same data.")
                if baseline is None:
                    continue
                test_data, test_metadata = select_data(full_data, rows)
                if test_metadata["split_id"] != split_id:
                    raise ValueError("Saved split could not be reproduced.")
                print("Final test evaluation. Do not use these results to tune hyperparameters.")
                evaluate_neural_network(path, test_data[4], test_data[5], test_metadata)
                evaluate_logistic_regression(baseline, test_data[4], test_data[5], test_metadata)
            else:
                print("Choose 0, 1, 2, 3 or 4.")
        except ValueError as error:
            print("Invalid input or configuration:", error)


if __name__ == "__main__":
    main()
