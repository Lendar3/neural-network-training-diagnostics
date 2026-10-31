from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.logistic_regression import train_logistic_regression
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


data = None

while True:
    print("\n1. Load data")
    print("2. Logistic regression")
    print("3. BCE training: learning rate, epochs, training rows")
    print("4. Evaluate saved BCE weights on test and save plots")
    print("0. Exit")
    choice = input("Choose an option: ").strip()

    match choice:
        case "1":
            if data is None:
                data = load_data()
                X_train, Y_train, X_validation, Y_validation, X_test, Y_test = data
                print("Data loaded.")
            else:
                print("Data is already loaded.")
        case "2":
            if data is None:
                print("Load data first: choose 1.")
            else:
                train_logistic_regression(X_train, Y_train, X_validation, Y_validation)
        case "3":
            if data is None:
                print("Load data first: choose 1.")
            else:
                learning_rate = float(input("Learning rate (e.g. 0.001): "))
                epochs = int(input("Epochs: "))
                rows = int(input(f"Training rows (1-{len(X_train)}): "))
                if not np.isfinite(learning_rate) or learning_rate <= 0 or epochs < 1 or not 1 <= rows <= len(X_train):
                    print("Use a positive learning rate, epochs >= 1 and a valid row count.")
                    continue
                train_neural_network(
                    X_train, Y_train, X_validation, Y_validation,
                    learning_rate, epochs, rows,
                )
        case "4":
            if data is None:
                print("Load data first: choose 1.")
                continue
            weight_files = sorted(WEIGHTS_DIR.glob("bce_*.npz"))
            if not weight_files:
                print("No saved weights. Run BCE training first.")
                continue
            for number, path in enumerate(weight_files, start=1):
                print(f"{number}. {path.name}")
            selected = input("Choose weights by number (0 to cancel): ").strip()
            if selected == "0":
                continue
            if not selected.isdigit() or not 1 <= int(selected) <= len(weight_files):
                print("Choose a number from the list.")
                continue
            evaluate_neural_network(weight_files[int(selected) - 1], X_test, Y_test)
        case "0":
            break
        case _:
            print("Choose 0, 1, 2, 3 or 4.")
