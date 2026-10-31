from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.plots import save_loss_plot, save_classification_plots


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "data" / "weights"


# Одинаковая архитектура для обучения и загрузки сохранённых весов.
def create_model():
    return nn.Sequential(nn.Linear(28,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))


# Обучение нейросети с BCE, восстановление лучших весов и график
def train_neural_network(X_train, Y_train, X_validation, Y_validation, learning_rate, epochs, rows, seed=12):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Scaler обучается только на выбранных строках, даже если rows = 1.
    scaler = StandardScaler()
    # z = x - mean /sd
    X_train = scaler.fit_transform(X_train[:rows])
    Y_train = Y_train[:rows]
    #z = x - mean / sd (mean and sd from x train)
    X_validation = scaler.transform(X_validation)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).float().to(device)

    X_validation_tensor = torch.from_numpy(X_validation).float().to(device)
    Y_validation_tensor = torch.from_numpy(Y_validation).float().to(device)


    #creating hidden layers

    torch.manual_seed(seed)
    model = create_model().to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    validation_losses = []

    best_validation_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(epochs):

        # Обучение: один шаг на всей обучающей выборке
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_tensor)
        loss = loss_function(logits, Y_train_tensor)

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            train_logits = model(X_train_tensor)
            train_loss = loss_function(train_logits, Y_train_tensor)

            validation_logits = model(X_validation_tensor)
            validation_loss = loss_function(
                validation_logits, Y_validation_tensor
            )

        train_losses.append(train_loss.item())
        validation_losses.append(validation_loss.item())

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_losses[-1]:.4f} | "
                f"Validation: {validation_losses[-1]:.4f}"
            )

        current_validation_loss = validation_loss.item()
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_epoch = epoch + 1
            best_weights = deepcopy(model.state_dict())

    print("Validation loss последней эпохи:", validation_losses[-1])
    print("Лучшая эпоха:", best_epoch)

    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        restored_logits = model(X_validation_tensor)
        restored_loss = loss_function(
            restored_logits, Y_validation_tensor
        )

    print("Validation loss восстановленной модели:", restored_loss.item())


    with torch.no_grad():
        validation_logits = model(X_validation_tensor)
        probabilities = torch.sigmoid(validation_logits)
        predictions = (probabilities >= 0.5).cpu().numpy().ravel()

    neural_accuracy = accuracy_score(Y_validation.ravel(), predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_validation.ravel(), predictions, average="binary", zero_division=0,
    )

    print("Neural network validation accuracy:", neural_accuracy)
    if len(np.unique(Y_validation)) == 2:
        print("Validation ROC-AUC:", roc_auc_score(Y_validation.ravel(), probabilities.cpu().numpy().ravel()))

    # Создаём папку и имя файла для этого запуска.
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = (
        f"bce_{timestamp}_rows{rows}"
        f"_lr{learning_rate:g}_epochs{epochs}_seed{seed}"
    )
    weights_path = WEIGHTS_DIR / f"{run_name}.npz"

    # Переносим лучшие веса с GPU на CPU и превращаем в массивы NumPy.
    data_to_save = {}
    for name, weight in best_weights.items():
        weight_on_cpu = weight.cpu()
        data_to_save[name] = weight_on_cpu.numpy()

    # Эти значения нужны для такой же стандартизации новых данных.
    data_to_save["scaler_mean"] = scaler.mean_
    data_to_save["scaler_scale"] = scaler.scale_

    # История ошибок нужна для построения графиков без повторного обучения.
    data_to_save["train_losses"] = train_losses
    data_to_save["validation_losses"] = validation_losses
    # Настройки запуска.
    data_to_save["best_epoch"] = best_epoch
    data_to_save["learning_rate"] = learning_rate
    data_to_save["epochs"] = epochs
    data_to_save["training_rows"] = rows
    data_to_save["seed"] = seed
    # Результаты на validation.
    data_to_save["validation_accuracy"] = neural_accuracy
    data_to_save["validation_precision"] = precision
    data_to_save["validation_recall"] = recall
    data_to_save["validation_f1"] = f1

    # ** передаёт содержимое словаря как именованные значения для NPZ-файла.
    np.savez(weights_path, **data_to_save)
    print("Weights saved:", weights_path)

    # Подготавливаем обычные массивы для графиков.
    validation_labels = Y_validation.ravel()
    validation_probabilities = probabilities.cpu().numpy().ravel()

    save_loss_plot(run_name, train_losses, validation_losses, best_epoch)
    save_classification_plots(
        run_name, validation_labels, validation_probabilities, "Validation"
    )
    return weights_path


# Загружаем выбранную модель и оцениваем её без повторного обучения.
def evaluate_neural_network(weights_path, X_test, Y_test):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model()
    with np.load(weights_path, allow_pickle=False) as saved:
        weights = {name: torch.from_numpy(saved[name]) for name in model.state_dict()}
        model.load_state_dict(weights)
        # Test преобразуем теми же mean и scale, что использовались при обучении.
        X_test = ((X_test - saved["scaler_mean"]) / saved["scaler_scale"]).astype(np.float32)
        train_losses = saved["train_losses"]
        validation_losses = saved["validation_losses"]
        best_epoch = int(saved["best_epoch"])
        print(f"Training rows: {int(saved['training_rows'])} | "
              f"Learning rate: {float(saved['learning_rate'])} | "
              f"Epochs: {int(saved['epochs'])} | Best epoch: {best_epoch}")

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        targets = torch.from_numpy(Y_test).float().to(device)
        test_loss = nn.BCEWithLogitsLoss()(logits, targets).item()
        probabilities = torch.sigmoid(logits).cpu().numpy().ravel()

    labels = Y_test.ravel()
    predictions = (probabilities >= 0.5).astype(int)
    print("Test BCE loss:", test_loss)
    print("Test accuracy:", accuracy_score(labels, predictions))
    if len(np.unique(labels)) == 2:
        print("Test ROC-AUC:", roc_auc_score(labels, probabilities))
    print(classification_report(labels, predictions, labels=[0, 1], digits=4, zero_division=0))

    # Названия содержат запуск и содержание графика; разные модели не смешиваются.
    run_name = Path(weights_path).stem
    save_loss_plot(run_name, train_losses, validation_losses, best_epoch)

    save_classification_plots(run_name, labels, probabilities, "Test")
