from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


from src.plots import save_classification_plots, save_final_bce_plot


def train_logistic_regression(X_train, Y_train, X_validation, Y_validation):
    # x - mean / sd: среднее и отклонение считаем только по train.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)

    #logistic regression data split 1 / 0
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, Y_train.ravel())
    probabilities = baseline.predict_proba(X_validation)[:, 1]
    prediction = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(Y_validation, prediction)

    print("Baseline validation accuracy:", accuracy)
    labels = Y_validation.ravel()
    print(classification_report(labels, prediction, labels=[0, 1], digits=4, zero_division=0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"logistic_{timestamp}_rows{len(X_train)}"

    if len(np.unique(labels)) == 2:
        print("Baseline validation ROC-AUC:", roc_auc_score(labels, probabilities))

    # Лекция 2: BCE по вероятностям, до преобразования в классы.
    # Эквивалентная формула по логитам устойчива даже для очень уверенных ответов.
    # Это итоговый loss после fit(), а не история обучения по эпохам.
    train_logits = baseline.decision_function(X_train)
    validation_logits = baseline.decision_function(X_validation)
    train_loss = np.mean(np.logaddexp(0, train_logits) - Y_train.ravel() * train_logits)
    validation_loss = np.mean(np.logaddexp(0, validation_logits) - labels * validation_logits)
    print("Baseline train BCE:", train_loss)
    print("Baseline validation BCE:", validation_loss)
    # Все картинки создаются в src/plots.py.
    save_classification_plots(run_name, labels, probabilities, "Validation", "logistic_regression")
    save_final_bce_plot(run_name, train_loss, validation_loss)
