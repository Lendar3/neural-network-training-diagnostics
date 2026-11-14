from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from src.plots import save_classification_plots, save_final_bce_plot

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'weights'


def train_logistic_regression(X_train, Y_train, X_validation, Y_validation):
    start = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    baseline = LogisticRegression(max_iter=1000, random_state=12)
    baseline.fit(X_train, Y_train.ravel())
    seconds = time.perf_counter() - start
    train_logits = baseline.decision_function(X_train)
    logits = baseline.decision_function(X_validation)
    probabilities = expit(logits)
    train_loss = np.mean(np.logaddexp(0, train_logits) - Y_train.ravel() * train_logits)
    val_loss = np.mean(np.logaddexp(0, logits) - Y_validation.ravel() * logits)
    labels = Y_validation.ravel()
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    metrics = {
        'bce': float(val_loss),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc_score(labels, probabilities),
    }
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = f'logistic_slice_{timestamp}_rows{len(X_train)}_seed12'
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f'{run_name}.npz'
    # Для бинарной логистической регрессии этих параметров достаточно для inference.
    saved = {}
    saved['training_rows'] = len(X_train)
    saved['validation_rows'] = len(X_validation)
    saved.update(coef=baseline.coef_, intercept=baseline.intercept_,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, seed=12,
                 training_seconds=seconds, train_bce=train_loss)
    for name, value in metrics.items():
        saved[f'validation_{name}'] = value
    saved['architecture'] = '28 -> 1'
    saved['optimizer'] = 'lbfgs'
    np.savez(path, **saved)
    save_classification_plots(run_name, Y_validation.ravel(), probabilities,
                              'Validation', 'logistic_regression')
    save_final_bce_plot(run_name, train_loss, val_loss)
    return path


def evaluate_logistic_regression(path, X_test, Y_test):
    with np.load(path, allow_pickle=False) as saved:
        scaled = (X_test - saved['scaler_mean']) / saved['scaler_scale']
        logits = (scaled @ saved['coef'].T + saved['intercept']).ravel()
    probabilities = expit(logits)
    bce = np.mean(np.logaddexp(0, logits) - Y_test.ravel() * logits)
    labels = Y_test.ravel()
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    metrics = {
        'bce': float(bce),
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc_score(labels, probabilities),
    }
    save_classification_plots(Path(path).stem, Y_test.ravel(), probabilities,
                              'Test', 'logistic_regression')
    return metrics
