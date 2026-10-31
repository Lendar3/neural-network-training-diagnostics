from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.experiment import SEED, classification_metrics, save_metrics
from src.plots import save_classification_plots, save_final_bce_plot

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / 'data' / 'weights'


def train_logistic_regression(X_train, Y_train, X_validation, Y_validation, metadata):
    start = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    baseline = LogisticRegression(max_iter=1000, random_state=SEED)
    baseline.fit(X_train, Y_train.ravel())
    seconds = time.perf_counter() - start
    train_logits = baseline.decision_function(X_train)
    logits = baseline.decision_function(X_validation)
    probabilities = expit(logits)
    train_loss = np.mean(np.logaddexp(0, train_logits) - Y_train.ravel() * train_logits)
    val_loss = np.mean(np.logaddexp(0, logits) - Y_validation.ravel() * logits)
    metrics = classification_metrics(Y_validation, probabilities, val_loss)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = f'logistic_{timestamp}_rows{len(X_train)}_seed{SEED}'
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f'{run_name}.npz'
    # Для бинарной логистической регрессии этих параметров достаточно для inference.
    saved = dict(metadata)
    saved.update(coef=baseline.coef_, intercept=baseline.intercept_,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, seed=SEED,
                 training_seconds=seconds, train_bce=train_loss)
    for name, value in metrics.items():
        saved[f'validation_{name}'] = value
    np.savez(path, **saved)
    save_metrics(run_name, 'logistic_regression', 'validation', metrics, metadata,
                 architecture='28 -> 1', optimizer='lbfgs', training_seconds=seconds)
    save_classification_plots(run_name, Y_validation.ravel(), probabilities,
                              'Validation', 'logistic_regression')
    save_final_bce_plot(run_name, train_loss, val_loss)
    print('Baseline saved:', path)
    return path


def evaluate_logistic_regression(path, X_test, Y_test, metadata):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['split_id']) != metadata['split_id']:
            raise ValueError('Baseline and selected data splits do not match.')
        scaled = (X_test - saved['scaler_mean']) / saved['scaler_scale']
        logits = (scaled @ saved['coef'].T + saved['intercept']).ravel()
    probabilities = expit(logits)
    bce = np.mean(np.logaddexp(0, logits) - Y_test.ravel() * logits)
    metrics = classification_metrics(Y_test, probabilities, bce)
    save_metrics(Path(path).stem, 'logistic_regression', 'test', metrics, metadata)
    save_classification_plots(Path(path).stem, Y_test.ravel(), probabilities,
                              'Test', 'logistic_regression')
    return metrics
