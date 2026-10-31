"""Общее разбиение и сохранение метрик двух моделей."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 12
ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / 'results' / 'metrics.csv'


def select_data(full_data, rows):
    """Уменьшаем каждый исходный split, не перемещая примеры между ними."""
    maximum = len(full_data[0])
    if not 10 <= rows <= maximum:
        raise ValueError(f'Training rows must be between 10 and {maximum}.')
    selected = []
    fingerprint = hashlib.sha256(b'proportional-v1-seed12')
    for offset in (0, 2, 4):
        X, y = full_data[offset:offset + 2]
        count = rows if offset == 0 else round(len(X) * rows / maximum)
        indices = np.arange(len(X))
        if count < len(X):
            indices, _ = train_test_split(
                indices, train_size=count, random_state=SEED, stratify=y.ravel()
            )
        if len(np.unique(y[indices])) != 2:
            raise ValueError('Each split must contain both classes; select more rows.')
        fingerprint.update(str(len(X)).encode())
        fingerprint.update(indices.astype('<i8').tobytes())
        if count == len(X):
            selected.extend((X, y))
        else:
            selected.extend((X[indices], y[indices]))
    metadata = {
        'split_version': 'proportional-v1',
        'split_id': fingerprint.hexdigest(),
        'split_seed': SEED,
        'training_rows': len(selected[0]),
        'validation_rows': len(selected[2]),
        'test_rows': len(selected[4]),
        'source_rows': sum(len(full_data[i]) for i in (0, 2, 4)),
    }
    return tuple(selected), metadata


def classification_metrics(labels, probabilities, bce):
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return dict(bce=float(bce), accuracy=accuracy_score(labels, predictions),
                precision=precision, recall=recall, f1=f1,
                roc_auc=roc_auc_score(labels, probabilities))


def save_metrics(run_name, model, split, metrics, metadata, **settings):
    row = dict(run=run_name, model=model, split=split, seed=SEED, threshold=0.5)
    row.update(metadata)
    row.update(settings)
    row.update(metrics)
    fields = ['run', 'model', 'split', 'seed', 'threshold', 'split_version', 'split_id',
              'split_seed', 'source_rows', 'training_rows', 'validation_rows', 'test_rows',
              'architecture', 'optimizer', 'batch_size', 'learning_rate', 'epochs',
              'best_epoch', 'training_seconds', 'bce', 'accuracy', 'precision',
              'recall', 'f1', 'roc_auc']
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0
    with RESULTS_PATH.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'{model} {split}:', metrics)
    print('Metrics saved:', RESULTS_PATH)
