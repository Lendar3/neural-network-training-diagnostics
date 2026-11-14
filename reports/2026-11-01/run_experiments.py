"""Запустить только validation-сравнение; финальный test — отдельным шагом."""
from pathlib import Path
import sys
import json
import hashlib
import platform
import importlib.metadata
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import main
from src.neural_network import train_neural_network
from src.logistic_regression import train_logistic_regression

OUT = Path(__file__).resolve().parent

def validation_summary(path, name):
    with np.load(path, allow_pickle=False) as saved:
        row = {'model': name, 'weights': str(path.relative_to(ROOT))}
        for key in ['training_rows', 'validation_rows', 'seed', 'learning_rate', 'epochs',
                    'best_epoch', 'training_seconds', 'validation_bce', 'validation_accuracy',
                    'validation_precision', 'validation_recall', 'validation_f1', 'validation_roc_auc']:
            if key in saved:
                row[key] = saved[key].item()
        if 'train_losses' in saved:
            train = saved['train_losses']; val = saved['validation_losses']
            best = int(saved['best_epoch']) - 1
            row.update(train_bce_at_best=float(train[best]), train_bce_last=float(train[-1]),
                       validation_bce_last=float(val[-1]))
        else:
            row['train_bce_at_best'] = float(saved['train_bce'])
    return row

if __name__ == '__main__':
    print('Loading original data and reproducing existing split...', flush=True)
    full_data = main.load_data()
    selected = main.select_data(full_data, 100000)
    small = main.select_data(full_data, 300)
    evidence = {'date': '2026-11-01', 'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
                'python': platform.python_version(), 'platform': platform.platform(),
                'versions': {name: importlib.metadata.version(name) for name in ['torch','numpy','scikit-learn','scipy','matplotlib']},
                'original_sizes': [len(full_data[i]) for i in [0,2,4]], 'selected_data': {}, 'runs': []}
    for title, data in [('main', selected), ('small', small)]:
        stats = {}
        for offset, split in [(0,'train'), (2,'validation'), (4,'test')]:
            X, y = data[offset:offset+2]
            assert np.isfinite(X).all() and np.isfinite(y).all()
            assert set(np.unique(y)) == {0,1}
            stats[split] = {'rows': len(X), 'class_0': int((y==0).sum()), 'class_1': int((y==1).sum())}
        evidence['selected_data'][title] = stats
    # Save exact selected arrays once: final evaluation reuses these arrays without loading 2.4 GB again.
    np.savez(OUT/'selected_data.npz', X_train=selected[0], Y_train=selected[1],
             X_validation=selected[2], Y_validation=selected[3], X_test=selected[4], Y_test=selected[5])
    small_net = ROOT/'data/weights/bce_slice_20261101_062930_346634_rows300_lr0.001_epochs1000_seed12.npz'
    small_base = ROOT/'data/weights/logistic_slice_20261101_062907_037975_rows300_seed12.npz'
    evidence['small_runs'] = [validation_summary(small_base,'logistic_regression'),validation_summary(small_net,'neural_network')]
    for p in [small_net, small_base]:
        with np.load(p) as saved:
            np.testing.assert_allclose(saved['scaler_mean'], small[0].mean(axis=0,dtype=np.float64), atol=1e-10)
    for file in ['main.py','src/neural_network.py','src/logistic_regression.py','src/plots.py']:
        evidence.setdefault('code_sha256',{})[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    data_path = ROOT/'data/processed/higgs.npz'
    evidence['dataset'] = {'path':str(data_path),'size_bytes':data_path.stat().st_size}
    del full_data, small
    def persist():
        (OUT/'experiment_results.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    persist()
    print('Training baseline on 100000 rows...', flush=True)
    path = train_logistic_regression(*selected[:4])
    main.save_result(path,'logistic_regression','validation')
    evidence['runs'].append(validation_summary(path,'logistic_regression')); persist()
    for rate in [0.0001,0.001,0.01]:
        print('Training network, LR =',rate,flush=True)
        path = train_neural_network(*selected[:4],rate,300)
        main.save_result(path,'neural_network','validation')
        evidence['runs'].append(validation_summary(path,'neural_network')); persist()
    print('VALIDATION EXPERIMENTS COMPLETE',flush=True)
