import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import experiment, logistic_regression as lr, neural_network as nn, plots


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(1000, 28)).astype('float32')
        y = (X[:, 0] + X[:, 1] > 0).astype('float32').reshape(-1, 1)
        xt, xr, yt, yr = train_test_split(X, y, test_size=.3, stratify=y, random_state=12)
        xv, xs, yv, ys = train_test_split(xr, yr, test_size=.5, stratify=yr, random_state=12)
        cls.full = (xt, yt, xv, yv, xs, ys)

    def test_sizes_reproducibility_and_disjointness(self):
        data, meta = experiment.select_data(self.full, 350)
        again, same = experiment.select_data(self.full, 350)
        self.assertEqual([len(data[i]) for i in (0, 2, 4)], [350, 75, 75])
        self.assertEqual(meta, same)
        for a, b in zip(data, again):
            np.testing.assert_array_equal(a, b)
        sets = [set(map(tuple, data[i])) for i in (0, 2, 4)]
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        full, _ = experiment.select_data(self.full, 700)
        np.testing.assert_array_equal(full[0], self.full[0])
        with self.assertRaises(ValueError):
            experiment.select_data(self.full, 701)

    def test_save_reload_metrics_and_matching_test(self):
        data, meta = experiment.select_data(self.full, 140)
        xt, yt, xv, yv, xs, ys = data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.ExitStack() as stack:
                for obj, attr, value in [(lr, 'WEIGHTS_DIR', root / 'weights'),
                                         (nn, 'WEIGHTS_DIR', root / 'weights'),
                                         (plots, 'IMG_DIR', root / 'img'),
                                         (experiment, 'RESULTS_PATH', root / 'metrics.csv')]:
                    stack.enter_context(patch.object(obj, attr, value))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                baseline = lr.train_logistic_regression(xt, yt, xv, yv, meta)
                network = nn.train_neural_network(xt, yt, xv, yv, .001, 3, meta)
                baseline_test = lr.evaluate_logistic_regression(baseline, xs, ys, meta)
                network_test = nn.evaluate_neural_network(network, xs, ys, meta)
                scaler = StandardScaler().fit(xt)
                reference = LogisticRegression(max_iter=1000, random_state=12).fit(scaler.transform(xt), yt.ravel())
                logits = reference.decision_function(scaler.transform(xs))
                expected = np.mean(np.logaddexp(0, logits) - ys.ravel() * logits)
                self.assertAlmostEqual(baseline_test['bce'], expected, places=6)
                for path in (baseline, network):
                    with np.load(path) as saved:
                        self.assertEqual(int(saved['seed']), 12)
                        self.assertEqual(str(saved['split_id']), meta['split_id'])
                        np.testing.assert_allclose(saved['scaler_mean'], xt.mean(axis=0), atol=1e-7)
                        self.assertIn('validation_roc_auc', saved.files)
                self.assertTrue(np.isfinite(network_test['bce']))
                wrong = dict(meta, split_id='different')
                for evaluate, path in [(lr.evaluate_logistic_regression, baseline), (nn.evaluate_neural_network, network)]:
                    with self.assertRaises(ValueError):
                        evaluate(path, xs, ys, wrong)
                with (root / 'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 4)
                self.assertEqual({r['split_id'] for r in rows}, {meta['split_id']})
                self.assertEqual({r['seed'] for r in rows}, {'12'})
                self.assertEqual(len(list((root / 'img').rglob('*.png'))), 10)


if __name__ == '__main__':
    unittest.main()
