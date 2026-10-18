"""CPU 单元测试；由用户执行 `python -m unittest discover -s tests -v`。

测试用的是小型构造数据，不运行项目对照实验，也不评价项目测试集表现。
GPU 数值对照单独通过 `python main.py check --device gpu` 执行。
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.backend import Backend
from src.data import LABELS, Standardizer, prepare_data, subset_indices
from src.experiments import build_cases
from src.metrics import classification_metrics
from src.network import BPNetwork, TrainConfig, load_model, save_model
from src.reporting import choose_configuration
from src.training import train
from src.validation import gradient_check


ROOT = Path(__file__).resolve().parents[1]


class BPTests(unittest.TestCase):
    def test_gradients_for_all_activations_with_l2(self):
        for activation in ("sigmoid", "tanh", "relu"):
            with self.subTest(activation=activation):
                self.assertLess(gradient_check(activation)["relative_error"], 1e-5)

    def test_deep_network_directional_derivative(self):
        """多隐藏层梯度用一个随机方向的中央差分做独立核对。"""
        backend = Backend("cpu")
        config = TrainConfig(layers=(2, 4, 5, 3), dtype="float64", l2=0.01)
        model = BPNetwork(config, backend)
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(6, 2)), np.array([0, 1, 2, 0, 1, 2])
        _, dw, db = model.loss_and_gradients(x, y)
        parameters = model.weights + model.biases
        directions = [rng.normal(size=p.shape) for p in parameters]
        originals = [p.copy() for p in parameters]
        epsilon = 1e-6
        values = []
        try:
            for sign in (1, -1):
                for p, original, direction in zip(parameters, originals, directions):
                    p[...] = original + sign * epsilon * direction
                values.append(float(model.objective(model.forward(x)[0], y)[0]))
        finally:
            for p, original in zip(parameters, originals):
                p[...] = original
        analytic = sum(float(np.sum(g * d)) for g, d in zip(dw + db, directions))
        self.assertAlmostEqual((values[0] - values[1]) / (2 * epsilon), analytic, places=6)

    def test_extreme_logits_are_stable_and_gradient_matches_loss(self):
        model = BPNetwork(TrainConfig(layers=(2, 3), dtype="float64"), Backend("cpu"))
        logits = np.array([[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0]])
        loss, _, p = model.objective(logits, np.array([1, 0]))
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(float(loss), 20000.0)
        np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_training_learns_separable_synthetic_data(self):
        x = np.array([[-2., -1.], [-1., -2.], [-1., -1.], [1., 1.], [1., 2.], [2., 1.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        config = TrainConfig(layers=(2, 5, 2), dtype="float64", epochs=150, learning_rate=0.1, batch_size=0)
        result = train(config, Backend("cpu"), x, y, seed=42, checkpoints=(30,))
        self.assertLess(result.history[-1]["train_loss"], result.history[0]["train_loss"] * 0.5)
        self.assertGreaterEqual(result.history[-1]["train_accuracy"], 0.99)
        self.assertEqual(result.snapshots[30]["updates"], 30)
        self.assertEqual(result.snapshots[150]["updates"], 150)

    def test_metrics_known_confusion_matrix(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 0, 2]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3)

    def test_model_roundtrip_includes_preprocessing(self):
        backend = Backend("cpu")
        config = TrainConfig(dtype="float64")
        model = BPNetwork(config, backend)
        raw = np.array([[5., 3., 1., 0.2], [6., 2., 4., 1.5], [7., 3., 6., 2.]])
        scaler = Standardizer().fit(raw)
        expected = model.predict_proba(scaler.transform(raw))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(path, model, scaler, LABELS, {"note": "中文元数据"})
            loaded, loaded_scaler, labels, metadata = load_model(path, backend)
            np.testing.assert_allclose(loaded.predict_proba(loaded_scaler.transform(raw)), expected)
            np.testing.assert_array_equal(labels, LABELS)
            self.assertEqual(metadata["extra"]["note"], "中文元数据")

    def test_fixed_split_no_overlap_and_raw_test_is_preserved(self):
        data = prepare_data(ROOT)
        self.assertEqual((len(data.train_y), len(data.val_y), len(data.pool_y), len(data.test_y)), (59, 15, 74, 75))
        self.assertFalse(set(data.train_ids) & set(data.val_ids))
        self.assertEqual(set(data.train_ids) | set(data.val_ids), set(data.pool_ids))
        self.assertFalse(set(map(tuple, data.pool_x)) & set(map(tuple, data.test_x)))
        np.testing.assert_array_equal(data.train_ids, prepare_data(ROOT).train_ids)

    def test_nested_sampling_and_equal_size_class_controls(self):
        data = prepare_data(ROOT)
        small = subset_indices(data.train_y, 42, fraction=0.25)
        medium = subset_indices(data.train_y, 42, fraction=0.5)
        self.assertTrue(set(small) <= set(medium))
        cases, _ = build_cases(TrainConfig(), ["class_balance"], data.train_y)
        balanced = subset_indices(data.train_y, 42, class_counts=cases[1].class_counts)
        skewed = subset_indices(data.train_y, 42, class_counts=cases[2].class_counts)
        self.assertEqual(len(balanced), len(skewed))
        self.assertNotEqual(cases[1].class_counts, cases[2].class_counts)
        self.assertFalse(cases[1].eligible)

    def test_standardizer_does_not_fit_validation_and_handles_constant_feature(self):
        train_x = np.array([[1., 7.], [3., 7.]])
        scaler = Standardizer().fit(train_x)
        np.testing.assert_allclose(scaler.transform(np.array([[101., 7.]])), [[99., 0.]])
        np.testing.assert_allclose(scaler.mean, [2., 7.])

    def test_selection_excludes_data_subset_even_if_score_is_higher(self):
        baseline = {"case": "baseline", "eligible": True, "val_macro_f1_mean": 0.8,
                    "val_accuracy_mean": 0.8, "val_loss_mean": 0.4, "parameters": 67}
        data_subset = dict(baseline, case="subset", eligible=False, val_macro_f1_mean=1.0)
        self.assertEqual(choose_configuration([data_subset, baseline])["case"], "baseline")


if __name__ == "__main__":
    unittest.main()
