"""由用户主动运行的算法检查；导入本模块不会执行测试或训练。"""

import numpy as np

from .backend import Backend
from .network import BPNetwork, TrainConfig


def gradient_check(activation="tanh", l2=0.03):
    """中央差分：(L(theta+h)-L(theta-h))/(2h)，逐个参数核对手写 BP。"""
    backend = Backend("cpu")
    config = TrainConfig(layers=(2, 4, 3), activation=activation, dtype="float64", l2=l2)
    model = BPNetwork(config, backend, seed=42)
    x = np.random.default_rng(103).normal(size=(5, 2))
    y = np.array([0, 1, 2, 0, 1])
    _, dw, db = model.loss_and_gradients(x, y)
    analytic, numeric = [], []
    step = 1e-6
    for parameter, gradient in zip(model.weights + model.biases, dw + db):
        for index in np.ndindex(parameter.shape):
            original = parameter[index].copy()
            try:
                parameter[index] = original + step
                plus = model.objective(model.forward(x)[0], y)[0]
                parameter[index] = original - step
                minus = model.objective(model.forward(x)[0], y)[0]
            finally:
                parameter[index] = original
            analytic.append(float(gradient[index]))
            numeric.append(float((plus - minus) / (2 * step)))
    analytic, numeric = np.array(analytic), np.array(numeric)
    relative_error = np.linalg.norm(analytic - numeric) / max(1e-12, np.linalg.norm(analytic) + np.linalg.norm(numeric))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
    if relative_error > 1e-5:
        raise AssertionError(f"{activation} 梯度相对误差超限：{relative_error}")
    return {"activation": activation, "l2": l2, "parameters_checked": len(analytic), "relative_error": float(relative_error)}


def backend_check(backend):
    """核对基础运算、前向、损失、梯度和一次 SGD 更新。"""
    x = np.random.default_rng(42).normal(size=(6, 4))
    y = np.array([0, 1, 2, 0, 1, 2])
    gx = backend.array(x, dtype="float64")
    gy = backend.array(y, dtype=np.int64)
    np.testing.assert_allclose(backend.cpu(gx @ gx.T), x @ x.T, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(backend.cpu(backend.xp.exp(gx)), np.exp(x), rtol=1e-7, atol=1e-9)
    checked = []
    for activation in ("sigmoid", "tanh", "relu"):
        config = TrainConfig(layers=(4, 5, 4, 3), activation=activation, dtype="float64", l2=0.02)
        cpu = BPNetwork(config, Backend("cpu"), seed=21)
        other = BPNetwork(config, backend, seed=21)
        np.testing.assert_allclose(backend.cpu(other.forward(gx)[0]), cpu.forward(x)[0], rtol=1e-7, atol=1e-9)
        cpu_loss, cpu_dw, cpu_db = cpu.loss_and_gradients(x, y)
        loss, dw, db = other.loss_and_gradients(gx, gy)
        np.testing.assert_allclose(backend.scalar(loss), float(cpu_loss), rtol=1e-7, atol=1e-9)
        for expected, actual in zip(cpu_dw + cpu_db, dw + db):
            np.testing.assert_allclose(backend.cpu(actual), expected, rtol=1e-7, atol=1e-9)
        cpu.step(cpu_dw, cpu_db)
        other.step(dw, db)
        for expected, actual in zip(cpu.weights + cpu.biases, other.weights + other.biases):
            np.testing.assert_allclose(backend.cpu(actual), expected, rtol=1e-7, atol=1e-9)
        checked.append(activation)
    backend.synchronize()
    return {"activations_checked": checked, "checks": ["matrix_multiply", "exp", "forward", "loss", "gradients", "sgd_update"], "passed": True}


def all_checks(backend):
    return {
        "environment": backend.info(),
        "finite_difference": [gradient_check(name) for name in ("sigmoid", "tanh", "relu")],
        "backend_comparison": backend_check(backend),
    }
