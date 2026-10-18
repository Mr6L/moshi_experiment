"""手写 BP：前向传播 → 手推梯度 → SGD 更新。

约定每一行是一个样本。第 l 层：Z_l = A_(l-1) @ W_l + b_l。
分类输出使用 Softmax，损失是平均交叉熵 + (L2 / 2) * Σ||W_l||²。
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .data import Standardizer


@dataclass(frozen=True)
class TrainConfig:
    layers: tuple = (4, 8, 3)
    activation: str = "tanh"
    learning_rate: float = 0.05
    batch_size: int = 16  # 0 代表全批量，最终重训时自动使用完整建模池。
    epochs: int = 2000
    standardize: bool = True
    initialization: str = "xavier"
    l2: float = 0.0
    dtype: str = "float32"

    def validate(self):
        if len(self.layers) < 2 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.layers):
            raise ValueError("网络层宽必须为正整数，且至少包含输入层和输出层")
        if self.activation not in ("sigmoid", "tanh", "relu"):
            raise ValueError("激活函数必须是 sigmoid、tanh 或 relu")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("学习率必须是有限正数")
        if (not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool)
                or not isinstance(self.epochs, int) or isinstance(self.epochs, bool)
                or self.batch_size < 0 or self.epochs < 1):
            raise ValueError("batch_size 必须非负，epochs 必须为正数")
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize 必须是 JSON 布尔值 true 或 false")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("L2 系数必须非负且有限")
        if self.initialization not in ("xavier", "he") or self.dtype not in ("float32", "float64"):
            raise ValueError("不支持的初始化方式或数据精度")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "layers" in values:
            values["layers"] = tuple(values["layers"])
        return cls(**values).validate()


class BPNetwork:
    def __init__(self, config, backend, seed=42):
        self.config = config.validate()
        self.backend = backend
        self.xp = backend.xp
        self.weights, self.biases = [], []
        # 在 CPU 生成初值再传往 GPU，使两个后端使用完全相同的初始权重。
        rng = np.random.default_rng(seed)
        for i, (fan_in, fan_out) in enumerate(zip(config.layers[:-1], config.layers[1:])):
            if config.initialization == "he" and i < len(config.layers) - 2:
                w = rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))
            else:
                bound = np.sqrt(6 / (fan_in + fan_out))
                w = rng.uniform(-bound, bound, size=(fan_in, fan_out))
            self.weights.append(backend.array(w, dtype=config.dtype))
            self.biases.append(backend.array(np.zeros(fan_out), dtype=config.dtype))

    def _activate(self, z):
        xp = self.xp
        if self.config.activation == "tanh":
            return xp.tanh(z)
        if self.config.activation == "relu":
            return xp.maximum(z, 0)
        # 分段形式避免 exp(很大的正数) 溢出；等价于 1/(1+exp(-z))。
        e = xp.exp(-xp.abs(z))
        return xp.where(z >= 0, 1 / (1 + e), e / (1 + e))

    def _activation_derivative(self, z, a):
        if self.config.activation == "tanh":
            return 1 - a * a
        if self.config.activation == "relu":
            return (z > 0).astype(z.dtype)
        return a * (1 - a)

    def forward(self, x):
        """保留中间结果，供反向传播使用；最后一层返回 logits。"""
        activations, preactivations = [x], []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ w + b
            preactivations.append(z)
            activations.append(self._activate(z))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        return logits, activations, preactivations

    def _distribution(self, logits):
        # 减去每行最大值不改变 Softmax，却能避免 exp(logits) 溢出。
        shifted = logits - self.xp.max(logits, axis=1, keepdims=True)
        exp_scores = self.xp.exp(shifted)
        denominator = self.xp.sum(exp_scores, axis=1, keepdims=True)
        log_probs = shifted - self.xp.log(denominator)
        return exp_scores / denominator, log_probs

    def objective(self, logits, y):
        p, log_p = self._distribution(logits)
        # 直接使用 log_softmax；不采用 log(clip(p))，以保持损失和梯度严格一致。
        data_loss = -self.xp.mean(log_p[self.xp.arange(len(y)), y])
        penalty = sum(self.xp.sum(w * w) for w in self.weights) * (0.5 * self.config.l2)
        return data_loss + penalty, data_loss, p

    def loss_and_gradients(self, x, y):
        logits, activations, zs = self.forward(x)
        loss, _, probabilities = self.objective(logits, y)

        # Softmax + 平均交叉熵：dL/dZ = (P - one_hot(y)) / N。
        delta = probabilities.copy()
        delta[self.xp.arange(len(y)), y] -= 1
        delta /= len(y)
        dw, db = [None] * len(self.weights), [None] * len(self.biases)
        for layer in reversed(range(len(self.weights))):
            # dL/dW_l = A_(l-1)^T @ delta_l + L2 * W_l。
            # dL/db_l = 对所有样本的 delta_l 求和。
            dw[layer] = activations[layer].T @ delta + self.config.l2 * self.weights[layer]
            db[layer] = self.xp.sum(delta, axis=0)
            if layer > 0:
                # 链式法则将误差传回前一隐藏层；此时所有权重仍是更新前的值。
                delta = (delta @ self.weights[layer].T) * self._activation_derivative(zs[layer - 1], activations[layer])
        return loss, dw, db

    def step(self, dw, db):
        # 所有梯度计算结束后再一起更新，避免用更新后的权重反传。
        for i in range(len(self.weights)):
            self.weights[i] -= self.config.learning_rate * dw[i]
            self.biases[i] -= self.config.learning_rate * db[i]

    def predict_proba(self, x):
        logits, _, _ = self.forward(x)
        return self._distribution(logits)[0]

    @property
    def parameter_count(self):
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


def save_model(path, model, scaler, labels, metadata=None):
    """NPZ 只保存数值数组和 JSON 字符串；加载时不启用 pickle。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if scaler.mean is None:
        raise ValueError("保存模型前必须拟合标准化器")
    arrays = {
        "metadata": np.array(json.dumps({"schema_version": 1, "config": model.config.to_dict(), "extra": metadata or {}}, ensure_ascii=False)),
        "scaler_mean": scaler.mean,
        "scaler_scale": scaler.scale,
        "labels": np.asarray(labels),
    }
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        arrays[f"W{i}"] = model.backend.cpu(w)
        arrays[f"b{i}"] = model.backend.cpu(b)
    np.savez_compressed(path, **arrays)


def load_model(path, backend):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError("不支持的模型文件版本")
        config = TrainConfig.from_dict(metadata["config"])
        model = BPNetwork(config, backend)
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            saved_w, saved_b = arrays[f"W{i}"], arrays[f"b{i}"]
            if saved_w.shape != w.shape or saved_b.shape != b.shape:
                raise ValueError(f"第 {i + 1} 层参数形状与模型结构不一致")
            if not np.isfinite(saved_w).all() or not np.isfinite(saved_b).all():
                raise ValueError("模型参数含非有限数值")
            model.weights[i] = backend.array(saved_w, dtype=config.dtype)
            model.biases[i] = backend.array(saved_b, dtype=config.dtype)
        scaler = Standardizer(config.standardize)
        scaler.mean = arrays["scaler_mean"].copy()
        scaler.scale = arrays["scaler_scale"].copy()
        labels = arrays["labels"].copy()
        if scaler.mean.shape != (config.layers[0],) or scaler.scale.shape != scaler.mean.shape:
            raise ValueError("标准化参数形状不正确")
        if not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or (scaler.scale <= 0).any():
            raise ValueError("标准化参数不正确")
        if labels.shape != (config.layers[-1],) or len(np.unique(labels)) != len(labels):
            raise ValueError("标签映射与输出层不匹配")
    return model, scaler, labels, metadata
