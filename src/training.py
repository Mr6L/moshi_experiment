"""训练过程没有自动求导，也没有现成优化器。

各实验运行完整的固定轮数，不早停；轮数实验从同一轨迹提取检查点。
这样结构/参数对照的训练预算明确，最终选择也不会依赖测试集。
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .metrics import classification_metrics
from .network import BPNetwork


def evaluate(model, x, y):
    logits, _, _ = model.forward(x)
    loss, data_loss, probabilities = model.objective(logits, y)
    predictions = model.backend.cpu(model.xp.argmax(probabilities, axis=1))
    metrics = classification_metrics(model.backend.cpu(y), predictions, model.config.layers[-1])
    metrics.update(loss=model.backend.scalar(loss), data_loss=model.backend.scalar(data_loss))
    if not np.isfinite([metrics["loss"], metrics["data_loss"]]).all():
        raise RuntimeError("损失出现 NaN/Inf，请检查学习率、数据尺度或梯度实现")
    return metrics


@dataclass
class TrainingResult:
    model: BPNetwork
    history: list
    snapshots: dict


def train(config, backend, train_x, train_y, val_x=None, val_y=None, seed=42,
          checkpoints=(), on_checkpoint=None, log_every=0):
    config.validate()
    if seed < 0 or log_every < 0:
        raise ValueError("随机种子和打印间隔必须非负")
    if (val_x is None) != (val_y is None):
        raise ValueError("验证特征和标签必须同时提供")
    if train_x.ndim != 2 or len(train_x) == 0 or train_x.shape[1] != config.layers[0] or train_y.ndim != 1 or len(train_y) != len(train_x):
        raise ValueError("训练数据形状与网络输入不一致")
    if val_x is not None:
        if val_x.ndim != 2 or len(val_x) == 0 or val_x.shape[1] != config.layers[0] or val_y.ndim != 1 or len(val_y) != len(val_x):
            raise ValueError("验证数据形状与网络输入不一致")
        if np.any(val_y < 0) or np.any(val_y >= config.layers[-1]) or not np.equal(val_y, np.floor(val_y)).all():
            raise ValueError("验证标签必须是输出层范围内的整数")
    if not np.isfinite(train_x).all() or (val_x is not None and not np.isfinite(val_x).all()):
        raise ValueError("输入特征必须为有限数值")
    if np.any(train_y < 0) or np.any(train_y >= config.layers[-1]) or not np.equal(train_y, np.floor(train_y)).all():
        raise ValueError("训练标签超出输出层范围")
    model = BPNetwork(config, backend, seed)
    x = backend.array(train_x, dtype=config.dtype)
    y = backend.array(train_y, dtype=np.int64)
    vx = backend.array(val_x, dtype=config.dtype) if val_x is not None else None
    vy = backend.array(val_y, dtype=np.int64) if val_y is not None else None
    targets = set(checkpoints) | {config.epochs}
    if any(epoch < 1 or epoch > config.epochs for epoch in targets):
        raise ValueError("检查点轮数必须在 1 到总训练轮数之间")
    batch_size = len(x) if config.batch_size == 0 else min(config.batch_size, len(x))
    # 打乱顺序的随机流与初始化独立，保证相同 seed 的不同拓扑共享样本顺序。
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1009]))
    history, snapshots = [], {}
    updates = 0
    backend.synchronize()
    start = perf_counter()
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            order = backend.array(rng.permutation(len(x)), dtype=np.int64)
            for offset in range(0, len(x), batch_size):
                ids = order[offset:offset + batch_size]
                _, dw, db = model.loss_and_gradients(x[ids], y[ids])
                model.step(dw, db)
                updates += 1

        # 包括第 0 轮（初始模型）在内逐轮记录；loss 是数据交叉熵，objective 含 L2。
        train_metrics = evaluate(model, x, y)
        val_metrics = evaluate(model, vx, vy) if vx is not None else None
        backend.synchronize()
        row = {
            "epoch": epoch, "updates": updates, "elapsed_seconds": perf_counter() - start,
            "train_loss": train_metrics["data_loss"], "train_objective": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"], "train_macro_f1": train_metrics["macro_f1"],
        }
        if val_metrics is not None:
            row.update(val_loss=val_metrics["data_loss"], val_accuracy=val_metrics["accuracy"], val_macro_f1=val_metrics["macro_f1"])
            for item in val_metrics["per_class"]:
                row[f"val_recall_{item['label']}"] = item["recall"]
        history.append(row)
        if epoch in targets:
            snapshots[epoch] = dict(row)
            if on_checkpoint is not None:
                on_checkpoint(epoch, model)
        if log_every and (epoch == 0 or epoch % log_every == 0 or epoch == config.epochs):
            validation = f", val_acc={row['val_accuracy']:.3f}" if val_metrics is not None else ""
            print(f"  epoch={epoch:4d}, train_loss={row['train_loss']:.5f}, train_acc={row['train_accuracy']:.3f}{validation}", flush=True)
    return TrainingResult(model, history, snapshots)
