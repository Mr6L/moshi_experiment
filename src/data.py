"""数据审计、固定分层划分、训练子集采样和标准化。

索引均为原始 trainData.txt 的 0 起始行索引；输出到 JSON 时还提供 1 起始行号。
测试集只用于预先约定的重复记录检查和最终评估，不参与参数选择。
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


LABELS = np.array([1, 2, 3], dtype=np.int64)
FILES = ("trainData.txt", "testData.txt", "totalData.txt")


def load_table(path):
    table = np.loadtxt(path, dtype=np.float64, encoding="utf-8-sig", ndmin=2)
    if table.shape[0] == 0 or table.shape[1] != 5:
        raise ValueError(f"{path} 必须是非空的 5 列数据：4 个特征 + 1 个标签")
    if not np.isfinite(table).all():
        raise ValueError(f"{path} 含 NaN 或无穷大")
    if not np.isin(table[:, -1], LABELS).all():
        raise ValueError(f"{path} 的类别标签必须为 1、2、3")
    return table


def counts(y):
    """输入内部标签 0/1/2，输出供人阅读的原始标签 1/2/3。"""
    return {str(label): int(np.count_nonzero(y == i)) for i, label in enumerate(LABELS)}


@dataclass
class DataBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    train_ids: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_ids: np.ndarray
    pool_x: np.ndarray
    pool_y: np.ndarray
    pool_ids: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    audit: dict
    split: dict


def prepare_data(root, split_seed=2026, validation_per_class=5):
    root = Path(root).resolve()
    tables = {name: load_table(root / name) for name in FILES}
    train, test, total = (tables[name] for name in FILES)
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}
    same_total = Counter(map(tuple, total)) == Counter(map(tuple, train)) + Counter(map(tuple, test))
    if not same_total:
        raise ValueError("totalData.txt 与 trainData.txt + testData.txt 的记录及出现次数不一致")

    # 同一特征对应不同标签属于数据矛盾，不能用重复过滤掩盖。
    feature_labels = {}
    for row in np.concatenate([train, test]):
        key = tuple(row[:-1])
        if key in feature_labels and feature_labels[key] != row[-1]:
            raise ValueError(f"相同特征存在冲突标签：{key}")
        feature_labels[key] = row[-1]

    test_rows = set(map(tuple, test))
    excluded = [i for i, row in enumerate(train) if tuple(row) in test_rows]
    pool_ids = np.array([i for i in range(len(train)) if i not in excluded], dtype=np.int64)
    pool = train[pool_ids]
    pool_y = pool[:, -1].astype(np.int64) - 1
    if validation_per_class < 1:
        raise ValueError("每类验证样本数必须至少为 1")

    # 每个类别独立抽取验证集；避免从按类别排序的数据中直接切一段。
    rng = np.random.default_rng(split_seed)
    val_local = []
    for label in range(3):
        ids = np.flatnonzero(pool_y == label)
        if len(ids) <= validation_per_class:
            raise ValueError(f"类别 {label + 1} 样本不足以同时构造训练集和验证集")
        val_local.extend(rng.permutation(ids)[:validation_per_class].tolist())
    val_local = np.array(sorted(val_local), dtype=np.int64)
    train_local = np.setdiff1d(np.arange(len(pool)), val_local)
    train_ids, val_ids = pool_ids[train_local], pool_ids[val_local]

    audit = {
        "data_dir": str(root),
        "fingerprints": fingerprints,
        "total_matches_train_plus_test": same_total,
        "files": {
            name: {
                "rows": len(table),
                "class_counts": counts(table[:, -1].astype(np.int64) - 1),
                "duplicate_rows": len(table) - len(set(map(tuple, table))),
                "feature_min": table[:, :-1].min(axis=0).tolist(),
                "feature_max": table[:, :-1].max(axis=0).tolist(),
            }
            for name, table in tables.items()
        },
        "removed_train_records": [{"line": i + 1, "values": train[i].tolist()} for i in excluded],
        "duplicate_policy": "保留测试集；仅在内存中排除训练集与测试集完全相同的记录",
    }
    split = {
        "split_seed": split_seed,
        "validation_per_class": validation_per_class,
        "train_indices": train_ids.tolist(),
        "validation_indices": val_ids.tolist(),
        "modeling_pool_indices": pool_ids.tolist(),
        "excluded_train_lines": [i + 1 for i in excluded],
        "train_lines": (train_ids + 1).tolist(),
        "validation_lines": (val_ids + 1).tolist(),
        "train_class_counts": counts(pool_y[train_local]),
        "validation_class_counts": counts(pool_y[val_local]),
        "modeling_pool_class_counts": counts(pool_y),
        "test_class_counts": counts(test[:, -1].astype(np.int64) - 1),
    }
    return DataBundle(
        pool[train_local, :-1], pool_y[train_local], train_ids,
        pool[val_local, :-1], pool_y[val_local], val_ids,
        pool[:, :-1], pool_y, pool_ids,
        test[:, :-1], test[:, -1].astype(np.int64) - 1,
        audit, split,
    )


class Standardizer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mean = None
        self.scale = None

    def fit(self, x):
        # 每次仅传入实际训练子集；验证集/测试集绝不能用来拟合均值和方差。
        self.mean = x.mean(axis=0) if self.enabled else np.zeros(x.shape[1])
        scale = x.std(axis=0) if self.enabled else np.ones(x.shape[1])
        self.scale = np.where(scale < 1e-12, 1.0, scale)
        return self

    def transform(self, x):
        if self.mean is None:
            raise ValueError("标准化器还没有 fit")
        return (x - self.mean) / self.scale


def subset_indices(y, seed, fraction=1.0, class_counts=None):
    """每类使用固定随机排列的前缀，使同种子的小子集嵌套于大子集。"""
    if not 0 < fraction <= 1:
        raise ValueError("训练比例必须在 (0, 1] 内")
    if class_counts is not None and (len(class_counts) != 3 or any(n < 1 for n in class_counts)):
        raise ValueError("class_counts 必须为三个正整数")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(3):
        candidates = rng.permutation(np.flatnonzero(y == label))
        n = class_counts[label] if class_counts is not None else max(1, int(np.floor(len(candidates) * fraction + 0.5)))
        if n > len(candidates):
            raise ValueError(f"类别 {label + 1} 请求 {n} 条，但训练池只有 {len(candidates)} 条")
        chosen.extend(candidates[:n].tolist())
    return np.array(sorted(chosen), dtype=np.int64)
