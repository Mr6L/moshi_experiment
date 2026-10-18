"""单因素对照实验调度。

一次只改变一个因素。训练轮数组复用同一次基线训练的检查点；
数据量、类别比例组不参与最终超参数选择，避免比较不同训练数据的配置。
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import new_directory, read_json, write_csv, write_json
from .data import LABELS, Standardizer, counts, prepare_data, subset_indices
from .network import TrainConfig, save_model
from .training import train


GROUPS = ("topology", "learning_rate", "activation", "batch_size", "standardize", "epochs", "data_size", "class_balance")
DEFAULT_SEEDS = (7, 21, 42, 87, 123)


@dataclass
class Case:
    name: str
    group: str
    config: TrainConfig
    fraction: float = 1.0
    class_counts: tuple = None
    eligible: bool = True


def build_cases(base, groups, train_y):
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"未知实验组：{sorted(unknown)}")
    cases = [Case("baseline", "baseline", base)]

    def add(name, group, **changes):
        config = replace(base, **changes).validate()
        if config != base:
            cases.append(Case(name, group, config))

    if "topology" in groups:
        for hidden in ((4,), (8,), (16,), (8, 8)):
            add("topology_" + "_".join(map(str, hidden)), "topology", layers=(4, *hidden, 3))
    if "learning_rate" in groups:
        for rate in (0.01, 0.05, 0.1):
            add(f"lr_{rate}", "learning_rate", learning_rate=rate)
    if "activation" in groups:
        for activation in ("sigmoid", "tanh", "relu"):
            add(f"activation_{activation}", "activation", activation=activation)
    if "batch_size" in groups:
        for size in (1, 16, 0):
            add(f"batch_{size or 'full'}", "batch_size", batch_size=size)
    if "standardize" in groups:
        for enabled in (False, True):
            add(f"standardize_{int(enabled)}", "standardize", standardize=enabled)
    if "data_size" in groups:
        for fraction in (0.25, 0.5, 0.75):
            cases.append(Case(f"data_{int(fraction * 100)}pct", "data_size", base, fraction=fraction, eligible=False))
    if "class_balance" in groups:
        minimum = min(np.count_nonzero(train_y == i) for i in range(3))
        unit = min(10, int(minimum) * 2 // 3)
        gap = min(unit - 1, int(minimum) - unit, max(1, unit // 2))
        if gap < 1:
            raise ValueError("训练样本过少，无法生成同总量的均衡/偏斜对照")
        cases.extend([
            Case("class_balanced", "class_balance", base, class_counts=(unit, unit, unit), eligible=False),
            Case("class_skewed", "class_balance", base, class_counts=(unit + gap, unit, unit - gap), eligible=False),
        ])
    epoch_targets = [epoch for epoch in (200, 1000) if epoch < base.epochs] if "epochs" in groups else []
    return cases, epoch_targets


def train_case(root, case, bundle, backend, seed, checkpoints=(), log_every=0):
    run_dir = root / "runs" / case.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    # 采样流与参数初始化不同，但每个实验种子都能精确重现它的训练子集。
    data_seed = int(np.random.SeedSequence([seed, 2003]).generate_state(1)[0])
    ids = subset_indices(bundle.train_y, data_seed, case.fraction, case.class_counts)
    raw_x, y = bundle.train_x[ids], bundle.train_y[ids]
    scaler = Standardizer(case.config.standardize).fit(raw_x)
    x, vx = scaler.transform(raw_x), scaler.transform(bundle.val_x)
    write_json(run_dir / "data.json", {
        "seed": seed, "data_seed": data_seed,
        "train_lines": (bundle.train_ids[ids] + 1).tolist(),
        "validation_lines": (bundle.val_ids + 1).tolist(),
        "train_class_counts": counts(y),
        "scaler_mean": scaler.mean.tolist(), "scaler_scale": scaler.scale.tolist(),
    })

    def checkpoint(epoch, model):
        # 保存时让元数据的 epochs 对应检查点，而不是整条轨迹的最大轮数。
        saved_config = model.config
        model.config = replace(saved_config, epochs=epoch)
        try:
            save_model(run_dir / f"epoch_{epoch}.npz", model, scaler, LABELS, {
                "case": case.name, "seed": seed, "epoch": epoch,
                "data_fingerprints": bundle.audit["fingerprints"],
                "purpose": "validation_experiment",
            })
        finally:
            model.config = saved_config

    targets = sorted(set(checkpoints) | {case.config.epochs})
    result = train(case.config, backend, x, y, vx, bundle.val_y, seed=seed,
                   checkpoints=targets, on_checkpoint=checkpoint, log_every=log_every)
    write_csv(run_dir / "history.csv", result.history)
    rows = {}
    for epoch in targets:
        row = {
            "case": case.name, "group": case.group, "seed": seed, "data_seed": data_seed,
            "eligible": case.eligible, "config": replace(case.config, epochs=epoch).to_dict(),
            "train_samples": len(ids), "val_samples": len(bundle.val_y),
            "train_class_counts": counts(y), "parameters": result.model.parameter_count,
            "history_path": (run_dir / "history.csv").relative_to(root).as_posix(),
            "model_path": (run_dir / f"epoch_{epoch}.npz").relative_to(root).as_posix(),
        }
        row.update(result.snapshots[epoch])
        rows[epoch] = row
    write_json(run_dir / "checkpoints.json", {str(k): v for k, v in rows.items()})
    return rows


def run_experiments(data_dir, output, backend, base, seeds=DEFAULT_SEEDS,
                    groups=GROUPS, split_seed=2026, validation_per_class=5, log_every=0):
    if base.layers[0] != 4 or base.layers[-1] != 3:
        raise ValueError("本项目输入层必须为 4、输出层必须为 3")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("种子列表必须非空、不重复且非负")
    bundle = prepare_data(data_dir, split_seed, validation_per_class)
    cases, epoch_targets = build_cases(base, groups, bundle.train_y)
    # 先确认所有条件和设备有效，再创建结果目录。
    environment = backend.info()
    root = new_directory(output)
    write_json(root / "data_audit.json", bundle.audit)
    write_json(root / "data_split.json", bundle.split)
    manifest = {
        "schema_version": 1, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment, "data_dir": str(Path(data_dir).resolve()),
        "fingerprints": bundle.audit["fingerprints"], "split_seed": split_seed,
        "validation_per_class": validation_per_class,
        "base_config": base.to_dict(), "groups": list(groups), "seeds": list(seeds),
        "expected_rows": (len(cases) + len(epoch_targets)) * len(seeds),
        "selection_rule": "仅全量训练池配置；验证 macro-F1 均值降序、准确率均值降序、交叉熵均值升序、参数量升序、case 名称升序",
        "timing_note": "累计同步训练与逐轮评估时间；包含此前检查点保存开销，不是纯 GPU 内核耗时",
    }
    write_json(root / "suite.json", manifest)
    rows = []
    try:
        for case in cases:
            for seed in seeds:
                print(f"[{case.group}] {case.name}, seed={seed}", flush=True)
                checkpoints = epoch_targets if case.name == "baseline" else ()
                snapshots = train_case(root, case, bundle, backend, seed, checkpoints, log_every)
                rows.append(snapshots[case.config.epochs])
                if case.name == "baseline":
                    for epoch in epoch_targets:
                        alias = dict(snapshots[epoch])
                        alias.update(case=f"epochs_{epoch}", group="epochs")
                        rows.append(alias)
                # 每个完成的运行立即落盘。中途失败时仍保留可检查的局部结果。
                write_json(root / "runs.json", rows)
                write_csv(root / "runs.csv", rows)
        manifest["status"] = "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "suite.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(root / "suite.json", manifest)
        raise
    return root


def load_completed_suite(root):
    root = Path(root).resolve()
    manifest = read_json(root / "suite.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("实验尚未完整结束，不能用局部结果选择最终模型")
    rows = read_json(root / "runs.json")
    if len(rows) != manifest["expected_rows"]:
        raise ValueError("实验记录数与清单不一致")
    expected_seeds = sorted(manifest["seeds"])
    for name in {row["case"] for row in rows}:
        if sorted(row["seed"] for row in rows if row["case"] == name) != expected_seeds:
            raise ValueError(f"实验 {name} 的种子记录缺失或重复")
    return manifest, rows
