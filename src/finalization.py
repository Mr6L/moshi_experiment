"""先根据验证集选方案并重训，之后才计算测试指标；不以测试分数选模型。"""

from pathlib import Path

import numpy as np

from .artifacts import new_directory, read_json, write_csv, write_json
from .data import LABELS, Standardizer, prepare_data
from .experiments import load_completed_suite
from .metrics import classification_metrics
from .network import TrainConfig, load_model, save_model
from .reporting import choose_configuration, plot_confusion, plot_learning_curve, read_history, summarize
from .training import train


def finalize(suite_dir, output, backend, seed=42, data_dir=None, log_every=100):
    if seed < 0:
        raise ValueError("最终重训种子必须非负")
    manifest, rows = load_completed_suite(suite_dir)
    winner = choose_configuration(summarize(rows))
    config = TrainConfig.from_dict(winner["config"])
    bundle = prepare_data(data_dir or manifest["data_dir"], manifest["split_seed"], manifest["validation_per_class"])
    if bundle.audit["fingerprints"] != manifest["fingerprints"]:
        raise ValueError("原始数据与对照实验时不同，不能沿用旧实验选择结果")
    environment = backend.info()
    root = new_directory(output)
    selection = {
        "status": "training", "source_suite": str(Path(suite_dir).resolve()),
        "selected_case": winner["case"], "config": config.to_dict(),
        "selection_rule": manifest["selection_rule"],
        "validation_statistics": winner, "final_seed": seed,
        "environment": environment, "fingerprints": manifest["fingerprints"],
        "train_samples": len(bundle.pool_y), "test_samples": len(bundle.test_y),
        "note": "使用完整建模池重训；标准化重新拟合；种子在测试前固定；训练不使用测试标签",
    }
    # 在读取测试表现之前把配置和种子写入文件，使选择过程可以追溯。
    write_json(root / "selection.json", selection)
    scaler = Standardizer(config.standardize).fit(bundle.pool_x)
    result = train(config, backend, scaler.transform(bundle.pool_x), bundle.pool_y,
                   seed=seed, log_every=log_every)
    save_model(root / "model.npz", result.model, scaler, LABELS, {
        "purpose": "final_model", "seed": seed, "source_suite": selection["source_suite"],
        "case": winner["case"], "data_fingerprints": manifest["fingerprints"],
    })
    write_csv(root / "history.csv", result.history)

    # 测试集前向只执行一次，所有测试指标和错误记录复用这一次的概率。
    x = backend.array(scaler.transform(bundle.test_x), dtype=config.dtype)
    y = backend.array(bundle.test_y, dtype=np.int64)
    logits = result.model.forward(x)[0]
    _, data_loss, probabilities = result.model.objective(logits, y)
    p = backend.cpu(probabilities)
    if not np.isfinite(p).all():
        raise RuntimeError("测试概率出现非有限值")
    predicted = p.argmax(axis=1)
    metrics = classification_metrics(bundle.test_y, predicted)
    metrics.update(cross_entropy=backend.scalar(data_loss), samples=len(bundle.test_y))
    write_json(root / "test_metrics.json", metrics)
    predictions = []
    for i, (features, truth, guess, probability) in enumerate(zip(bundle.test_x, bundle.test_y, predicted, p)):
        row = {"test_line": i + 1, **{f"feature_{j + 1}": float(v) for j, v in enumerate(features)},
               "true_label": int(LABELS[truth]), "predicted_label": int(LABELS[guess]), "correct": bool(truth == guess)}
        row.update({f"probability_{label}": float(v) for label, v in zip(LABELS, probability)})
        predictions.append(row)
    write_csv(root / "test_predictions.csv", predictions)
    wrong = [row for row in predictions if not row["correct"]]
    # 即使零错误，也输出带表头的文件，便于后续处理。
    if wrong:
        write_csv(root / "misclassified.csv", wrong)
    else:
        import csv
        with (root / "misclassified.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(predictions[0])).writeheader()
    selection["status"] = "evaluated"
    write_json(root / "selection.json", selection)
    generate_final_report(root)
    return root, metrics


def generate_final_report(root):
    """从已保存的测试记录重建图表；不重训，也不重新预测测试集。"""
    root = Path(root).resolve()
    selection = read_json(root / "selection.json")
    if selection.get("status") not in ("evaluated", "complete"):
        raise ValueError("最终测试尚未完成，不能生成最终报告")
    metrics = read_json(root / "test_metrics.json")
    history = read_history(root, {"history_path": "history.csv", "epoch": selection["config"]["epochs"]})
    plot_confusion(metrics["confusion_matrix"], root / "confusion_matrix.png")
    plot_learning_curve(history, root / "learning_curves.png")
    lines = [
        "# BP 最终分类评估", "",
        f"配置来自 `{selection['selected_case']}`，按验证集多种子统计选择，最终训练种子为 `{selection['final_seed']}`。",
        f"完整建模池 {selection['train_samples']} 条，测试集 {selection['test_samples']} 条；已在训练数据中排除与测试集相同的记录。", "",
        f"- 测试准确率：{metrics['accuracy']:.4%}。",
        f"- 宏平均 F1：{metrics['macro_f1']:.6f}。",
        f"- 交叉熵：{metrics['cross_entropy']:.6f}。",
        f"- 错误数：{metrics['errors']} / {metrics['samples']}。", "",
        "| 类别 | Precision | Recall | F1 | 样本数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics["per_class"]:
        lines.append(f"| {item['label']} | {item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} | {item['support']} |")
    lines.extend(["", "![混淆矩阵](confusion_matrix.png)", "", "![重训曲线](learning_curves.png)", "",
                  "混淆矩阵行是真实标签，列是预测标签；逐条预测和误分类见本目录 CSV。",
                  "本结果是固定测试集的一次评估，不能反复依据测试分数修改配置或挑选随机种子。", ""])
    (root / "最终评估报告.md").write_text("\n".join(lines), encoding="utf-8")
    selection["status"] = "complete"
    write_json(root / "selection.json", selection)
    return root / "最终评估报告.md"


def predict(model_path, backend, features):
    model, scaler, labels, _ = load_model(model_path, backend)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != model.config.layers[0] or not np.isfinite(x).all():
        raise ValueError(f"预测输入必须为非空、有限数值的 N×{model.config.layers[0]} 特征矩阵")
    probabilities = backend.cpu(model.predict_proba(backend.array(scaler.transform(x), dtype=model.config.dtype)))
    if not np.isfinite(probabilities).all():
        raise RuntimeError("预测概率出现非有限数值")
    result = []
    for i, (values, p) in enumerate(zip(x, probabilities)):
        result.append({
            "row": i + 1, **{f"feature_{j + 1}": float(v) for j, v in enumerate(values)},
            "predicted_label": int(labels[p.argmax()]),
            **{f"probability_{label}": float(v) for label, v in zip(labels, p)},
        })
    return result
