"""把真实运行记录汇总成 CSV、图表及中文 Markdown；不预填实验结论。"""

import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import write_csv, write_json
from .experiments import load_completed_suite


MEASUREMENTS = ("train_loss", "train_accuracy", "train_macro_f1", "val_loss", "val_accuracy", "val_macro_f1", "val_recall_1", "val_recall_2", "val_recall_3", "elapsed_seconds", "updates")
GROUP_NAMES = {
    "topology": "拓扑结构", "learning_rate": "学习率", "activation": "激活函数",
    "batch_size": "批大小", "standardize": "标准化", "epochs": "训练轮数",
    "data_size": "训练样本量", "class_balance": "训练类别比例",
}


def summarize(rows):
    summary = []
    for name in sorted({row["case"] for row in rows}):
        selected = [row for row in rows if row["case"] == name]
        first = selected[0]
        item = {key: first[key] for key in ("case", "group", "eligible", "config", "parameters", "train_samples", "val_samples", "train_class_counts")}
        item["repeats"] = len(selected)
        for key in MEASUREMENTS:
            values = np.array([row[key] for row in selected], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} 的 {key} 出现非有限值")
            item[f"{key}_mean"] = float(values.mean())
            # 样本标准差；只跑一个种子时设为 0，报告会明确重复数。
            item[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def choose_configuration(summary):
    candidates = [item for item in summary if item["eligible"]]
    if not candidates:
        raise ValueError("没有可参与最终选择的完整训练数据配置")
    return min(candidates, key=lambda item: (
        -item["val_macro_f1_mean"], -item["val_accuracy_mean"],
        item["val_loss_mean"], item["parameters"], item["case"],
    ))


def plotting():
    # 不弹出窗口，可在终端和无图形界面的环境中生成图片。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def read_history(root, row):
    with (root / row["history_path"]).open(encoding="utf-8-sig", newline="") as handle:
        return [{key: float(value) for key, value in record.items()} for record in csv.DictReader(handle) if int(record["epoch"]) <= row["epoch"]]


def plot_learning_curve(history, path):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for prefix in ("train", "val"):
        if f"{prefix}_loss" not in history[0]:
            continue
        axes[0].plot([r["epoch"] for r in history], [r[f"{prefix}_loss"] for r in history], label=prefix)
        axes[1].plot([r["epoch"] for r in history], [r[f"{prefix}_accuracy"] for r in history], label=prefix)
    for ax, title in zip(axes, ("Cross-entropy", "Accuracy")):
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.25)
    axes[1].set_ylim(0, 1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(matrix, path):
    plt = plotting()
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(xticks=range(len(matrix)), yticks=range(len(matrix)),
           xticklabels=range(1, len(matrix) + 1), yticklabels=range(1, len(matrix) + 1),
           xlabel="Predicted class", ylabel="True class", title="Final test confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def group_members(summary, group):
    # 类别比例组已经控制了两组相同样本总量；不混入 59 条的基线作同量比较。
    members = [item for item in summary if item["group"] == group]
    if group != "class_balance":
        members = [item for item in summary if item["case"] == "baseline"] + members
    return members


def plot_group(root, group, members, rows, figure_dir):
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    positions = np.arange(len(members))
    for offset, metric, label in ((-0.18, "val_accuracy", "Accuracy"), (0.18, "val_macro_f1", "Macro F1")):
        axes[0].bar(positions + offset, [i[f"{metric}_mean"] for i in members], width=0.36,
                    yerr=[i[f"{metric}_std"] for i in members], capsize=3, label=label)
    axes[0].set_xticks(positions, [i["case"] for i in members], rotation=35, ha="right")
    axes[0].set(ylim=(0, 1.08), title="Validation metrics (mean +/- sample SD)")
    axes[0].legend()
    for item in members:
        histories = [read_history(root, row) for row in rows if row["case"] == item["case"]]
        epochs = np.array([r["epoch"] for r in histories[0]])
        values = np.array([[r["val_loss"] for r in history] for history in histories])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        axes[1].plot(epochs, mean, label=item["case"])
        axes[1].fill_between(epochs, mean - std, mean + std, alpha=0.12)
    axes[1].set(xlabel="Epoch", ylabel="Validation cross-entropy", title="Convergence")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{group}.png", dpi=160)
    plt.close(fig)


def generate_report(root):
    root = Path(root).resolve()
    manifest, rows = load_completed_suite(root)
    summary = summarize(rows)
    winner = choose_configuration(summary)
    write_json(root / "summary.json", summary)
    write_csv(root / "summary.csv", summary)
    report_dir = root / "reports"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    environment = manifest["environment"]
    lines = [
        "# BP 神经网络对照实验报告", "",
        "本报告根据本目录真实运行记录自动生成。测试集尚未参与本报告的模型选择。", "",
        "## 环境与方法", "",
        f"- 后端：`{environment['device']}`；设备：`{environment.get('gpu', 'CPU')}`。",
        f"- NumPy：`{environment['numpy']}`；CuPy：`{environment.get('cupy', '未使用')}`。",
        f"- Python：`{environment['python'].split()[0]}`；CUDA runtime：`{environment.get('cuda_runtime', '未使用')}`；CUDA driver API：`{environment.get('cuda_driver', '未使用')}`。",
        f"- 固定划分种子：`{manifest['split_seed']}`；实验种子：`{manifest['seeds']}`。",
        "- 网络、交叉熵梯度和 SGD 全部手写；标准化仅拟合实际训练子集。",
        "- 每组重复运行，报告均值及样本标准差；重复数为 1 时标准差记为 0，不能据此判断稳定性。",
        "- 当前实验使用同一个固定验证集；多种子波动主要反映初始化/打乱顺序，数据子集组还包含采样波动，不是总体泛化置信区间。",
        "- 数据过滤和划分详见 `../data_audit.json`、`../data_split.json`；所有原始文件保持不变。",
        f"- 计时口径：{manifest['timing_note']}。", "",
    ]
    baseline = next(item for item in summary if item["case"] == "baseline")
    baseline_rows = [row for row in rows if row["case"] == "baseline"]
    plot_learning_curve(read_history(root, baseline_rows[0]), root / "learning_curves.png")
    lines.extend([
        "## 基线结果", "",
        f"网络 `{baseline['config']['layers']}`，{baseline['train_samples']} 条训练、{baseline['val_samples']} 条验证，重复 {baseline['repeats']} 次。",
        f"训练准确率：{baseline['train_accuracy_mean']:.4f} ± {baseline['train_accuracy_std']:.4f}；"
        f"验证准确率：{baseline['val_accuracy_mean']:.4f} ± {baseline['val_accuracy_std']:.4f}；"
        f"验证宏 F1：{baseline['val_macro_f1_mean']:.4f} ± {baseline['val_macro_f1_std']:.4f}。",
        "", "![首个基线种子的训练与验证曲线](../learning_curves.png)", "",
    ])
    for group in manifest["groups"]:
        members = group_members(summary, group)
        if not members:
            continue
        plot_group(root, group, members, rows, figure_dir)
        lines.extend([
            f"## {GROUP_NAMES[group]}", "",
            "| 条件 | 训练数/类别数 | 参数量 | 轮数 | 验证准确率 | 验证宏 F1 | 验证交叉熵 | 更新次数 | 累计秒数 |",
            "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
        ])
        for item in members:
            lines.append(
                f"| {item['case']} | {item['train_samples']} / {list(item['train_class_counts'].values())} | "
                f"{item['parameters']} | {item['config']['epochs']} | "
                f"{item['val_accuracy_mean']:.4f} ± {item['val_accuracy_std']:.4f} | "
                f"{item['val_macro_f1_mean']:.4f} ± {item['val_macro_f1_std']:.4f} | "
                f"{item['val_loss_mean']:.4f} ± {item['val_loss_std']:.4f} | "
                f"{item['updates_mean']:.0f} | {item['elapsed_seconds_mean']:.2f} |"
            )
        if group in ("class_balance", "data_size"):
            lines.extend(["", "| 条件 | 类别 1 召回率 | 类别 2 召回率 | 类别 3 召回率 |",
                          "| --- | --- | --- | --- |"])
            for item in members:
                values = [f"{item[f'val_recall_{label}_mean']:.4f} ± {item[f'val_recall_{label}_std']:.4f}" for label in (1, 2, 3)]
                lines.append(f"| {item['case']} | " + " | ".join(values) + " |")
        best = min(members, key=lambda item: (-item["val_macro_f1_mean"], -item["val_accuracy_mean"], item["val_loss_mean"]))
        lines.extend(["", f"本组验证宏 F1 均值最高的条件之一为 `{best['case']}`（并列时按准确率、交叉熵排序）；差异大小应结合标准差和学习曲线判断。",
                      "", f"![{GROUP_NAMES[group]}](figures/{group}.png)", ""])
    lines.extend([
        "## 待最终评估的配置", "",
        f"按预先固定的验证排序规则，选择 `{winner['case']}`；训练数据子集组不参与此排序。",
        "", "```json", json.dumps(winner["config"], indent=2, ensure_ascii=False), "```", "",
        "使用 finalize 命令在完整建模池上重训，再读取测试标签计算最终指标。最终随机种子在评估前指定，不能依据测试表现挑种子。", "",
        "## 实验分析提示", "",
        "1. 比较训练/验证曲线：两者都较差可能欠拟合；训练持续变好而验证变差可能过拟合。",
        "2. 比较宽度与层数：结合参数量、收敛速度和验证表现解释，不预设更深更好。",
        "3. 比较学习率和激活函数：观察震荡、收敛缓慢或饱和；激活对照固定初始化策略。",
        "4. 批大小同时影响每轮更新次数，需结合更新数和耗时分析。",
        "5. 数据量组使用分层嵌套子集；类别比例组使用相同总样本量，关注各类召回率。",
        "6. 验证集很小且重复用于方案选择；微小领先不能直接推断为普遍优越。", "",
    ])
    (report_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    return report_dir / "实验报告.md"
