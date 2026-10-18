"""项目命令入口；所有训练/测试都必须由用户显式执行对应子命令。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from src.artifacts import new_directory, read_json, write_csv, write_json
from src.backend import Backend
from src.data import prepare_data
from src.experiments import DEFAULT_SEEDS, GROUPS, run_experiments
from src.network import TrainConfig


PROJECT = Path(__file__).resolve().parent


def training_arguments(parser):
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--data-dir", type=Path, default=PROJECT)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "baseline.json")
    parser.add_argument("--epochs", type=int, help="覆盖配置文件中的训练轮数")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=0, help="逐轮控制台打印间隔；0 表示只打印每次运行的标题")
    parser.add_argument("--output", type=Path, required=True, help="必须为尚不存在的新目录")


def parser_for_cli():
    parser = argparse.ArgumentParser(description="手写 BP 网络：数据检查、算法检查、对照实验、最终评估和预测")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="检查数据及固定划分，不训练")
    inspect.add_argument("--data-dir", type=Path, default=PROJECT)
    inspect.add_argument("--output", type=Path, default=PROJECT / "results" / "data_check")
    inspect.add_argument("--split-seed", type=int, default=2026)
    inspect.add_argument("--validation-per-class", type=int, default=5)

    check = sub.add_parser("check", help="有限差分梯度检查和后端数值对照，不训练项目数据")
    check.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    check.add_argument("--output", type=Path, help="可选：保存检查结果 JSON（目标文件必须不存在）")

    baseline = sub.add_parser("baseline", help="运行单种子基线，不使用测试分数")
    training_arguments(baseline)
    baseline.add_argument("--seed", type=int, default=42)

    experiments = sub.add_parser("experiments", help="运行多种子单因素对照实验")
    training_arguments(experiments)
    experiments.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    experiments.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))

    report = sub.add_parser("report", help="从已完成的实验记录生成/更新图表和中文报告，不训练")
    report_source = report.add_mutually_exclusive_group(required=True)
    report_source.add_argument("--suite", type=Path, help="对照实验目录")
    report_source.add_argument("--final-dir", type=Path, help="已经完成测试的最终评估目录")

    final = sub.add_parser("finalize", help="按验证结果选配置，在完整建模池重训并最终测试")
    final.add_argument("--suite", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    final.add_argument("--seed", type=int, default=42)
    final.add_argument("--data-dir", type=Path, help="项目搬移后可指定数据位置；文件指纹必须相同")
    final.add_argument("--log-every", type=int, default=100)

    inference = sub.add_parser("predict", help="加载模型并对 4 维特征分类")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    sources = inference.add_mutually_exclusive_group(required=True)
    sources.add_argument("--features", type=float, nargs=4, help="单个样本的四个特征")
    sources.add_argument("--input", type=Path, help="仅含四列特征的空白分隔 TXT，可包含多行")
    inference.add_argument("--output", type=Path, help="可选：输出预测 CSV（目标文件必须不存在）")
    return parser


def dispatch(args):
    if args.command == "inspect":
        bundle = prepare_data(args.data_dir, args.split_seed, args.validation_per_class)
        root = new_directory(args.output)
        write_json(root / "data_audit.json", bundle.audit)
        write_json(root / "data_split.json", bundle.split)
        print(json.dumps({"audit": bundle.audit, "split": bundle.split}, indent=2, ensure_ascii=False))
        print(f"数据检查文件：{root}")
        return

    if args.command == "check":
        from src.validation import all_checks
        if args.output and args.output.exists():
            raise ValueError(f"检查结果文件已存在：{args.output}")
        result = all_checks(Backend(args.device))
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("CHECK PASSED")
        return

    if args.command in ("baseline", "experiments"):
        from src.reporting import generate_report
        config = TrainConfig.from_dict(read_json(args.config))
        if args.epochs is not None:
            config = replace(config, epochs=args.epochs).validate()
        is_baseline = args.command == "baseline"
        root = run_experiments(
            args.data_dir, args.output, Backend(args.device), config,
            seeds=[args.seed] if is_baseline else args.seeds,
            groups=[] if is_baseline else list(dict.fromkeys(args.groups)),
            split_seed=args.split_seed, validation_per_class=args.validation_per_class,
            log_every=args.log_every,
        )
        report = generate_report(root)
        print(f"运行完成：{root}\n报告：{report}")
        return

    if args.command == "report":
        if args.suite:
            from src.reporting import generate_report
            report_path = generate_report(args.suite)
        else:
            from src.finalization import generate_final_report
            report_path = generate_final_report(args.final_dir)
        print(f"报告：{report_path}")
        return

    if args.command == "finalize":
        from src.finalization import finalize
        root, metrics = finalize(args.suite, args.output, Backend(args.device), args.seed, args.data_dir, args.log_every)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"最终评估输出：{root}")
        return

    if args.command == "predict":
        from src.finalization import predict
        if args.output and args.output.exists():
            raise ValueError(f"预测文件已存在：{args.output}")
        features = [args.features] if args.features is not None else np.loadtxt(args.input, dtype=float, encoding="utf-8-sig", ndmin=2)
        rows = predict(args.model, Backend(args.device), features)
        if args.output:
            write_csv(args.output, rows)
        print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = parser_for_cli()
    try:
        dispatch(parser.parse_args())
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n")
