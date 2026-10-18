"""统一保存实验文件；新运行不覆盖已有目录，避免不同实验结果混在一起。"""

import csv
import json
from pathlib import Path


def new_directory(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError(f"输出目录已存在：{path}。请指定一个新的 --output 路径。")
    path.mkdir(parents=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    # utf-8-sig 方便 Windows Excel 直接打开中文列值。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
