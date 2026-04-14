"""
统计 Prompt 历史长度分布。

用法示例：
python3 /mnt/data/liuwei/yewenhao/main/inference/prompt_history_stats.py \
    --prompt_file /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_2k.json \
    --prompt_field input \
    --save_path /mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_2k_hist_stats.json
"""

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


VISIT_PATTERN = re.compile(r"visited\s+<[a-d]_\d+><[a-d]_\d+><[a-d]_\d+>(?:<[a-d]_\d+>)?")


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    left = int(k)
    right = min(left + 1, len(sorted_values) - 1)
    if left == right:
        return float(sorted_values[left])
    return sorted_values[left] * (right - k) + sorted_values[right] * (k - left)


def parse_bins(bin_spec: str) -> list[tuple[int, int]]:
    bins = []
    for token in bin_spec.split(","):
        token = token.strip()
        if not token:
            continue
        start, end = token.split("-")
        bins.append((int(start), int(end)))
    return bins


def get_prompt_text(record: dict[str, Any], field: str) -> str:
    if field in record and isinstance(record[field], str):
        return record[field]
    for fallback in ["input", "prompt", "query", "text"]:
        value = record.get(fallback)
        if isinstance(value, str):
            return value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 prompt 历史长度分布")
    parser.add_argument("--prompt_file", type=str, required=True, help="JSON 文件路径（list[dict]）")
    parser.add_argument("--prompt_field", type=str, default="input", help="用于提取历史的文本字段")
    parser.add_argument(
        "--bins",
        type=str,
        default="1-5,6-10,11-20,21-30,31-40,41-50,51-9999",
        help="分桶配置，如 1-5,6-10,11-20",
    )
    parser.add_argument("--save_path", type=str, default="", help="可选：保存结果 JSON")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_file)
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("prompt_file 必须是 JSON 数组")

    lengths = []
    for row in data:
        if not isinstance(row, dict):
            continue
        text = get_prompt_text(row, args.prompt_field)
        lengths.append(len(VISIT_PATTERN.findall(text)))

    if not lengths:
        raise ValueError("未提取到任何样本长度")

    lengths_sorted = sorted(lengths)
    length_freq = Counter(lengths)

    summary: dict[str, Any] = {
        "total": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 4),
        "median": statistics.median(lengths),
        "p90": round(percentile(lengths_sorted, 0.90), 4),
        "p95": round(percentile(lengths_sorted, 0.95), 4),
        "p99": round(percentile(lengths_sorted, 0.99), 4),
        "top_freq": [[k, v] for k, v in length_freq.most_common(20)],
        "bins": {},
    }

    for start, end in parse_bins(args.bins):
        count = sum(1 for x in lengths if start <= x <= end)
        summary["bins"][f"{start}-{end}"] = {
            "count": count,
            "ratio": round(count / len(lengths), 6),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
