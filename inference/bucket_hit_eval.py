"""
按历史长度分桶评估命中率（HR@1/HR@K/MRR@K）。

用法示例：
python3 /mnt/data/liuwei/yewenhao/main/inference/bucket_hit_eval.py \
    --pred_file /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/generated_predictions.jsonl \
    --top_k 5 \
    --prompt_field prompt \
    --save_path /mnt/data/liuwei/yewenhao/main/output/llamafactory/yelp_prompts_phil_lora_10k2k_predict/bucket_hit_eval.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


VISIT_PATTERN = re.compile(r"visited\s+<[a-d]_\d+><[a-d]_\d+><[a-d]_\d+>(?:<[a-d]_\d+>)?")
ANGLE_SID_PATTERN = re.compile(r"(?:<[a-d]_\d+>){3,4}")
DASH_SID_PATTERN = re.compile(r"\d+-\d+-\d+(?:\[[^\]]+\])?")


def parse_bins(bin_spec: str) -> list[tuple[int, int]]:
    bins = []
    for token in bin_spec.split(","):
        token = token.strip()
        if not token:
            continue
        start, end = token.split("-")
        bins.append((int(start), int(end)))
    return bins


def is_angle_bracket_sid(value: str) -> bool:
    return bool(re.fullmatch(r"(?:<[a-d]_\d+>){3,4}", value.strip()))


def to_angle_bracket_sid(value: str) -> str:
    sid = value.strip()
    if is_angle_bracket_sid(sid):
        return sid

    match = re.fullmatch(r"(\d+)-(\d+)-(\d+)(?:\[([^\]]+)\])?", sid)
    if not match:
        raise ValueError(f"Unsupported SID format: {value}")

    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    base_sid = "".join(f"<{label}_{item}>" for label, item in zip(["a", "b", "c"], values))
    disambig = match.group(4)
    if disambig is None:
        return base_sid

    trailing_index = re.search(r"_(\d+)$", disambig)
    if trailing_index:
        return f"{base_sid}<d_{int(trailing_index.group(1))}>"

    pure_number = re.fullmatch(r"\d+", disambig)
    if pure_number:
        return f"{base_sid}<d_{int(disambig)}>"

    return f"{base_sid}<d_0>"


def normalize_sid_text(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    angle_match = ANGLE_SID_PATTERN.search(text)
    if angle_match:
        return angle_match.group(0)

    dash_match = DASH_SID_PATTERN.search(text)
    if dash_match:
        try:
            return to_angle_bracket_sid(dash_match.group(0))
        except ValueError:
            return text

    try:
        return to_angle_bracket_sid(text)
    except ValueError:
        return text


def parse_candidates(record: dict[str, Any]) -> list[str]:
    for key in ["candidates", "topk", "top_k", "top_k_predictions", "predict_topk"]:
        value = record.get(key)
        if isinstance(value, list):
            candidates = []
            seen = set()
            for item in value:
                if not isinstance(item, str):
                    continue
                norm = normalize_sid_text(item)
                if norm and norm not in seen:
                    seen.add(norm)
                    candidates.append(norm)
            if candidates:
                return candidates

    pred_text = ""
    for key in ["predict", "prediction", "pred", "text"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            pred_text = value
            break

    if not pred_text:
        return []

    extracted = ANGLE_SID_PATTERN.findall(pred_text)
    if not extracted:
        extracted = [to_angle_bracket_sid(x) for x in DASH_SID_PATTERN.findall(pred_text)]

    if extracted:
        dedup = []
        seen = set()
        for item in extracted:
            norm = normalize_sid_text(item)
            if norm and norm not in seen:
                seen.add(norm)
                dedup.append(norm)
        return dedup

    single = normalize_sid_text(pred_text)
    return [single] if single else []


def parse_label(record: dict[str, Any]) -> str:
    for key in ["label", "output", "target_sid", "gt", "ground_truth"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_sid_text(value)
    return ""


def parse_history_len(record: dict[str, Any], prompt_field: str) -> int:
    text = ""
    if isinstance(record.get(prompt_field), str):
        text = record[prompt_field]
    else:
        for fallback in ["prompt", "input", "query"]:
            value = record.get(fallback)
            if isinstance(value, str):
                text = value
                break
    return len(VISIT_PATTERN.findall(text))


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if path.suffix.lower() == ".json":
        loaded = json.loads(content)
        if not isinstance(loaded, list):
            raise ValueError("JSON 预测文件必须是数组")
        return loaded
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def init_bucket_stats() -> dict[str, float | int]:
    return {
        "total": 0,
        "hr@1_count": 0,
        "hr@k_count": 0,
        "mrr_sum": 0.0,
        "valid_pred_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按历史长度分桶评估命中率")
    parser.add_argument("--pred_file", type=str, required=True, help="预测文件路径（json/jsonl）")
    parser.add_argument("--top_k", type=int, default=5, help="评估的 K")
    parser.add_argument("--bins", type=str, default="1-5,6-10,11-20,21-30,31-40,41-50")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="用于统计历史长度的文本字段")
    parser.add_argument("--save_path", type=str, default="", help="可选：保存评估结果")
    args = parser.parse_args()

    bins = parse_bins(args.bins)
    records = read_json_or_jsonl(Path(args.pred_file))
    if not records:
        raise ValueError("预测文件为空")

    overall = init_bucket_stats()
    bucket_stats = {f"{a}-{b}": init_bucket_stats() for a, b in bins}

    for row in records:
        label = parse_label(row)
        candidates = parse_candidates(row)
        history_len = parse_history_len(row, args.prompt_field)

        bucket_name = None
        for start, end in bins:
            if start <= history_len <= end:
                bucket_name = f"{start}-{end}"
                break
        if bucket_name is None:
            continue

        if not label or not candidates:
            continue

        topk = candidates[: max(1, args.top_k)]

        def _update(stats: dict[str, float | int]) -> None:
            stats["total"] += 1
            if topk[0] == label:
                stats["hr@1_count"] += 1
            if label in topk:
                stats["hr@k_count"] += 1
                rank = topk.index(label) + 1
                stats["mrr_sum"] += 1.0 / rank
            if topk[0]:
                stats["valid_pred_count"] += 1

        _update(overall)
        _update(bucket_stats[bucket_name])

    def _finalize(stats: dict[str, float | int]) -> dict[str, float | int]:
        total = int(stats["total"])
        if total == 0:
            return {
                "total": 0,
                "hr@1": 0.0,
                f"hr@{args.top_k}": 0.0,
                f"mrr@{args.top_k}": 0.0,
                "valid_pred_rate": 0.0,
            }
        return {
            "total": total,
            "hr@1": stats["hr@1_count"] / total,
            f"hr@{args.top_k}": stats["hr@k_count"] / total,
            f"mrr@{args.top_k}": stats["mrr_sum"] / total,
            "valid_pred_rate": stats["valid_pred_count"] / total,
        }

    result = {
        "overall": _finalize(overall),
        "buckets": {name: _finalize(stats) for name, stats in bucket_stats.items()},
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
