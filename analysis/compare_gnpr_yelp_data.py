#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


SID_TOKEN_RE = re.compile(r"<[a-d]_\d+>")
ANGLE_SID_RE = re.compile(r"(?:<[a-d]_\d+>\s*){3,4}")
DASH_SID_RE = re.compile(r"(\d+)-(\d+)-(\d+)(?:(?:<d_(\d+)>)|\[([^\]]+)\])?")
VISITED_SID_RE = re.compile(r"visited\s+((?:<[a-d]_\d+>\s*){3,4})", re.IGNORECASE)
DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")
USER_RE = re.compile(r"user_[a-z0-9_\-]+", re.IGNORECASE)
SPACES_RE = re.compile(r"\s+")


def percentile(sorted_values: List[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def gini_from_counts(counter: Counter) -> float:
    values = sorted(counter.values())
    if not values:
        return 0.0
    total = sum(values)
    n = len(values)
    weighted_sum = 0
    for idx, value in enumerate(values, start=1):
        weighted_sum += (2 * idx - n - 1) * value
    return weighted_sum / (n * total)


def normalize_sid(raw: str) -> Optional[str]:
    text = str(raw).strip()
    if not text:
        return None

    tokens = SID_TOKEN_RE.findall(text)
    if len(tokens) >= 3:
        return "".join(tokens[:4]) if len(tokens) >= 4 else "".join(tokens[:3])

    dash = DASH_SID_RE.search(text)
    if dash:
        first, second, third = int(dash.group(1)), int(dash.group(2)), int(dash.group(3))
        base_sid = f"<a_{first}><b_{second}><c_{third}>"
        d_suffix = dash.group(4)
        if d_suffix is not None:
            return f"{base_sid}<d_{int(d_suffix)}>"
        disambig = dash.group(5)
        if disambig is None:
            return base_sid
        trailing_index = re.search(r"_(\d+)$", disambig)
        if trailing_index:
            return f"{base_sid}<d_{int(trailing_index.group(1))}>"
        if re.fullmatch(r"\d+", disambig):
            return f"{base_sid}<d_{int(disambig)}>"
        return f"{base_sid}<d_0>"

    return None


def get_text_field(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def canonicalize_context(text: str) -> str:
    text = DATETIME_RE.sub("<TIME>", text)
    text = USER_RE.sub("user_<ID>", text)
    text = SPACES_RE.sub(" ", text).strip().lower()
    return text


def sid_level(sid: str) -> str:
    count = len(SID_TOKEN_RE.findall(sid))
    if count == 3:
        return "level3"
    if count == 4:
        return "level4"
    return "other"


def sid_distribution_metrics(counter: Counter, total: int) -> Dict[str, float | int]:
    if total == 0 or not counter:
        return {
            "unique_sid": 0,
            "top1_share": 0.0,
            "top10_share": 0.0,
            "top100_share": 0.0,
            "gini": 0.0,
        }
    most_common = counter.most_common()
    top1 = most_common[0][1]
    top10 = sum(v for _, v in most_common[:10])
    top100 = sum(v for _, v in most_common[:100])
    return {
        "unique_sid": len(counter),
        "top1_share": top1 / total,
        "top10_share": top10 / total,
        "top100_share": top100 / total,
        "gini": gini_from_counts(counter),
    }


def length_stats(lengths: List[int], bins: List[Tuple[int, int]]) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "bucket_ratio": {f"{a}-{b}": 0.0 for a, b in bins},
        }
    sorted_lengths = sorted(lengths)
    bucket_counter = Counter()
    for length in lengths:
        for a, b in bins:
            if a <= length <= b:
                bucket_counter[f"{a}-{b}"] += 1
                break
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "p90": percentile(sorted_lengths, 0.90),
        "p95": percentile(sorted_lengths, 0.95),
        "p99": percentile(sorted_lengths, 0.99),
        "bucket_ratio": {
            f"{a}-{b}": bucket_counter.get(f"{a}-{b}", 0) / len(lengths)
            for a, b in bins
        },
    }


def analyze_dataset(
    records: List[Dict[str, Any]],
    dataset_name: str,
    bins: List[Tuple[int, int]],
) -> Dict[str, Any]:
    total = len(records)
    input_texts = []
    outputs = []
    for row in records:
        input_texts.append(get_text_field(row, ["input", "query", "prompt"]))
        outputs.append(get_text_field(row, ["output", "label", "response", "target_sid"]))

    parsed_labels = []
    label_level_counter = Counter()
    label_parse_fail = 0
    for output in outputs:
        norm = normalize_sid(output)
        if norm is None:
            label_parse_fail += 1
            continue
        parsed_labels.append(norm)
        label_level_counter[sid_level(norm)] += 1

    history_lengths = []
    prompt_sid_counter = Counter()
    prompt_sid_parse_fail = 0
    prompt_with_no_history = 0
    for text in input_texts:
        visited = VISITED_SID_RE.findall(text)
        if not visited:
            prompt_with_no_history += 1
        history_lengths.append(len(visited))
        for sid_text in visited:
            sid_norm = normalize_sid(sid_text)
            if sid_norm is None:
                prompt_sid_parse_fail += 1
            else:
                prompt_sid_counter[sid_norm] += 1

    context_to_labels = defaultdict(set)
    exact_input_counter = Counter(input_texts)
    for text, output in zip(input_texts, outputs):
        sid_norm = normalize_sid(output)
        if sid_norm is None:
            continue
        key = canonicalize_context(text)
        context_to_labels[key].add(sid_norm)

    conflict_groups = sum(1 for labels in context_to_labels.values() if len(labels) > 1)
    conflict_samples = 0
    for text, output in zip(input_texts, outputs):
        sid_norm = normalize_sid(output)
        if sid_norm is None:
            continue
        if len(context_to_labels[canonicalize_context(text)]) > 1:
            conflict_samples += 1

    duplicate_input_samples = sum(c for _, c in exact_input_counter.items() if c > 1)

    result = {
        "dataset": dataset_name,
        "total_samples": total,
        "label_parse_fail_count": label_parse_fail,
        "label_parse_fail_rate": label_parse_fail / total if total else 0.0,
        "label_level_distribution": {
            "level3": label_level_counter.get("level3", 0),
            "level4": label_level_counter.get("level4", 0),
            "other": label_level_counter.get("other", 0),
        },
        "history_length": length_stats(history_lengths, bins),
        "prompt_history_parse_fail_count": prompt_sid_parse_fail,
        "prompt_history_parse_fail_rate": (
            prompt_sid_parse_fail / max(1, sum(history_lengths))
        ),
        "prompt_with_no_history_count": prompt_with_no_history,
        "prompt_with_no_history_rate": prompt_with_no_history / total if total else 0.0,
        "label_sid_distribution": sid_distribution_metrics(Counter(parsed_labels), len(parsed_labels)),
        "prompt_history_sid_distribution": sid_distribution_metrics(prompt_sid_counter, sum(history_lengths)),
        "context_conflict_group_rate": conflict_groups / max(1, len(context_to_labels)),
        "context_conflict_sample_rate": conflict_samples / max(1, total),
        "duplicate_input_sample_rate": duplicate_input_samples / max(1, total),
    }
    return result


def load_records(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError(f"{path} is not a JSON list")
    if max_samples and max_samples > 0:
        return content[:max_samples]
    return content


def build_comparison(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "history_mean_gap": b["history_length"]["mean"] - a["history_length"]["mean"],
        "history_p95_gap": b["history_length"]["p95"] - a["history_length"]["p95"],
        "label_top10_share_gap": (
            b["label_sid_distribution"]["top10_share"] - a["label_sid_distribution"]["top10_share"]
        ),
        "label_gini_gap": b["label_sid_distribution"]["gini"] - a["label_sid_distribution"]["gini"],
        "prompt_sid_top10_share_gap": (
            b["prompt_history_sid_distribution"]["top10_share"]
            - a["prompt_history_sid_distribution"]["top10_share"]
        ),
        "conflict_sample_rate_gap": (
            b["context_conflict_sample_rate"] - a["context_conflict_sample_rate"]
        ),
        "no_history_rate_gap": b["prompt_with_no_history_rate"] - a["prompt_with_no_history_rate"],
    }


def print_summary(title: str, stats: Dict[str, Any]) -> None:
    h = stats["history_length"]
    ldist = stats["label_sid_distribution"]
    pdist = stats["prompt_history_sid_distribution"]
    print(f"\n=== {title} ===")
    print(f"samples: {stats['total_samples']}")
    print(
        f"history_len mean/median/p95: {h['mean']:.2f}/{h['median']:.2f}/{h['p95']:.2f} "
        f"(min={h['min']}, max={h['max']})"
    )
    print(
        f"label_sid unique={ldist['unique_sid']}, top10_share={ldist['top10_share']:.4f}, "
        f"gini={ldist['gini']:.4f}"
    )
    print(
        f"prompt_sid unique={pdist['unique_sid']}, top10_share={pdist['top10_share']:.4f}, "
        f"gini={pdist['gini']:.4f}"
    )
    print(
        f"label_parse_fail_rate={stats['label_parse_fail_rate']:.4f}, "
        f"prompt_parse_fail_rate={stats['prompt_history_parse_fail_rate']:.4f}, "
        f"no_history_rate={stats['prompt_with_no_history_rate']:.4f}"
    )
    print(
        f"conflict_sample_rate={stats['context_conflict_sample_rate']:.4f}, "
        f"conflict_group_rate={stats['context_conflict_group_rate']:.4f}, "
        f"duplicate_input_rate={stats['duplicate_input_sample_rate']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GNPR NYC and Yelp Phil data characteristics (history length, SID distribution, conflict/noise signals)."
    )
    parser.add_argument(
        "--gnpr_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/GNPR-SID/V1/datasets/nyc/llm_train.json",
        help="Path to GNPR NYC training prompts JSON",
    )
    parser.add_argument(
        "--yelp_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/train_prompts_50k.json",
        help="Path to Yelp Phil training prompts JSON",
    )
    parser.add_argument(
        "--gnpr_max_samples",
        type=int,
        default=0,
        help="Optional cap for GNPR samples (0 means all)",
    )
    parser.add_argument(
        "--yelp_max_samples",
        type=int,
        default=0,
        help="Optional cap for Yelp samples (0 means all)",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="1-5,6-10,11-20,21-30,31-40,41-50,51-100,101-200",
        help="History length bins",
    )
    parser.add_argument("--save_path", type=str, default="", help="Optional path to save JSON report")
    args = parser.parse_args()

    bins = []
    for token in args.bins.split(","):
        token = token.strip()
        if not token:
            continue
        start, end = token.split("-")
        bins.append((int(start), int(end)))

    gnpr_records = load_records(Path(args.gnpr_file), args.gnpr_max_samples)
    yelp_records = load_records(Path(args.yelp_file), args.yelp_max_samples)

    gnpr_stats = analyze_dataset(gnpr_records, "GNPR_NYC", bins)
    yelp_stats = analyze_dataset(yelp_records, "YELP_PHIL", bins)
    comparison = build_comparison(gnpr_stats, yelp_stats)

    print_summary("GNPR_NYC", gnpr_stats)
    print_summary("YELP_PHIL", yelp_stats)

    print("\n=== DELTA (YELP - GNPR) ===")
    for key, value in comparison.items():
        print(f"{key}: {value:.6f}")

    report = {
        "gnpr": gnpr_stats,
        "yelp": yelp_stats,
        "delta_yelp_minus_gnpr": comparison,
    }
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
