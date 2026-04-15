#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


VISITED_RE = re.compile(r"visited\s+((?:<[a-d]_\d+>\s*){3,4})", re.IGNORECASE)


def parse_bins(bin_spec: str) -> List[Tuple[int, int]]:
    bins: List[Tuple[int, int]] = []
    for token in bin_spec.split(","):
        token = token.strip()
        if not token:
            continue
        start, end = token.split("-")
        bins.append((int(start), int(end)))
    if not bins:
        raise ValueError("No valid bins parsed from --bins")
    return bins


def history_len(record: Dict[str, Any]) -> int:
    text = ""
    for key in ["input", "prompt", "query"]:
        value = record.get(key)
        if isinstance(value, str):
            text = value
            break
    return len(VISITED_RE.findall(text))


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must be a JSON list")
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def build_target_allocation(
    gnpr_rows: List[Dict[str, Any]],
    bins: List[Tuple[int, int]],
) -> Tuple[int, Dict[str, int], Dict[str, float], Dict[str, int]]:
    total = len(gnpr_rows)
    gnpr_bucket_counts = Counter()
    for row in gnpr_rows:
        length = history_len(row)
        assigned = False
        for start, end in bins:
            if start <= length <= end:
                gnpr_bucket_counts[f"{start}-{end}"] += 1
                assigned = True
                break
        if not assigned:
            pass

    bucket_ratio = {
        f"{start}-{end}": gnpr_bucket_counts.get(f"{start}-{end}", 0) / max(1, total)
        for start, end in bins
    }

    raw_alloc = {
        name: bucket_ratio[name] * total
        for name in bucket_ratio
    }
    target_alloc = {name: int(value) for name, value in raw_alloc.items()}
    remain = total - sum(target_alloc.values())
    frac_order = sorted(
        ((name, raw_alloc[name] - int(raw_alloc[name])) for name in raw_alloc),
        key=lambda item: item[1],
        reverse=True,
    )
    for idx in range(remain):
        target_alloc[frac_order[idx % len(frac_order)][0]] += 1

    return total, target_alloc, bucket_ratio, dict(gnpr_bucket_counts)


def sample_yelp_rows(
    yelp_rows: List[Dict[str, Any]],
    bins: List[Tuple[int, int]],
    target_alloc: Dict[str, int],
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], int]:
    pool_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    out_of_bins = 0

    for row in yelp_rows:
        length = history_len(row)
        assigned = False
        for start, end in bins:
            name = f"{start}-{end}"
            if start <= length <= end:
                pool_by_bucket[name].append(row)
                assigned = True
                break
        if not assigned:
            out_of_bins += 1

    selected: List[Dict[str, Any]] = []
    actual_before_topup: Dict[str, int] = {}

    for start, end in bins:
        name = f"{start}-{end}"
        need = target_alloc.get(name, 0)
        pool = pool_by_bucket.get(name, [])
        take = min(need, len(pool))
        actual_before_topup[name] = take
        if take > 0:
            selected.extend(rng.sample(pool, take))

    target_total = sum(target_alloc.values())
    shortage = target_total - len(selected)
    if shortage > 0:
        used_ids = {id(item) for item in selected}
        remaining = [row for row in yelp_rows if id(row) not in used_ids]
        if len(remaining) < shortage:
            raise ValueError(
                f"Not enough remaining Yelp rows to top up shortage={shortage}, have={len(remaining)}"
            )
        selected.extend(rng.sample(remaining, shortage))

    rng.shuffle(selected)

    source_pool_sizes = {f"{start}-{end}": len(pool_by_bucket.get(f"{start}-{end}", [])) for start, end in bins}
    return selected, actual_before_topup, source_pool_sizes, out_of_bins


def summarize_selected(rows: List[Dict[str, Any]], bins: List[Tuple[int, int]]) -> Dict[str, Any]:
    length_counter = Counter()
    for row in rows:
        length = history_len(row)
        for start, end in bins:
            if start <= length <= end:
                length_counter[f"{start}-{end}"] += 1
                break
    total = len(rows)
    return {
        "count": total,
        "bucket_counts": {f"{start}-{end}": length_counter.get(f"{start}-{end}", 0) for start, end in bins},
        "bucket_ratio": {
            f"{start}-{end}": length_counter.get(f"{start}-{end}", 0) / max(1, total)
            for start, end in bins
        },
    }


def generate_one_split(
    split_name: str,
    gnpr_path: Path,
    yelp_path: Path,
    out_path: Path,
    bins: List[Tuple[int, int]],
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    gnpr_rows = load_json_list(gnpr_path)
    yelp_rows = load_json_list(yelp_path)

    target_total, target_alloc, bucket_ratio, gnpr_bucket_counts = build_target_allocation(gnpr_rows, bins)
    selected, actual_before_topup, source_pool_sizes, yelp_out_of_bins = sample_yelp_rows(
        yelp_rows,
        bins,
        target_alloc,
        rng,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "split": split_name,
        "seed": seed,
        "gnpr_file": str(gnpr_path),
        "yelp_file": str(yelp_path),
        "output_file": str(out_path),
        "gnpr_total": len(gnpr_rows),
        "yelp_total": len(yelp_rows),
        "target_total": target_total,
        "selected_total": len(selected),
        "gnpr_bucket_counts": gnpr_bucket_counts,
        "gnpr_bucket_ratio": bucket_ratio,
        "target_alloc": target_alloc,
        "actual_alloc_before_topup": actual_before_topup,
        "source_pool_sizes": source_pool_sizes,
        "yelp_out_of_bins": yelp_out_of_bins,
        "selected_summary": summarize_selected(selected, bins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fairness-matched Yelp train/val splits according to GNPR train/val history-length distributions."
    )
    parser.add_argument(
        "--gnpr_train_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/GNPR-SID/V1/datasets/nyc/llm_train.json",
    )
    parser.add_argument(
        "--gnpr_val_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/GNPR-SID/V1/datasets/nyc/llm_val.json",
    )
    parser.add_argument(
        "--gnpr_test_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/GNPR-SID/V1/datasets/nyc/llm_test.json",
    )
    parser.add_argument(
        "--yelp_train_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/train_prompts_50k.json",
    )
    parser.add_argument(
        "--yelp_val_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/val_prompts_10k.json",
    )
    parser.add_argument(
        "--yelp_test_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_10k.json",
    )
    parser.add_argument(
        "--out_train_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/train_prompts_50k_fair_to_gnpr_train.json",
    )
    parser.add_argument(
        "--out_val_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/val_prompts_10k_fair_to_gnpr_val.json",
    )
    parser.add_argument(
        "--out_test_file",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/dataset_cache/test_prompts_10k_fair_to_gnpr_test.json",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="/mnt/data/liuwei/yewenhao/main/output/analysis/fair_train_val_sampling_meta.json",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="1-5,6-10,11-20,21-30,31-40,41-50,51-100,101-200",
    )
    parser.add_argument("--seed", type=int, default=20260415)
    args = parser.parse_args()

    bins = parse_bins(args.bins)

    train_meta = generate_one_split(
        split_name="train",
        gnpr_path=Path(args.gnpr_train_file),
        yelp_path=Path(args.yelp_train_file),
        out_path=Path(args.out_train_file),
        bins=bins,
        seed=args.seed,
    )
    val_meta = generate_one_split(
        split_name="val",
        gnpr_path=Path(args.gnpr_val_file),
        yelp_path=Path(args.yelp_val_file),
        out_path=Path(args.out_val_file),
        bins=bins,
        seed=args.seed + 1,
    )
    test_meta = generate_one_split(
        split_name="test",
        gnpr_path=Path(args.gnpr_test_file),
        yelp_path=Path(args.yelp_test_file),
        out_path=Path(args.out_test_file),
        bins=bins,
        seed=args.seed + 2,
    )

    meta = {
        "bins": [f"{start}-{end}" for start, end in bins],
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
    }

    meta_path = Path(args.meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved train:", args.out_train_file)
    print("saved val:", args.out_val_file)
    print("saved test:", args.out_test_file)
    print("saved meta:", args.meta_path)
    print("train selected/target:", train_meta["selected_total"], "/", train_meta["target_total"])
    print("val selected/target:", val_meta["selected_total"], "/", val_meta["target_total"])
    print("test selected/target:", test_meta["selected_total"], "/", test_meta["target_total"])


if __name__ == "__main__":
    main()
