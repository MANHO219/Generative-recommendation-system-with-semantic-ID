"""
从已构建的数据缓存中提取固定规模子集（如 10k/2k/2k）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, List, Dict, Tuple


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _write_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    temp_path.replace(path)


def _choose_indices(total: int, count: int, seed: int, shuffle: bool) -> List[int]:
    count = min(total, max(0, count))
    indices = list(range(total))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return sorted(indices[:count])


def _extract_by_user_fraction(
    source_dir: Path,
    target_dir: Path,
    split_name: str,
    user_fraction: float,
    seed: int,
) -> Tuple[int, int]:
    """按用户比例抽取：随机选取指定比例的用户，保留这些用户的所有样本。"""
    sample_path = source_dir / f"{split_name}_samples.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing source split file: {sample_path}")

    samples = _load_json(sample_path)

    # 按 user_id 分组
    user_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        uid = sample.get("user_id", f"__anonymous_{idx}")
        user_to_indices.setdefault(uid, []).append(idx)

    all_users = list(user_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_users)

    n_users = max(1, int(len(all_users) * user_fraction))
    selected_users = set(all_users[:n_users])

    picked_indices = []
    for uid, indices in user_to_indices.items():
        if uid in selected_users:
            picked_indices.extend(indices)

    picked_indices.sort()
    subset_samples = [samples[idx] for idx in picked_indices]
    _write_json(target_dir / f"{split_name}_samples.json", subset_samples)

    prompt_path = source_dir / f"{split_name}_prompts.json"
    if prompt_path.exists():
        prompts = _load_json(prompt_path)
        if len(prompts) != len(samples):
            raise ValueError(
                f"Prompt/sample size mismatch for {split_name}: {len(prompts)} vs {len(samples)}"
            )
        subset_prompts = [prompts[idx] for idx in picked_indices]
        _write_json(target_dir / f"{split_name}_prompts.json", subset_prompts, indent=2)

    return len(samples), len(subset_samples)


def _extract_split(
    source_dir: Path,
    target_dir: Path,
    split_name: str,
    count: int,
    seed: int,
    shuffle: bool,
) -> Tuple[int, int]:
    sample_path = source_dir / f"{split_name}_samples.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing source split file: {sample_path}")

    samples = _load_json(sample_path)
    picked = _choose_indices(len(samples), count, seed, shuffle)
    subset_samples = [samples[idx] for idx in picked]
    _write_json(target_dir / f"{split_name}_samples.json", subset_samples)

    prompt_path = source_dir / f"{split_name}_prompts.json"
    if prompt_path.exists():
        prompts = _load_json(prompt_path)
        if len(prompts) != len(samples):
            raise ValueError(
                f"Prompt/sample size mismatch for {split_name}: {len(prompts)} vs {len(samples)}"
            )
        subset_prompts = [prompts[idx] for idx in picked]
        _write_json(target_dir / f"{split_name}_prompts.json", subset_prompts, indent=2)

    return len(samples), len(subset_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 dataset_cache 提取固定规模子集。")
    parser.add_argument("--source_dir", type=str, required=True, help="源缓存目录（含 *_samples.json）")
    parser.add_argument("--target_dir", type=str, required=True, help="目标输出目录")
    parser.add_argument("--train_n", type=int, default=None, help="train 子集大小（默认 10000）")
    parser.add_argument("--val_n", type=int, default=None, help="val 子集大小（默认 2000）")
    parser.add_argument("--test_n", type=int, default=None, help="test 子集大小（默认 2000）")
    parser.add_argument(
        "--test_source",
        type=str,
        default="test",
        choices=["test", "test_last_item"],
        help="测试集来源（test 或 test_last_item）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_shuffle", action="store_true", help="不打乱，按原顺序截取前 N 条")
    # 用户比例抽取模式
    parser.add_argument("--user_fraction", type=float, default=None,
                        help="按用户比例抽取：随机选取该比例的用户，保留其所有样本（与 N 参数互斥）")
    parser.add_argument("--val_user_fraction", type=float, default=None,
                        help="val 集的独立用户比例（默认同 --user_fraction）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    shuffle = not args.no_shuffle

    use_user_fraction = args.user_fraction is not None
    train_n = args.train_n if args.train_n is not None else 10000
    val_n = args.val_n if args.val_n is not None else 2000
    test_n = args.test_n if args.test_n is not None else 2000

    if use_user_fraction:
        val_frac = args.val_user_fraction if args.val_user_fraction is not None else args.user_fraction

        train_total, train_kept = _extract_by_user_fraction(
            source_dir, target_dir, "train", args.user_fraction, args.seed + 1
        )
        val_total, val_kept = _extract_by_user_fraction(
            source_dir, target_dir, "val", val_frac, args.seed + 2
        )
        test_total, test_kept = _extract_by_user_fraction(
            source_dir, target_dir, args.test_source, args.user_fraction, args.seed + 3
        )
    else:
        train_total, train_kept = _extract_split(source_dir, target_dir, "train", train_n, args.seed + 1, shuffle)
        val_total, val_kept = _extract_split(source_dir, target_dir, "val", val_n, args.seed + 2, shuffle)
        test_total, test_kept = _extract_split(
            source_dir,
            target_dir,
            args.test_source,
            test_n,
            args.seed + 3,
            shuffle,
        )

    if args.test_source != "test":
        src_samples = target_dir / f"{args.test_source}_samples.json"
        dst_samples = target_dir / "test_samples.json"
        if src_samples.exists():
            src_samples.replace(dst_samples)

        src_prompts = target_dir / f"{args.test_source}_prompts.json"
        dst_prompts = target_dir / "test_prompts.json"
        if src_prompts.exists():
            src_prompts.replace(dst_prompts)

    summary = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "shuffle": shuffle,
        "seed": args.seed,
        "counts": {
            "train": {"source": train_total, "subset": train_kept},
            "val": {"source": val_total, "subset": val_kept},
            "test": {"source": test_total, "subset": test_kept, "source_split": args.test_source},
        },
    }
    _write_json(target_dir / "subset_manifest.json", summary, indent=2)

    print("Subset extraction finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
