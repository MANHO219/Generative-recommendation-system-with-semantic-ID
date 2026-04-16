"""
Semantic ID 训练前的 k-core 过滤工具。

目标：
1. 在 user-business 交互层面执行迭代 k-core（默认 5-core）
2. 过滤 business/review/checkin/user 文件，保证后续 codebook 与训练口径一致
3. 生成一个独立的过滤后数据目录，不污染原始数据
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REVIEW_FILES = ("review_filtered.json", "review_poi.json")
BUSINESS_FILES = ("business_poi.json",)
CHECKIN_FILES = ("checkin_poi.json",)
USER_FILES = ("user_active.json",)


def _read_records(path: Path) -> Tuple[List[Dict], str]:
    """读取 JSON / JSONL 文件并统一成 records 列表。"""
    if not path.exists():
        return [], "missing"

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "empty"

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            records = []
            for key, value in data.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("__source_key__", str(key))
                    records.append(item)
            return records, "json-dict"
        if isinstance(data, list):
            records = [x for x in data if isinstance(x, dict)]
            return records, "json-list"
    except json.JSONDecodeError:
        pass

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    return records, "jsonl"


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in records:
            row = dict(row)
            row.pop("__source_key__", None)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_interactions(records: List[Dict]) -> List[Tuple[str, str]]:
    interactions: List[Tuple[str, str]] = []
    for row in records:
        user_id = row.get("user_id")
        business_id = row.get("business_id")
        if user_id is None or business_id is None:
            continue
        user_id = str(user_id)
        business_id = str(business_id)
        if user_id and business_id:
            interactions.append((user_id, business_id))
    return interactions


def _iterative_k_core(interactions: List[Tuple[str, str]], k: int) -> Tuple[set, set, List[Tuple[str, str]]]:
    """迭代 user/item 过滤直到稳定。"""
    current = interactions
    while True:
        prev_size = len(current)
        user_counts = Counter(user_id for user_id, _ in current)
        item_counts = Counter(item_id for _, item_id in current)

        keep_users = {user_id for user_id, count in user_counts.items() if count >= k}
        keep_items = {item_id for item_id, count in item_counts.items() if count >= k}

        current = [
            (user_id, item_id)
            for user_id, item_id in current
            if user_id in keep_users and item_id in keep_items
        ]

        if len(current) == prev_size:
            break

    final_users = {user_id for user_id, _ in current}
    final_items = {item_id for _, item_id in current}
    return final_users, final_items, current


def prepare_k_core_data_dir(
    data_dir: str,
    k: int = 5,
    output_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[str, Dict[str, int]]:
    """
    在 semantic id 训练前执行 k-core 过滤并返回可训练目录。

    Returns:
        (filtered_data_dir, stats)
    """
    src_dir = Path(data_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {src_dir}")

    if k <= 1:
        logging.info("k-core 未启用（k <= 1），直接使用原始数据目录")
        return str(src_dir), {
            "k": k,
            "users": -1,
            "items": -1,
            "interactions": -1,
        }

    if output_dir is None:
        output_dir = str(src_dir / ".cache" / f"kcore_{k}")
    dst_dir = Path(output_dir)

    marker_file = dst_dir / "kcore_stats.json"
    if use_cache and marker_file.exists():
        logging.info(f"复用已存在 k-core 数据目录: {dst_dir}")
        stats = json.loads(marker_file.read_text(encoding="utf-8"))
        return str(dst_dir), stats

    review_records: List[Dict] = []
    loaded_review_files: List[str] = []
    for filename in REVIEW_FILES:
        records, mode = _read_records(src_dir / filename)
        if records:
            review_records.extend(records)
            loaded_review_files.append(f"{filename}({mode})")

    if not review_records:
        logging.warning("未找到可用于 k-core 的 review 文件，跳过过滤")
        return str(src_dir), {
            "k": k,
            "users": -1,
            "items": -1,
            "interactions": -1,
        }

    interactions = _extract_interactions(review_records)
    if not interactions:
        logging.warning("review 文件不包含 user_id/business_id，跳过 k-core")
        return str(src_dir), {
            "k": k,
            "users": -1,
            "items": -1,
            "interactions": -1,
        }

    # 迭代 k-core
    keep_users, keep_items, filtered_interactions = _iterative_k_core(interactions, k)

    stats = {
        "k": k,
        "input_interactions": len(interactions),
        "interactions": len(filtered_interactions),
        "users": len(keep_users),
        "items": len(keep_items),
        "loaded_review_files": loaded_review_files,
    }

    if not keep_users or not keep_items:
        raise RuntimeError(
            f"k-core 过滤后为空，请降低 k（当前 k={k}）。"
        )

    # 复制原目录文件到新目录，再覆写关键文件
    dst_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_dir.iterdir():
        if entry.is_file():
            shutil.copy2(entry, dst_dir / entry.name)

    # 过滤并覆写 review 文件
    for filename in REVIEW_FILES:
        src_file = src_dir / filename
        if not src_file.exists():
            continue
        records, _ = _read_records(src_file)
        filtered = []
        for row in records:
            user_id = row.get("user_id")
            business_id = row.get("business_id")
            if user_id is None or business_id is None:
                continue
            if str(user_id) in keep_users and str(business_id) in keep_items:
                filtered.append(row)
        _write_jsonl(dst_dir / filename, filtered)

    # 过滤 business / checkin
    for filename in (*BUSINESS_FILES, *CHECKIN_FILES):
        src_file = src_dir / filename
        if not src_file.exists():
            continue
        records, _ = _read_records(src_file)
        filtered = []
        for row in records:
            business_id = row.get("business_id")
            if business_id is None:
                business_id = row.get("bid")
            if business_id is None:
                continue
            if str(business_id) in keep_items:
                filtered.append(row)
        _write_jsonl(dst_dir / filename, filtered)

    # 过滤 user 文件
    for filename in USER_FILES:
        src_file = src_dir / filename
        if not src_file.exists():
            continue
        records, _ = _read_records(src_file)
        filtered = []
        for row in records:
            user_id = row.get("user_id")
            if user_id is None:
                user_id = row.get("__source_key__")
            if user_id is None:
                continue
            if str(user_id) in keep_users:
                filtered.append(row)
        _write_jsonl(dst_dir / filename, filtered)

    marker_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "k-core 完成: k=%d, users=%d, items=%d, interactions=%d",
        stats["k"], stats["users"], stats["items"], stats["interactions"],
    )

    return str(dst_dir), stats
