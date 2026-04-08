"""
按州或按城市预拆分 Yelp 数据集。

split_by=state 时输出目录结构：
  {data_dir}/{STATE}/business_poi.json
  {data_dir}/{STATE}/review_poi.json
  {data_dir}/{STATE}/user_poi_interactions.json
  {data_dir}/{STATE}/user_active.json

split_by=city 时输出目录结构（需配合 --states 指定单州）：
  {data_dir}/{STATE}/{CITY}/business_poi.json
  {data_dir}/{STATE}/{CITY}/review_poi.json
  {data_dir}/{STATE}/{CITY}/user_poi_interactions.json
  {data_dir}/{STATE}/{CITY}/user_active.json

用户数据策略：截断 + 过滤短序列
  - 只保留与目标商铺有交互的记录
  - 每个用户的交互数 < min_interactions 则丢弃

用法：
  # 按州拆分（全部州）
  python tool/split_by_state.py --data_dir ./dataset/yelp/processed

  # 按州拆分（指定州）
  python tool/split_by_state.py --data_dir ./dataset/yelp/processed --states PA FL

  # 按城市拆分（单州）
  python tool/split_by_state.py --data_dir ./dataset/yelp/processed --split_by city --states PA
  python tool/split_by_state.py --data_dir ./dataset/yelp/processed --split_by city --states PA --min_interactions 3
"""

import argparse
import json
import os
from collections import defaultdict


# ============================================================
# 解析参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Yelp 数据集按州/城市预拆分')
    parser.add_argument("--data_dir", required=True,
                        help="processed/ 目录路径")
    parser.add_argument("--split_by", choices=["state", "city"], default="state",
                        help="按州拆分还是按城市拆分（city 需配合 --states 指定单州）")
    parser.add_argument("--min_interactions", type=int, default=3,
                        help="用户最少交互数（默认 3）")
    parser.add_argument("--states", nargs="*", default=None,
                        help="只处理指定州，默认处理全部州")
    parser.add_argument("--min_city_pois", type=int, default=100,
                        help="city 模式下最少 POI 数，少于该值则跳过（默认 100）")
    return parser.parse_args()


# ============================================================
# Step 1: 读取 business，建立 business -> (state, city) 查找表
# ============================================================

def load_businesses(data_dir, target_states=None):
    """
    读取 business_poi.json，返回:
      businesses: list of (line, obj)
      biz_key: {business_id: split_key}  # split_key = state 或 "state-city"
      key_info: {split_key: info}        # info = {"state": ..., "city": ...}
    """
    businesses = []
    biz_key = {}

    src = os.path.join(data_dir, "business_poi.json")
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            state = obj.get("state", "")
            city = obj.get("city", "")
            if not state:
                continue
            if target_states and state not in target_states:
                continue
            businesses.append((line, obj))

            if args.split_by == "state":
                split_key = state
            else:  # city
                split_key = f"{state}-{city}"

            biz_key[obj["business_id"]] = split_key

    return businesses, biz_key


def get_all_keys(businesses, split_by):
    """收集所有需要创建的 split_key（州或城市）"""
    keys = set()
    for _, obj in businesses:
        state = obj["state"]
        city = obj.get("city", "")
        if split_by == "state":
            keys.add(state)
        else:
            keys.add(f"{state}-{city}")
    return sorted(keys)


def split_key_info(split_key, split_by):
    """从 split_key 反解 state / city"""
    if split_by == "state":
        return {"state": split_key}
    else:
        state, city = split_key.split("-", 1)
        return {"state": state, "city": city}


def make_output_dir(data_dir, split_key, split_by):
    """创建输出目录并返回路径"""
    if split_by == "state":
        out_dir = os.path.join(data_dir, split_key)
    else:
        state, city = split_key.split("-", 1)
        out_dir = os.path.join(data_dir, state, city)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ============================================================
# Step 1: 拆分 business_poi.json
# ============================================================

def split_businesses_by_key(data_dir, businesses, biz_key, keys, split_by):
    """按 split_key（州或城市）分组写入 business_poi.json"""
    # 收集每个 key 对应的 POI 行
    key_lines = defaultdict(list)
    key_count = defaultdict(int)

    for line, obj in businesses:
        key = biz_key[obj["business_id"]]
        if key in keys:
            key_lines[key].append(line)

    for key in keys:
        out_dir = make_output_dir(data_dir, key, split_by)
        out_path = os.path.join(out_dir, "business_poi.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(key_lines[key]) + "\n")
        key_count[key] = len(key_lines[key])

        info = split_key_info(key, split_by)
        if split_by == "state":
            print(f"  [{info['state']}] business_poi: {key_count[key]} POI")
        else:
            print(f"  [{info['state']}/{info['city']}] business_poi: {key_count[key]} POI")

    return key_count


# ============================================================
# Step 2: 流式拆分 review_poi.json
# ============================================================

def split_reviews_by_key(data_dir, biz_key, keys, split_by):
    """流式处理 review_poi.json，按 business_id 分发到各 key 文件"""
    handles = {}
    counts = defaultdict(int)

    src = os.path.join(data_dir, "review_poi.json")
    try:
        # 打开所有 key 的文件句柄
        for key in keys:
            out_dir = make_output_dir(data_dir, key, split_by)
            handles[key] = open(
                os.path.join(out_dir, "review_poi.json"),
                "w", encoding="utf-8"
            )

        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = biz_key.get(obj.get("business_id", ""))
                if key and key in handles:
                    handles[key].write(line + "\n")
                    counts[key] += 1
    finally:
        for h in handles.values():
            h.close()

    for key in keys:
        info = split_key_info(key, split_by)
        if split_by == "state":
            print(f"  [{info['state']}] review_poi: {counts[key]} 条")
        else:
            print(f"  [{info['state']}/{info['city']}] review_poi: {counts[key]} 条")


# ============================================================
# Step 3: 拆分 user_poi_interactions.json
# ============================================================

def split_interactions_by_key(data_dir, biz_key, keys, split_by, min_interactions):
    """
    按 key 截断用户交互数据，过滤交互数 < min_interactions 的用户。
    """
    # key -> user_id -> [line, ...]
    key_user_lines = {k: defaultdict(list) for k in keys}

    src = os.path.join(data_dir, "user_poi_interactions.json")
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = biz_key.get(obj.get("business_id", ""))
            if key and key in key_user_lines:
                key_user_lines[key][obj["user_id"]].append(line)

    # 过滤 & 写出
    kept_user_ids_by_key = {}
    for key in keys:
        user_map = key_user_lines[key]
        kept_users = 0
        dropped_users = 0
        out_lines = []
        kept_user_ids = set()
        for uid, lines in user_map.items():
            if len(lines) >= min_interactions:
                out_lines.extend(lines)
                kept_users += 1
                kept_user_ids.add(uid)
            else:
                dropped_users += 1

        out_dir = make_output_dir(data_dir, key, split_by)
        out_path = os.path.join(out_dir, "user_poi_interactions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        kept_user_ids_by_key[key] = kept_user_ids

        info = split_key_info(key, split_by)
        label = f"{info['state']}/{info['city']}" if split_by == "city" else info["state"]
        print(f"  [{label}] interactions: 保留 {kept_users} 用户 / 丢弃 {dropped_users} 用户（< {min_interactions}）")

    return kept_user_ids_by_key


# ============================================================
# Step 4: 按子集导出 user_active.json
# ============================================================

def split_user_active_by_key(data_dir, keys, split_by, kept_user_ids_by_key):
    """
    根据每个 key 在 interactions 中保留下来的用户集合，
    从全量 user_active.json 导出子集 user_active.json。
    """
    user_to_keys = defaultdict(list)
    for key in keys:
        for user_id in kept_user_ids_by_key.get(key, set()):
            user_to_keys[user_id].append(key)

    handles = {}
    counts = defaultdict(int)
    src = os.path.join(data_dir, "user_active.json")

    try:
        for key in keys:
            out_dir = make_output_dir(data_dir, key, split_by)
            handles[key] = open(
                os.path.join(out_dir, "user_active.json"),
                "w", encoding="utf-8"
            )

        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key_list = user_to_keys.get(obj.get("user_id", ""))
                if not key_list:
                    continue
                for key in key_list:
                    handles[key].write(line + "\n")
                    counts[key] += 1
    finally:
        for h in handles.values():
            h.close()

    for key in keys:
        info = split_key_info(key, split_by)
        label = f"{info['state']}/{info['city']}" if split_by == "city" else info["state"]
        print(f"  [{label}] user_active: {counts[key]} 条")


# ============================================================
# 城市模式：过滤 POI 过少的城市
# ============================================================

def filter_keys_by_poi_count(businesses, biz_key, keys, split_by, min_city_pois):
    """city 模式下，跳过 POI 数少于 min_city_pois 的城市"""
    # 统计每个 key 的 POI 数
    key_count = defaultdict(int)
    for _, obj in businesses:
        key = biz_key[obj["business_id"]]
        if key in keys:
            key_count[key] += 1

    filtered = sorted(k for k in keys if key_count[k] >= min_city_pois)
    removed = sorted(k for k in keys if key_count[k] < min_city_pois)

    for k in removed:
        info = split_key_info(k, split_by)
        print(f"  [SKIP {info['state']}/{info['city']}] POI={key_count[k]} < {min_city_pois}")

    return filtered


# ============================================================
# 主流程
# ============================================================

args = None  # 全局参数


def main():
    global args
    args = parse_args()

    data_dir = args.data_dir
    target_states = set(args.states) if args.states else None
    split_by = args.split_by

    if split_by == "city":
        if not target_states or len(target_states) != 1:
            raise ValueError("--split_by city 必须配合 --states 指定单个州，例如 --states PA")
        print(f"[city 模式] 目标州: {list(target_states)[0]}，最小 POI 数: {args.min_city_pois}\n")
    else:
        print("[state 模式]\n")

    # ---- Step 0: 读取 business，建立 biz_key 查找表 ----
    print("=== Step 0: 加载 business_poi.json ===")
    businesses, biz_key = load_businesses(data_dir, target_states)
    all_keys = get_all_keys(businesses, split_by)

    if split_by == "city":
        all_keys = filter_keys_by_poi_count(
            businesses, biz_key, all_keys, split_by, args.min_city_pois
        )

    print(f"共 {len(all_keys)} 个分组\n")

    # ---- Step 1: 拆分 business_poi.json ----
    print(f"=== Step 1: 拆分 business_poi.json ===")
    split_businesses_by_key(data_dir, businesses, biz_key, all_keys, split_by)
    print()

    # ---- Step 2: 流式拆分 review_poi.json ----
    print("=== Step 2: 拆分 review_poi.json（流式）===")
    split_reviews_by_key(data_dir, biz_key, all_keys, split_by)
    print()

    # ---- Step 3: 拆分 user_poi_interactions.json ----
    print(f"=== Step 3: 拆分 user_poi_interactions.json（min_interactions={args.min_interactions}）===")
    kept_user_ids_by_key = split_interactions_by_key(
        data_dir, biz_key, all_keys, split_by, args.min_interactions
    )
    print()

    # ---- Step 4: 按子集导出 user_active.json ----
    print("=== Step 4: 拆分 user_active.json（按保留用户子集）===")
    split_user_active_by_key(data_dir, all_keys, split_by, kept_user_ids_by_key)
    print()

    print("完成。")


if __name__ == "__main__":
    main()
