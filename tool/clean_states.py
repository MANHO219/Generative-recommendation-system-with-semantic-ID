"""
扫描按州拆分后的目录，识别并删除垃圾州子目录。

判定标准（满足任一即为垃圾州）：
  - POI 数量 < min_poi（默认 100）
  - 保留用户数 == 0

默认为 dry-run 模式，只打印不删除。加 --execute 才真正删除。

用法：
  python tool/clean_states.py --data_dir ./dataset/yelp/processed
  python tool/clean_states.py --data_dir ./dataset/yelp/processed --execute
  python tool/clean_states.py --data_dir ./dataset/yelp/processed --min_poi 50 --execute
"""

import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="processed/ 目录路径")
    parser.add_argument("--min_poi", type=int, default=100,
                        help="POI 数量低于此值视为垃圾州（默认 100）")
    parser.add_argument("--execute", action="store_true",
                        help="真正删除目录，默认只 dry-run")
    return parser.parse_args()


def count_lines(path):
    """统计文件行数（跳过空行）。"""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_users(interaction_path):
    """统计 user_poi_interactions.json 中的不重复用户数。"""
    import json
    users = set()
    with open(interaction_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            users.add(obj["user_id"])
    return len(users)


def scan_states(data_dir, min_poi):
    """扫描所有州子目录，返回 (keep, trash) 两个列表，每项为 (state, poi, users)。"""
    keep = []
    trash = []

    entries = sorted(os.listdir(data_dir))
    for entry in entries:
        state_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(state_dir):
            continue

        biz_path = os.path.join(state_dir, "business_poi.json")
        inter_path = os.path.join(state_dir, "user_poi_interactions.json")

        if not os.path.exists(biz_path) or not os.path.exists(inter_path):
            continue  # 不是州子目录（如原始文件目录）

        poi_count = count_lines(biz_path)
        user_count = count_users(inter_path)

        info = (entry, poi_count, user_count)
        if poi_count < min_poi or user_count == 0:
            trash.append(info)
        else:
            keep.append(info)

    return keep, trash


def main():
    args = parse_args()

    print(f"扫描目录: {args.data_dir}  (min_poi={args.min_poi})\n")
    keep, trash = scan_states(args.data_dir, args.min_poi)

    print(f"{'州':<6} {'POI':>8} {'用户':>8}  状态")
    print("-" * 35)
    for state, poi, users in keep:
        print(f"{state:<6} {poi:>8} {users:>8}  保留")
    for state, poi, users in trash:
        tag = "删除" if args.execute else "待删除(dry-run)"
        print(f"{state:<6} {poi:>8} {users:>8}  {tag}")

    print(f"\n保留 {len(keep)} 个州，{'删除' if args.execute else '待删除'} {len(trash)} 个州")

    if not trash:
        print("无需清理。")
        return

    if not args.execute:
        print("\n[dry-run] 加 --execute 参数才会真正删除。")
        return

    confirm = input(f"\n确认删除以下州目录？{'、'.join(s for s, *_ in trash)} [y/N] ").strip().lower()
    if confirm != "y":
        print("已取消。")
        return

    for state, poi, users in trash:
        state_dir = os.path.join(args.data_dir, state)
        shutil.rmtree(state_dir)
        print(f"  已删除: {state_dir}")

    print("清理完成。")


if __name__ == "__main__":
    main()
