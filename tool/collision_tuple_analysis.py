"""
Collision Tuple 种类与地区分布分析。

给定一个 (p1-p2-p3) tuple，展示该 tuple 下所有 POI 的
Category / Region 分布堆叠图。

用法：
    # 直接指定要分析的 (p1-p2-p3) tuple
    python tool/collision_tuple_analysis.py --experiment PA_main_city --prefix 3-7-12

    # 分析 prefix_len=2 的 (p1-p2) tuple
    python tool/collision_tuple_analysis.py --experiment PA_main_city --prefix 3-7 --prefix_len 2

    # 指定 region_key（city / postal_code / neighborhood）
    python tool/collision_tuple_analysis.py --experiment PA_philly_attrs_emb --prefix 3-7-12 --region_key neighborhood

参数：
    --experiment      实验名称（对应 output/{exp}/semantic_ids.json）
    --prefix          要分析的 (p1-p2-p3) 或 (p1-p2) tuple，如 3-7-12
    --prefix_len      前缀长度，2 或 3（默认 3）
    --region_key      Region 维度使用的字段（默认 city）
    --data_dir        business_poi.json 所在目录（默认 ./dataset/yelp/processed）
    --output_dir      PNG/CSV 输出目录（默认 ./analysis/collision）
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from semantic_id_analysis import (
    load_semantic_ids, load_poi_meta,
    compute_prefix_distribution, plot_single_prefix,
    parse_sid
)
from collections import defaultdict, Counter


def collect_pois_by_prefix(semantic_ids, target_prefix, prefix_len):
    """
    找出所有 sid 前缀等于 target_prefix 的 POI business_id 列表。
    prefix_len=3: 匹配完整 p1-p2-p3
    prefix_len=2: 匹配 p1-p2
    """
    matched = []
    for bid, sid in semantic_ids.items():
        p1, p2, p3, _ = parse_sid(sid)
        if prefix_len == 2:
            key = f"{p1}-{p2}"
        else:
            key = f"{p1}-{p2}-{p3}"
        if key == target_prefix:
            matched.append(bid)
    return matched


def main():
    parser = argparse.ArgumentParser(description='Collision Tuple 分布分析')
    parser.add_argument('--experiment', required=True,
                        help='实验名称（对应 output/{exp}/semantic_ids.json）')
    parser.add_argument('--prefix', type=str, required=True,
                        help='要分析的 tuple，如 3-7-12（prefix_len=3）或 3-7（prefix_len=2）')
    parser.add_argument('--prefix_len', type=int, default=3, choices=[2, 3],
                        help='前缀长度（默认 3）')
    parser.add_argument('--region_key', choices=['city', 'postal_code', 'neighborhood'],
                        default='city', help='Region 维度字段（默认 city）')
    parser.add_argument('--data_dir', default='dataset/yelp/processed',
                        help='business_poi.json 所在目录')
    parser.add_argument('--output_dir', default='./analysis/collision',
                        help='PNG/CSV 输出目录')
    args = parser.parse_args()

    poi_path = os.path.join(args.data_dir, 'business_poi.json')
    sid_path = os.path.join('output', args.experiment, 'semantic_ids.json')

    print(f"Loading POI metadata: {poi_path}")
    poi_meta = load_poi_meta(poi_path)
    print(f"  {len(poi_meta)} POI records loaded.")

    print(f"\nLoading semantic IDs: {sid_path}")
    semantic_ids = load_semantic_ids(sid_path)
    print(f"  {len(semantic_ids)} semantic IDs loaded.")

    # CLI 参数名到 POI 实际字段名的映射
    region_field_map = {'neighborhood': 'plus_code_neighborhood'}
    region_field = region_field_map.get(args.region_key, args.region_key)

    # 收集所有匹配该 prefix 的 POI
    bids = collect_pois_by_prefix(semantic_ids, args.prefix, args.prefix_len)
    print(f"\n  Prefix '{args.prefix}' (len={args.prefix_len}): {len(bids)} POIs matched.")

    if not bids:
        print("[WARN] No POIs found for this prefix.")
        return

    # 打印 POI 明细
    print("\n  POI details:")
    for bid in bids:
        biz = poi_meta.get(bid, {})
        cat = biz.get('primary_category', 'Unknown')
        region = biz.get(region_field, 'Unknown')
        name = biz.get('name', 'Unknown')
        print(f"    [{bid}] {name} | {cat} | {region}")

    # 构造 target semantic_ids 以复用 compute_prefix_distribution
    target_sids = {bid: args.prefix for bid in bids}

    # 计算分布
    df = compute_prefix_distribution(
        target_sids, poi_meta, args.prefix_len,
        p1_filter=None, region_key=region_field
    )

    # Region 字段换回 CLI 参数名显示
    region_display = args.region_key

    os.makedirs(args.output_dir, exist_ok=True)

    # 输出 CSV
    csv_path = os.path.join(
        args.output_dir,
        f'{args.experiment}_tuple-{args.prefix}_len{args.prefix_len}_region-{region_display}.csv'
    )
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # 生成单 tuple 详图
    out_png = os.path.join(
        args.output_dir,
        f'{args.experiment}_tuple-{args.prefix}_len{args.prefix_len}_region-{region_display}.png'
    )
    plot_single_prefix(df, args.experiment, out_png)

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
