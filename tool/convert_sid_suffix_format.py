"""
一次性转换脚本：将旧格式 <d_N> 后缀的 semantic_ids.json 转为新格式 [GG] 后缀

用法：
    python tool/convert_sid_suffix_format.py \
        --input output/semantic_ids.json \
        --output output/semantic_ids_v2.json \
        --dataset dataset/yelp/processed/business_poi.json
先用 --dry-run 预览前 10 条转换结果，确认格式正确后再正式写入

新格式：12-34-56[GQ]  (GQ = plus_code_neighborhood 最后2位)
冲突时：12-34-56[GQ], 12-34-56[GQ_1], 12-34-56[GQ_2], ...
"""

import json
import argparse
from collections import defaultdict


def load_pluscode_map(business_poi_path: str) -> dict:
    """从 business_poi.json 构建 {business_id: plus_code_neighborhood} 映射"""
    pluscode_map = {}
    with open(business_poi_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            bid = obj.get('business_id', '')
            neighborhood = obj.get('plus_code_neighborhood', '')
            if bid:
                pluscode_map[bid] = neighborhood
    return pluscode_map


def strip_old_suffix(sid: str) -> str:
    """去掉旧后缀 <d_N>，返回 base SID"""
    return sid.split('<')[0]


def convert_sid_format(
    semantic_ids: dict,
    pluscode_map: dict,
    suffix_mode: str = "grid"  # "grid" | "index"
) -> dict:
    """
    将旧格式 <d_N> 转换为新格式 [GG]

    Args:
        semantic_ids: {business_id: old_sid_string}
        pluscode_map: {business_id: plus_code_neighborhood}
        suffix_mode: "grid" 使用 [GG]，或 "index" 退回 <d_N>（仅保底）

    Returns:
        {business_id: new_sid_string}
    """
    # Step 1: 按 base SID 分组
    base_to_bids = defaultdict(list)
    for bid, sid in semantic_ids.items():
        base = strip_old_suffix(sid)
        base_to_bids[base].append(bid)

    # Step 2: 对每个组分配新后缀
    converted = {}

    for base, bids in base_to_bids.items():
        if len(bids) == 1:
            # 无冲突，保持 base SID（不加后缀）
            converted[bids[0]] = base
        else:
            if suffix_mode == "index":
                for i, bid in enumerate(bids):
                    converted[bid] = f"{base}<d_{i}>"
                continue

            # 有冲突：按 plus_code_neighborhood[-2:] 分组
            gg_groups = defaultdict(list)
            for bid in bids:
                neighborhood = pluscode_map.get(bid, '')
                if len(neighborhood) >= 2:
                    gg = neighborhood[-2:]
                else:
                    gg = 'UNK'
                gg_groups[gg].append(bid)

            # 组内再按 _n 序号分配，保证全局唯一
            for gg, group in gg_groups.items():
                for i, bid in enumerate(group):
                    if i == 0:
                        suffix = f"[{gg}]"
                    else:
                        suffix = f"[{gg}_{i}]"
                    converted[bid] = base + suffix

    return converted


def main():
    parser = argparse.ArgumentParser(description="将 SID 后缀从 <d_N> 转为 [GG]")
    parser.add_argument('--input', '-i', required=True, help='输入 semantic_ids.json 路径')
    parser.add_argument('--output', '-o', required=True, help='输出路径')
    parser.add_argument('--dataset', '-d', required=True,
                        help='business_poi.json 路径（用于获取 plus_code_neighborhood）')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印前10条转换结果，不写入文件')
    parser.add_argument('--suffix-mode', default='grid',
                        choices=['grid', 'index'],
                        help='grid=[GG], index=<d_N>（仅保底）')
    args = parser.parse_args()

    # 加载数据
    print(f"加载 semantic_ids: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        semantic_ids = json.load(f)
    print(f"  共 {len(semantic_ids)} 条记录")

    print(f"加载 plus_code_neighborhood: {args.dataset}")
    pluscode_map = load_pluscode_map(args.dataset)
    print(f"  共 {len(pluscode_map)} 条 POI 记录")

    # 转换
    print("正在转换...")
    converted = convert_sid_format(semantic_ids, pluscode_map, args.suffix_mode)

    # 统计
    old_has_d = sum(1 for v in semantic_ids.values() if '<d_' in v)
    new_has_bracket = sum(1 for v in converted.values() if '[' in v)
    print(f"  旧格式含 <d_> 的条目: {old_has_d}")
    print(f"  新格式含 [  的条目: {new_has_bracket}")

    # 抽样展示
    print("\n前 10 条转换示例：")
    count = 0
    for bid, new_sid in converted.items():
        old_sid = semantic_ids.get(bid, 'N/A')
        print(f"  {old_sid} → {new_sid}")
        count += 1
        if count >= 10:
            break

    if args.dry_run:
        print("\n[dry-run] 未写入文件")
        return

    # 写入
    print(f"\n写入: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print("完成！")
    print("\n注意：更换 semantic_ids.json 后，请删除 dataset_cache 并重建：")
    print("  rm -rf output/dataset_cache")


if __name__ == '__main__':
    main()
