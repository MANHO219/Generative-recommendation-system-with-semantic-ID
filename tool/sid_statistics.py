"""
分析 semantic_ids.json 的码本利用统计。
对比训练日志中的"活跃码字"与最终量化输出的"实际激活码字"。

用法：
    python tool/sid_statistics.py --experiment CA_cosine_nowarmup
    python tool/sid_statistics.py --experiment CA_cosine_nowarmup --codebook_size 128 --num_quantizers 3
"""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd


def parse_sid(sid):
    """解析 semantic_id，返回 (p1, p2, p3, disambig)"""
    sid = sid.strip()
    disambig = None
    split_pos = len(sid)
    for marker in ('<', '['):
        idx = sid.find(marker)
        if idx != -1:
            split_pos = min(split_pos, idx)

    if split_pos < len(sid):
        base = sid[:split_pos]
        disambig = sid[split_pos:]
    else:
        base = sid

    parts = base.split('-')
    while len(parts) < 3:
        parts.append('*')
    return parts[0], parts[1], parts[2], disambig


def load_semantic_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_quantized_outputs(semantic_ids, codebook_size, num_quantizers):
    """
    分析 semantic_ids.json 中实际激活的码字分布。

    Returns:
        dict: {
            'total_pois': int,
            'disambig_count': int,
            'disambig_rate': float,
            'layer_stats': {
                0: {'active_codes': int, 'codebook_size': int, 'utilization': float,
                    'distribution': Counter, 'pois_per_code': dict},
                ...
            },
            'stop_layer_distribution': dict,  # 各层停止的 POI 比例
        }
    """
    # 按层分组统计
    layer_codes = [defaultdict(list) for _ in range(num_quantizers)]
    disambig_count = 0
    stop_layers = defaultdict(int)  # 每层停止的 POI 数

    for bid, sid in semantic_ids.items():
        p1, p2, p3, disambig = parse_sid(sid)

        if disambig:
            disambig_count += 1

        # 记录每层码字
        if p1 != '*':
            layer_codes[0][p1].append(bid)
        if p2 != '*':
            layer_codes[1][p2].append(bid)
        if p3 != '*':
            layer_codes[2][p3].append(bid)

        # 判断在哪层停止（最后一个非 * 的位置）
        if p3 != '*':
            stop_layer = 3
        elif p2 != '*':
            stop_layer = 2
        elif p1 != '*':
            stop_layer = 1
        else:
            stop_layer = 0
        stop_layers[stop_layer] += 1

    total = len(semantic_ids)
    result = {
        'total_pois': total,
        'disambig_count': disambig_count,
        'disambig_rate': disambig_count / total * 100 if total > 0 else 0,
        'stop_layer_distribution': {k: v / total * 100 for k, v in stop_layers.items()},
        'layer_stats': {}
    }

    for layer_idx in range(num_quantizers):
        active_codes = len(layer_codes[layer_idx])
        utilization = active_codes / codebook_size * 100 if codebook_size > 0 else 0
        # 每个码字的 POI 数量分布
        pois_per_code = {code: len(bids) for code, bids in layer_codes[layer_idx].items()}
        code_counter = Counter(pois_per_code.values())  # POI数 -> 有多少个码字是这个POI数

        result['layer_stats'][layer_idx] = {
            'active_codes': active_codes,
            'codebook_size': codebook_size,
            'utilization': utilization,
            'pois_per_code': pois_per_code,
            'distribution': dict(code_counter),
            'avg_pois_per_code': sum(pois_per_code.values()) / active_codes if active_codes > 0 else 0,
        }

    return result


def analyze_collisions(semantic_ids):
    """
    分析三层量化后的语义冲突。

    L1/L2 的"共享前缀"是层次量化的结构本身，不算冲突。
    只有当 (p1,p2,p3) tuple 对应 >1 个 POI 时才需要 disambig 后缀。

    Returns:
        dict: {
            'total_pois': int,
            'total_tuples': int,        # 唯一 (p1,p2,p3) tuple 数
            'collision_tuples': int,    # 有冲突的 tuple 数
            'collision_pois': int,     # 位于冲突 tuple 中的 POI 数
            'collision_rate': float,    # 冲突 POI 占比
            'collision_size_dist': dict,# {size: count_of_tuples}
            'full_collision_tuples': dict,  # (p1,p2,p3) -> [bid, ...]
        }
    """
    total = len(semantic_ids)
    tuple_groups = defaultdict(list)

    for bid, sid in semantic_ids.items():
        p1, p2, p3, _ = parse_sid(sid)
        tuple_groups[(p1, p2, p3)].append(bid)

    unique_tuples = sum(1 for bids in tuple_groups.values() if len(bids) == 1)
    collision_tuples = sum(1 for bids in tuple_groups.values() if len(bids) > 1)
    collision_pois = sum(len(bids) - 1 for bids in tuple_groups.values() if len(bids) > 1)
    size_dist = Counter(len(bids) for bids in tuple_groups.values() if len(bids) > 1)

    return {
        'total_pois': total,
        'total_tuples': unique_tuples + collision_tuples,
        'unique_tuples': unique_tuples,
        'collision_tuples': collision_tuples,
        'collision_pois': collision_pois,
        'collision_rate': collision_pois / total * 100 if total > 0 else 0,
        'collision_rate_by_tuple': collision_tuples / (unique_tuples + collision_tuples) * 100 if tuple_groups else 0,
        'collision_size_dist': dict(sorted(size_dist.items())),
        'full_collision_tuples': {k: v for k, v in tuple_groups.items() if len(v) > 1},
    }


def print_report(stats, experiment):
    """打印分析报告"""
    print(f"\n{'='*60}")
    print(f"Semantic ID 量化输出分析 — {experiment}")
    print(f"{'='*60}")
    print(f"总 POI 数:      {stats['total_pois']:,}")
    print(f"消歧 POI 数:    {stats['disambig_count']:,} ({stats['disambig_rate']:.1f}%)")

    print(f"\n--- 停止层分布（最后有效码字的层）---")
    for layer in sorted(stats['stop_layer_distribution']):
        pct = stats['stop_layer_distribution'][layer]
        bar = '█' * int(pct / 2)
        print(f"  Layer {layer}: {pct:5.1f}%  {bar}")

    print(f"\n--- 各层码字激活统计 ---")
    print(f"{'Layer':<8} {'激活码数':>10} {'码本大小':>10} {'利用率':>10} {'每码平均POI':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for layer_idx, ls in stats['layer_stats'].items():
        print(f"P{layer_idx+1} (L{layer_idx}): "
              f"{ls['active_codes']:>10} / {ls['codebook_size']:>10} "
              f"{ls['utilization']:>9.1f}%  {ls['avg_pois_per_code']:>11.1f}")

    print(f"\n--- 各层 POI 分布（每码 POI 数 -> 码字数）---")
    for layer_idx, ls in stats['layer_stats'].items():
        dist = ls['distribution']
        sorted_items = sorted(dist.items(), key=lambda x: x[0])
        parts = [f"{pois} POI × {cnt} codes" for pois, cnt in sorted_items]
        print(f"  L{layer_idx}: {', '.join(parts)}")

    print(f"\n--- 各层 Top-10 码字（按 POI 数量排序）---")
    for layer_idx, ls in stats['layer_stats'].items():
        pois_per_code = ls['pois_per_code']
        top_codes = sorted(pois_per_code.items(), key=lambda x: -x[1])[:10]
        print(f"  Layer {layer_idx}:")
        for code, count in top_codes:
            pct = count / stats['total_pois'] * 100
            print(f"    Code {code:>3}: {count:>5} POIs ({pct:5.1f}%)")

    print()


def print_collision_report(cs, experiment):
    """打印冲突分析报告"""
    print(f"\n{'='*60}")
    print(f"码本冲突分析 — {experiment}")
    print(f"{'='*60}")

    print(f"\n--- 三层量化后冲突汇总 ---")
    print(f"  总 POI 数:           {cs['total_pois']:,}")
    print(f"  唯一 tuple 数:       {cs['unique_tuples']:,} ({100 - cs['collision_rate_by_tuple']:.1f}% of tuples)")
    print(f"  冲突 tuple 数:       {cs['collision_tuples']:,} ({cs['collision_rate_by_tuple']:.1f}% of tuples)")
    print(f"  冲突涉及 POI 数:     {cs['collision_pois']:,} ({cs['collision_rate']:.1f}% of POIs)")
    print(f"  需消歧后缀 POI 数:   {cs['collision_pois']:,}")

    if cs['collision_size_dist']:
        print(f"\n--- 冲突规模分布 ---")
        for size, cnt in cs['collision_size_dist'].items():
            print(f"  {size} POIs / tuple × {cnt} tuples")

    # Top 冲突 tuple
    print(f"\n--- 冲突最严重的 10 个 (p1-p2-p3) tuple ---")
    sorted_tuples = sorted(cs['full_collision_tuples'].items(), key=lambda x: -len(x[1]))
    for key, bids in sorted_tuples[:10]:
        print(f"  {key[0]}-{key[1]}-{key[2]}: {len(bids)} POIs")


def export_collision_csv(cs, output_dir, experiment):
    """导出冲突统计为 CSV"""
    # 冲突 tuple 明细
    rows_detail = []
    for key, bids in cs['full_collision_tuples'].items():
        rows_detail.append({
            'p1': key[0], 'p2': key[1], 'p3': key[2],
            'n_pois': len(bids),
            'business_ids': '|'.join(bids[:5]) + ('...' if len(bids) > 5 else '')
        })
    df_detail = pd.DataFrame(rows_detail).sort_values('n_pois', ascending=False)
    path_detail = Path(output_dir) / f'{experiment}_collision_tuples.csv'
    df_detail.to_csv(path_detail, index=False)
    print(f"  导出: {path_detail}")


def export_csv(stats, output_dir, experiment):
    """导出各层码字分布为 CSV"""
    for layer_idx, ls in stats['layer_stats'].items():
        pois_per_code = ls['pois_per_code']
        rows = [{'layer': layer_idx + 1,
                 'code': code,
                 'n_pois': count,
                 'pct_of_total': count / stats['total_pois'] * 100}
                for code, count in sorted(pois_per_code.items(),
                                         key=lambda x: -x[1])]
        df = pd.DataFrame(rows)
        path = Path(output_dir) / f'{experiment}_layer{layer_idx+1}_codes.csv'
        df.to_csv(path, index=False)
        print(f"  导出: {path}")


def main():
    parser = argparse.ArgumentParser(description='Semantic ID 量化输出统计分析')
    parser.add_argument('--experiment', required=True,
                        help='实验名称（对应 output/{exp}/semantic_ids.json）')
    parser.add_argument('--codebook_size', type=int, default=128,
                        help='码本大小（默认 128）')
    parser.add_argument('--num_quantizers', type=int, default=3,
                        help='量化器层数（默认 3）')
    parser.add_argument('--output_dir', default='./analysis',
                        help='CSV 输出目录（默认 ./analysis）')
    parser.add_argument('--log_path', default=None,
                        help='训练日志路径（用于对比训练时利用率）')
    args = parser.parse_args()

    sid_path = f'output/{args.experiment}/semantic_ids.json'
    print(f"加载: {sid_path}")
    semantic_ids = load_semantic_ids(sid_path)

    stats = analyze_quantized_outputs(
        semantic_ids,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers
    )

    print_report(stats, args.experiment)
    export_csv(stats, args.output_dir, args.experiment)

    # 冲突分析
    cs = analyze_collisions(semantic_ids)
    print_collision_report(cs, args.experiment)
    export_collision_csv(cs, args.output_dir, args.experiment)

    # 读取训练日志对比
    if args.log_path:
        try:
            log_util = extract_training_utilization(args.log_path)
            if log_util:
                print(f"{'='*60}")
                print(f"训练日志 vs 最终输出 对比")
                print(f"{'='*60}")
                print(f"{'Layer':<8} {'训练活跃码字':>15} {'最终激活码字':>15} {'差异':>10}")
                print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*10}")
                for layer_idx, (train_code, train_util) in log_util.items():
                    final_code = stats['layer_stats'][layer_idx]['active_codes']
                    diff = final_code - train_code
                    print(f"L{layer_idx+1}: "
                          f"{train_code:>15} / {args.codebook_size} ({train_util:.1f}%)  "
                          f"{final_code:>15}          {diff:>+10}")
        except Exception as e:
            print(f"[WARN] 无法读取训练日志: {e}")


def extract_training_utilization(log_path):
    """从训练日志提取各层码字利用率"""
    import re
    utilization = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配 "level_0: 活跃码字 = 126/128, 利用率 = 98.44%"
            m = re.search(r'level_(\d+):\s*活跃码字\s*=\s*(\d+)/(\d+),\s*利用率\s*=\s*([\d.]+)%', line)
            if m:
                layer = int(m.group(1))
                active = int(m.group(2))
                total = int(m.group(3))
                util = float(m.group(4))
                # 只取第一个（非 best_model 时辰）
                if layer not in utilization:
                    utilization[layer] = (active, util)
    return utilization


if __name__ == '__main__':
    main()
