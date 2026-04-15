"""
Semantic ID Prefix 分布分析工具
展示同一 prefix 下 POI 的 Category / Region 分布。

用法：
    # P1 层概览（最多展示 10 个 prefix）
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup --prefix_len 1

    # P1 层，展示全部 prefix
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup --prefix_len 1 --max_prefixes 200

    # P2 层概览（以 P1=1 为例）
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup --prefix_len 2 --p1_filter 1

    # P2 层概览（指定 P1 和 P2）
    python tool/semantic_id_analysis.py --experiments PA_main_city --prefix_len 2 --p1_filter 28 --p2_filter 39

    # P2 层，展示全部 P1=1 下的 P2 prefix
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup --prefix_len 2 --p1_filter 1 --max_prefixes 200

    # P3 层详图（只画单个 prefix）
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup --prefix_len 3 --prefix_index 0

    # 多实验对比
    python tool/semantic_id_analysis.py --experiments CA_cosine_nowarmup PA_cosine_nowarmup --prefix_len 1

    # 城市级数据分析：用 postal_code 或 plus_code_neighborhood 代替 city 作为 Region 维度
    python tool/semantic_id_analysis.py --experiments PA_philly_attrs_emb --prefix_len 1 --region_key postal_code
    python tool/semantic_id_analysis.py --experiments PA_philly_attrs_emb --prefix_len 1 --region_key neighborhood

参数：
    --data_dir        business_poi.json 所在目录（默认 ./dataset/yelp/processed）
    --output_dir      PNG/CSV 输出目录（默认 ./analysis）
    --experiments     实验名称列表，对应 output/{exp}/semantic_ids.json
    --prefix_len      前缀长度，1/2/3（默认 1）
    --p1_filter       只展示指定 P1 值（如 1、2）的 prefix
    --p2_filter       只展示指定 P2 值（需配合 prefix_len>=2）
    --max_prefixes    概览图最多展示 prefix 数（默认 10）
    --prefix_index    指定展示第几个 prefix（从 0 开始），设此值则只画单图不画概览
    --region_key      Region 维度使用的字段，可选 city / postal_code / neighborhood（默认 city）

输出：
    {output_dir}/{exp}_prefix{len}.csv           # 统计表
    {output_dir}/{exp}_prefix{len}_overview.png # 概览图
    {output_dir}/{exp}_prefix{len}_i{n}.png     # 单 prefix 详图
"""

import argparse
import json
import os
from collections import defaultdict, Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===================== 解析函数 =====================

def parse_sid(sid):
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


def get_prefix(sid, n):
    p1, p2, p3, _ = parse_sid(sid)
    return '-'.join([p1, p2, p3][:n])


# ===================== 数据加载 =====================

def load_semantic_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_poi_meta(path):
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            meta[obj['business_id']] = obj
    return meta


# ===================== 分布计算 =====================

def compute_entropy(prefix_counts):
    """基于 prefix 频次计算信息熵（以 2 为底）"""
    import math
    total = sum(prefix_counts.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in prefix_counts.values() if count > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def compute_gini(prefix_counts):
    """计算 prefix 分布的 Gini 系数"""
    counts = sorted(prefix_counts.values())
    n = len(counts)
    if n == 0:
        return 0.0
    total_sum = sum(counts)
    gini_sum = sum((2 * i + 1) * c for i, c in enumerate(counts))
    return (2 * gini_sum / (n * total_sum)) - (n + 1) / n


def compute_prefix_distribution(semantic_ids, poi_meta, prefix_len, p1_filter=None, p2_filter=None, region_key='city'):
    """
    返回 DataFrame，含百分比和实际 Top 名称。
    列: Semantic_ID, Aspect, n_pois,
        Top1_name, Top1, Top2_name, Top2, ..., Others

    p1_filter: int or None  — 若指定，则只保留 P1 等于该值的 prefix
    p2_filter: int or None  — 若指定，则只保留 P2 等于该值的 prefix（需 prefix_len>=2）
    region_key: str         — Region 维度字段，可选 city / postal_code（默认 city）
    """
    groups = defaultdict(list)
    for bid, sid in semantic_ids.items():
        prefix = get_prefix(sid, prefix_len)
        groups[prefix].append(bid)

    rows = []

    def sort_key(item):
        parts = item[0].split('-')
        p1 = int(parts[0]) if parts[0].isdigit() else float('inf')
        p2 = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else float('inf')
        p3 = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else float('inf')
        return (p1, p2, p3)

    for prefix, bids in sorted(groups.items(), key=sort_key):
        # 过滤：排除只激活 P1（prefix_len>=2 时不含 '-'）
        if prefix_len >= 2 and '-' not in prefix:
            continue
        # 过滤：排除只激活 P2（prefix_len=3 时不含第三个 '-' 段）
        if prefix_len == 3 and prefix.count('-') < 2:
            continue
        # P1 过滤
        p1_val = prefix.split('-')[0]
        if p1_filter is not None and int(p1_val) != p1_filter:
            continue
        # P2 过滤
        if p2_filter is not None and prefix_len >= 2:
            p2_val = prefix.split('-')[1]
            if int(p2_val) != p2_filter:
                continue

        total = len(bids)
        if total < 1:
            continue
        total = len(bids)
        if total < 1:
            continue

        for aspect, key in [('Category', 'primary_category'), ('Region', region_key)]:
            counter = Counter(
                poi_meta[b].get(key, 'Unknown')
                for b in bids if b in poi_meta
            )
            sorted_items = counter.most_common(6)  # Top5 + Others
            names = [name for name, _ in sorted_items[:5]]
            counts = [count for _, count in sorted_items[:5]]
            while len(names) < 5:
                names.append('')
                counts.append(0)
            others = total - sum(counts)
            counts.append(max(0, others))

            row = {'Semantic_ID': prefix, 'Aspect': aspect, 'n_pois': total}
            for k in range(1, 6):
                row[f'Top{k}_name'] = names[k - 1]
                row[f'Top{k}'] = round(counts[k - 1] / total * 100, 1)
            row['Others'] = round(counts[5] / total * 100, 1)
            rows.append(row)

    return pd.DataFrame(rows)


# ===================== 单 Prefix 绘图 =====================

def plot_single_prefix(df_row, experiment, output_path):
    """
    参考图样式：只画一行 Category + 一行 Region，
    Y 轴显示实际名称（Top1_name）。
    df_row: 包含 Aspect in ['Category', 'Region'] 的两行 DataFrame
    """
    colors = ['#95cbee', '#fab0b0', '#b9dcb2', '#ccb0e6', '#ffe09e', '#e6e6e6']
    rank_keys = [('Top1', 'Top1_name'), ('Top2', 'Top2_name'),
                 ('Top3', 'Top3_name'), ('Top4', 'Top4_name'),
                 ('Top5', 'Top5_name'), ('Others', None)]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10

    cat_row = df_row[df_row['Aspect'] == 'Category'].iloc[0]
    reg_row = df_row[df_row['Aspect'] == 'Region'].iloc[0]
    prefix = cat_row['Semantic_ID']
    n_pois = cat_row['n_pois']

    # 构建两条 Y 轴标签（Category 用 Top1_name, Region 用 Top1_name）
    cat_labels = []
    cat_values = []
    for rank, name_key in rank_keys:
        name = cat_row.get(name_key, '') if name_key else 'Others'
        val = cat_row[rank]
        if val > 0:
            cat_labels.append(f"{name} ({val:.0f}%)" if name else f"Others ({val:.0f}%)")
            cat_values.append(val)

    reg_labels = []
    reg_values = []
    for rank, name_key in rank_keys:
        name = reg_row.get(name_key, '') if name_key else 'Others'
        val = reg_row[rank]
        if val > 0:
            reg_labels.append(f"{name} ({val:.0f}%)" if name else f"Others ({val:.0f}%)")
            reg_values.append(val)

    # 绘图：两条水平条，各自堆叠
    fig, (ax_cat, ax_reg) = plt.subplots(1, 2, figsize=(12, 4),
                                           gridspec_kw={'width_ratios': [1, 1]})
    plt.subplots_adjust(wspace=0.35)

    # Category 子图
    left = 0
    for j, (label, val) in enumerate(zip(cat_labels, cat_values)):
        ax_cat.barh(0, val, left=left, color=colors[j], height=0.5,
                    edgecolor='white', linewidth=0.5)
        if val > 8:
            ax_cat.text(left + val / 2, 0, f'{val:.1f}%',
                        va='center', ha='center', fontsize=8.5, color='white')
        left += val
    ax_cat.set_xlim(0, 100)
    ax_cat.set_yticks([0])
    ax_cat.set_yticklabels([f'C: {cat_row["Top1_name"]}'])
    ax_cat.set_xlabel('Percentage (%)')
    ax_cat.set_title(f'Category  (n={n_pois})', fontsize=10)
    ax_cat.spines['top'].set_visible(False)
    ax_cat.spines['right'].set_visible(False)

    # Region 子图
    left = 0
    for j, (label, val) in enumerate(zip(reg_labels, reg_values)):
        ax_reg.barh(0, val, left=left, color=colors[j], height=0.5,
                    edgecolor='white', linewidth=0.5)
        if val > 8:
            ax_reg.text(left + val / 2, 0, f'{val:.1f}%',
                        va='center', ha='center', fontsize=8.5, color='white')
        left += val
    ax_reg.set_xlim(0, 100)
    ax_reg.set_yticks([0])
    ax_reg.set_yticklabels([f'R: {reg_row["Top1_name"]}'])
    ax_reg.set_xlabel('Percentage (%)')
    ax_reg.set_title(f'Region  (n={n_pois})', fontsize=10)
    ax_reg.spines['top'].set_visible(False)
    ax_reg.spines['right'].set_visible(False)

    # 图例
    legend_names = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5', 'Others']
    legend_patches = [mpatches.Patch(color=c, label=n)
                      for c, n in zip(colors, legend_names)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=6,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{experiment}  |  Prefix = {prefix}', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ===================== 多 Prefix 概览图 =====================

def plot_overview(df, experiment, prefix_len, output_path, max_prefixes=10, region_key='city'):
    """
    按 semantic ID 分组画概览。
    Y 轴为 prefix 编号，条形内部显示 Top1~3 真值名称。
    """
    colors = ['#2e7fbf', '#c94a4a', '#5daa5d', '#8e6eb8', '#d4a855', '#aaaaaa']
    rank_keys = [
        ('Top1', 'Top1_name'), ('Top2', 'Top2_name'), ('Top3', 'Top3_name'),
        ('Top4', 'Top4_name'), ('Top5', 'Top5_name'), ('Others', None)
    ]

    unique_sids = df['Semantic_ID'].unique()[:max_prefixes]
    n_groups = len(unique_sids)
    if n_groups == 0:
        return

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8

    fig_height = max(4, n_groups * 0.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    y_positions = np.arange(n_groups) * 2.5

    # Y 轴标签：P2 代码，左侧标注 P1 分组（仅首次出现时显示）
    # prefix_len=1 时 Y 轴已显示完整 prefix，无需重复标注 P1
    if prefix_len == 1:
        p1_annotations = [''] * len(unique_sids)
    else:
        prev_p1 = None
        p1_annotations = []
        for sid in unique_sids:
            parts = str(sid).split('-')
            p1 = parts[0]
            if p1 != prev_p1:
                p1_annotations.append(f'P1={p1}')
                prev_p1 = p1
            else:
                p1_annotations.append('')

    ax.set_yticks(y_positions + 0.45)
    ax.set_yticklabels([f'({sid})' for sid in unique_sids], fontsize=9)

    # 在最左侧标注 P1 分组（跳过该分组首个 prefix，避免与 Y 轴标签重叠）
    for i, label in enumerate(p1_annotations):
        if label and i > 0:
            ax.text(-6, y_positions[i] + 0.45, label,
                    va='center', ha='right', fontsize=7.5,
                    color='#333333', fontweight='bold')

    for i, sid in enumerate(unique_sids):
        group = df[df['Semantic_ID'] == sid]

        # Category 行（上方），bar 稍宽
        cat_row = group[group['Aspect'] == 'Category'].iloc[0]
        left = 0
        for j, (val_key, name_key) in enumerate(rank_keys):
            val = cat_row[val_key]
            name = cat_row[name_key] if name_key else ''
            if val > 0:
                ax.barh(y_positions[i] + 1.0, val, left=left,
                        color=colors[j], height=1.2, edgecolor='white')
                # Top1~3 显示名称，Others 不显示文字
                if j < 3 and val > 5:
                    ax.text(left + val / 2, y_positions[i] + 1.0,
                            f'{name}\n{val:.0f}%',
                            va='center', ha='center', fontsize=7,
                            color='black', linespacing=1.2)
                left += val

        # Region 行（下方）
        reg_row = group[group['Aspect'] == 'Region'].iloc[0]
        left = 0
        for j, (val_key, name_key) in enumerate(rank_keys):
            val = reg_row[val_key]
            name = reg_row[name_key] if name_key else ''
            if val > 0:
                ax.barh(y_positions[i], val, left=left,
                        color=colors[j], height=1.0, edgecolor='white')
                if j < 3 and val > 5:
                    ax.text(left + val / 2, y_positions[i],
                            f'{name}\n{val:.0f}%',
                            va='center', ha='center', fontsize=7,
                            color='black', linespacing=1.2)
                left += val

    # C/R 侧标签
    for i in range(n_groups):
        ax.text(-5.5, y_positions[i] + 1.0, 'C', va='center', ha='right',
                fontsize=7.5, color='#555555')
        ax.text(-5.5, y_positions[i], 'R', va='center', ha='right',
                fontsize=7.5, color='#555555')

    ax.set_xlim(-5, 100)
    ax.set_xlabel('Percentage (%)', fontsize=10)
    ax.set_title(f'{experiment} — Prefix Length = {prefix_len} — Region={region_key}', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    legend_patches = [mpatches.Patch(color=c, label=n)
                      for c, n in zip(colors, ['Top 1', 'Top 2', 'Top 3',
                                                 'Top 4', 'Top 5', 'Others'])]
    ax.legend(handles=legend_patches, loc='lower right', ncol=6,
              fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    fig.subplots_adjust(left=0.12)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ===================== 入口 =====================

def main():
    parser = argparse.ArgumentParser(description='Semantic ID Prefix 分布分析')
    parser.add_argument('--data_dir', default='dataset/yelp/processed',
                        help='business_poi.json 所在目录')
    parser.add_argument('--output_dir', default='./analysis', help='输出目录')
    parser.add_argument('--experiments', nargs='+',
                        default=['PA_cosine_nowarmup'],
                        help='实验名称（对应 output/{exp}/semantic_ids.json）')
    parser.add_argument('--prefix_len', type=int, default=1,
                        choices=[1, 2, 3], help='前缀长度')
    parser.add_argument('--prefix_index', type=int, default=None,
                        help='指定展示第几个 prefix（从 0 开始，None 则展示概览图）')
    parser.add_argument('--max_prefixes', type=int, default=10,
                        help='概览图最多展示的 prefix 个数（默认 10）')
    parser.add_argument('--p1_filter', type=int, default=None,
                        help='只展示指定 P1 值（如 0、1、2 等）')
    parser.add_argument('--p2_filter', type=int, default=None,
                        help='只展示指定 P2 值（需配合 prefix_len>=2）')
    parser.add_argument('--region_key', choices=['city', 'postal_code', 'neighborhood'], default='city',
                        help='Region 维度使用的字段（默认 city）')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='输出 Coverage, Collision Rate, Entropy, Gini, PlusCode Rate 五项指标')
    args = parser.parse_args()

    poi_path = os.path.join(args.data_dir, 'business_poi.json')
    print(f"Loading POI metadata: {poi_path}")
    poi_meta = load_poi_meta(poi_path)
    print(f"  {len(poi_meta)} POI records loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    for exp in args.experiments:
        sid_path = os.path.join('output', exp, 'semantic_ids.json')
        print(f"\nExperiment: {exp}")

        if not os.path.exists(sid_path):
            print(f"  [WARN] Not found: {sid_path}")
            continue

        semantic_ids = load_semantic_ids(sid_path)
        print(f"  {len(semantic_ids)} semantic IDs.")

        n_disambig = sum(1 for s in semantic_ids.values() if '<' in s)
        print(f"  Disambig: {n_disambig} ({n_disambig/len(semantic_ids)*100:.1f}%)")

        if args.compute_metrics:
            # 计算 prefix 分布统计
            prefix_counts = defaultdict(int)
            pluscode_counts = 0
            for sid in semantic_ids.values():
                prefix = get_prefix(sid, args.prefix_len)
                prefix_counts[prefix] += 1
                if '<' in sid:
                    pluscode_counts += 1

            total_pois = len(semantic_ids)
            n_prefixes = len(prefix_counts)
            coverage = n_prefixes / total_pois * 100 if total_pois > 0 else 0

            # Collision Rate: 同一 prefix 下的 POI 数量方差（归一化）
            if n_prefixes > 0:
                counts = list(prefix_counts.values())
                mean_count = total_pois / n_prefixes
                collision_rate = sum((c - mean_count) ** 2 for c in counts) / n_prefixes / (mean_count ** 2) if mean_count > 0 else 0
            else:
                collision_rate = 0.0

            entropy = compute_entropy(prefix_counts)
            gini = compute_gini(prefix_counts)
            pluscode_rate = pluscode_counts / total_pois * 100 if total_pois > 0 else 0

            print(f"  Coverage: {coverage:.2f}%")
            print(f"  Collision Rate: {collision_rate:.4f}")
            print(f"  Entropy: {entropy:.4f}")
            print(f"  Gini: {gini:.4f}")
            print(f"  PlusCode Rate: {pluscode_rate:.2f}%")

        # CLI 参数名到 POI 实际字段名的映射
        region_field_map = {'neighborhood': 'plus_code_neighborhood'}
        region_field = region_field_map.get(args.region_key, args.region_key)

        df = compute_prefix_distribution(semantic_ids, poi_meta, args.prefix_len,
                                        p1_filter=args.p1_filter,
                                        p2_filter=args.p2_filter,
                                        region_key=region_field)
        print(f"  {len(df)//2} unique prefixes (prefix_len={args.prefix_len}, p1_filter={args.p1_filter}).")

        p1_suffix = f'_p1{args.p1_filter}' if args.p1_filter is not None else ''
        p2_suffix = f'_p2{args.p2_filter}' if args.p2_filter is not None else ''
        region_suffix = f'_region-{args.region_key}'
        csv_path = os.path.join(args.output_dir, f'{exp}_prefix{args.prefix_len}{p1_suffix}{p2_suffix}{region_suffix}.csv')
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")

        if args.prefix_index is not None:
            # 单 prefix 详图
            prefixes = df['Semantic_ID'].unique()
            if args.prefix_index >= len(prefixes):
                print(f"  [WARN] prefix_index={args.prefix_index} out of range (max={len(prefixes)-1})")
            else:
                selected = prefixes[args.prefix_index]
                df_sel = df[df['Semantic_ID'] == selected]
                out_png = os.path.join(args.output_dir,
                                       f'{exp}_prefix{args.prefix_len}{p1_suffix}{p2_suffix}{region_suffix}_i{args.prefix_index}.png')
                plot_single_prefix(df_sel, exp, out_png)
        else:
            # 概览图（最多 max_prefixes 个 prefix）
            df_plot = df.head(args.max_prefixes * 2)
            out_png = os.path.join(args.output_dir, f'{exp}_prefix{args.prefix_len}{p1_suffix}{p2_suffix}{region_suffix}_overview.png')
            plot_overview(df_plot, exp, args.prefix_len, out_png,
                          max_prefixes=args.max_prefixes, region_key=args.region_key)

    print(f"\nDone. Output: {args.output_dir}/")


if __name__ == '__main__':
    main()
