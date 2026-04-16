"""
仅构建 llm_finetune 数据缓存与 prompt 文件，不启动训练。
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import DATA_CONFIG
    from .dataset import prepare_datasets
except ImportError:
    from config import DATA_CONFIG
    from dataset import prepare_datasets


def _remove_cache_files(cache_dir: Path) -> None:
    targets = [
        cache_dir / 'train_samples.json',
        cache_dir / 'val_samples.json',
        cache_dir / 'test_samples.json',
        cache_dir / 'schema.txt',
        cache_dir / 'train_prompts.json',
        cache_dir / 'val_prompts.json',
        cache_dir / 'test_prompts.json',
    ]
    for path in targets:
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='只构建 llm_finetune 数据缓存与 prompts，不进入训练。'
    )
    parser.add_argument('--dataset_dir', type=str, default=None, help='数据目录（含 business_poi/user_active/review_poi）')
    parser.add_argument('--semantic_ids_path', type=str, required=True, help='semantic_ids.json 路径（可指向新 codebook）')
    parser.add_argument('--cache_dir', type=str, default=None, help='样本缓存输出目录')
    parser.add_argument('--prompt_export_dir', type=str, default=None, help='prompt 导出目录')
    parser.add_argument('--min_user_interactions', type=int, default=None, help='用户最小有效访问数阈值（默认取 config）')
    parser.add_argument('--strict_kcore', action='store_true', help='构建前执行严格 k-core 闭包过滤')
    parser.add_argument('--k_core', type=int, default=None, help='严格 k-core 的 k 值（默认取 config）')
    parser.add_argument('--k_core_output_dir', type=str, default=None, help='严格 k-core 输出目录（默认 dataset_dir/.cache/kcore_<k>）')
    parser.add_argument('--k_core_no_cache', action='store_true', help='严格 k-core 不复用缓存，强制重算')
    parser.add_argument('--no_export_prompts', action='store_true', help='不导出 *_prompts.json')
    parser.add_argument('--force_rebuild', action='store_true', help='删除已有缓存并强制重建')

    parser.add_argument('--gnpr_train_json', type=str, default=None, help='GNPR train JSON 路径（可选）')
    parser.add_argument('--gnpr_val_json', type=str, default=None, help='GNPR val JSON 路径（可选）')
    parser.add_argument('--gnpr_test_json', type=str, default=None, help='GNPR test JSON 路径（可选）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_dir:
        DATA_CONFIG['dataset_dir'] = str(Path(args.dataset_dir).resolve())
    DATA_CONFIG['semantic_ids_path'] = str(Path(args.semantic_ids_path).resolve())

    if args.cache_dir:
        DATA_CONFIG['cache_dir'] = str(Path(args.cache_dir).resolve())
    cache_dir = Path(DATA_CONFIG['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_export_dir:
        DATA_CONFIG['prompt_export_dir'] = str(Path(args.prompt_export_dir).resolve())

    if args.min_user_interactions is not None:
        DATA_CONFIG['min_user_interactions'] = max(2, int(args.min_user_interactions))

    DATA_CONFIG['enable_strict_kcore'] = bool(args.strict_kcore)
    if args.k_core is not None:
        DATA_CONFIG['k_core'] = int(args.k_core)
    if args.k_core_output_dir:
        DATA_CONFIG['k_core_output_dir'] = str(Path(args.k_core_output_dir).resolve())
    if args.k_core_no_cache:
        DATA_CONFIG['k_core_use_cache'] = False

    DATA_CONFIG['export_prompts_on_prepare'] = not args.no_export_prompts

    if args.gnpr_train_json and args.gnpr_val_json and args.gnpr_test_json:
        DATA_CONFIG['gnpr_data_paths'] = {
            'train': str(Path(args.gnpr_train_json).resolve()),
            'val': str(Path(args.gnpr_val_json).resolve()),
            'test': str(Path(args.gnpr_test_json).resolve()),
        }

    if args.force_rebuild:
        _remove_cache_files(cache_dir)

    train_dataset, val_dataset, test_dataset = prepare_datasets()
    print('\nData-only build finished.')
    print(f'Train: {len(train_dataset)}')
    print(f'Val:   {len(val_dataset)}')
    print(f'Test:  {len(test_dataset)}')
    print(f'Cache dir: {cache_dir}')


if __name__ == '__main__':
    main()
