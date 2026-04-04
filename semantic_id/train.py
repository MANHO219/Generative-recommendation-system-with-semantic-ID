"""
Semantic ID 模型训练脚本

用法:
    python train.py --data_dir ./data/yelp/processed
    python train.py --config config.yaml
    python train.py --preset large
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_id.config import get_config, get_preset_config, save_config
from semantic_id.model import create_model, SemanticIDModel
from semantic_id.dataset import YelpPOIDataset, create_dataloaders
from semantic_id.trainer import SemanticIDTrainer


def setup_logging(log_dir: str):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'train.log', encoding='utf-8')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Semantic ID 模型训练')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/yelp/processed',
                        help='处理后的数据目录')

    # 配置参数
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (JSON/YAML)')
    parser.add_argument('--preset', type=str, choices=['small', 'base', 'large'],
                        default='base', help='预设配置')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=None)
    parser.add_argument('--num_quantizers', type=int, default=None)
    parser.add_argument('--codebook_size', type=int, default=None)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # 其他
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--run_name', type=str, default=None,
                        help='运行名称，追加到所有输出路径（如州名 PA）')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = get_config(args.config)
    else:
        config = get_preset_config(args.preset)

    # 命令行参数覆盖配置
    config['data']['data_dir'] = args.data_dir

    if args.run_name:
        for key in ('log_dir', 'checkpoint_dir', 'output_dir'):
            config['paths'][key] = str(Path(config['paths'][key]) / args.run_name)

    if args.embedding_dim:
        config['model']['embedding_dim'] = args.embedding_dim
    if args.num_quantizers:
        config['model']['num_quantizers'] = args.num_quantizers
    if args.codebook_size:
        config['model']['codebook_size'] = args.codebook_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # 设置日志
    setup_logging(config['paths']['log_dir'])

    logging.info("=" * 50)
    logging.info("Semantic ID 模型训练")
    logging.info("=" * 50)

    # 保存配置
    save_config(config, Path(config['paths']['log_dir']) / 'config.json')

    # 设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA 不可用，使用 CPU")
        device = 'cpu'

    logging.info(f"设备: {device}")

    # 加载数据
    logging.info("加载数据...")
    geo_config = config.get('geo', {})
    gnpr_v2_config = config.get('gnpr_v2', {})

    # 检查是否使用 GNPR-SID V2 风格
    if gnpr_v2_config.get('use_gnpr_v2_features', False):
        logging.info("使用 GNPR-SID V2 风格数据集")
        from semantic_id.dataset import create_gnpr_dataloaders
        train_loader, val_loader, test_loader, poi_dataset = create_gnpr_dataloaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            train_ratio=0.8,
            val_ratio=0.1,
            geo_stratified=geo_config.get('geo_stratified_split', False),
            min_city_poi=geo_config.get('min_city_poi_count', 200),
            category_embeddings_path=gnpr_v2_config.get('category_embeddings_path'),
            use_sentence_transformer=gnpr_v2_config.get('use_sentence_transformer', False),
            sentence_model_name=gnpr_v2_config.get('sentence_model_name', 'all-MiniLM-L6-v2'),
            use_attributes=gnpr_v2_config.get('use_attributes', False),
            attribute_dim=gnpr_v2_config.get('attribute_dim', 64),
            use_attribute_cache=gnpr_v2_config.get('use_attribute_cache', True),
            use_attribute_projection=gnpr_v2_config.get('use_attribute_projection', True)
        )
    else:
        train_loader, val_loader, test_loader, poi_dataset = create_dataloaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training'].get('num_workers', 4),
            geo_stratified=geo_config.get('geo_stratified_split', True),
            min_city_poi=geo_config.get('min_city_poi_count', 200)
        )

    logging.info(f"训练集: {len(train_loader.dataset)} 样本")
    logging.info(f"验证集: {len(val_loader.dataset)} 样本")
    logging.info(f"测试集: {len(test_loader.dataset)} 样本")

    # 更新类别数
    config['model']['num_categories'] = len(poi_dataset.category_vocab)
    logging.info(f"类别数: {config['model']['num_categories']}")

    # GNPR-SID V2 维度对齐（以数据集真实维度为准）
    if gnpr_v2_config.get('use_gnpr_v2_features', False):
        gnpr_v2_config['category_dim'] = int(getattr(poi_dataset, 'category_dim', gnpr_v2_config.get('category_dim', 64)))
        gnpr_v2_config['spatial_dim'] = int(getattr(poi_dataset, 'spatial_dim', gnpr_v2_config.get('spatial_dim', 3)))
        gnpr_v2_config['temporal_dim'] = int(getattr(poi_dataset, 'temporal_dim', gnpr_v2_config.get('temporal_dim', 12)))
        gnpr_v2_config['attribute_dim'] = int(getattr(poi_dataset, 'attribute_dim', gnpr_v2_config.get('attribute_dim', 0)))
        logging.info(
            f"GNPR-SID V2 真实维度: cat={gnpr_v2_config['category_dim']}, "
            f"spatial={gnpr_v2_config['spatial_dim']}, temporal={gnpr_v2_config['temporal_dim']}, "
            f"attr={gnpr_v2_config['attribute_dim']}, total={getattr(poi_dataset, 'feature_dim', 'n/a')}"
        )

    # 获取预训练类别嵌入（仅原方案）
    if not gnpr_v2_config.get('use_gnpr_v2_features', False):
        if config['model'].get('use_pretrained_category', False):
            logging.info("加载预训练类别嵌入...")
            try:
                config['model']['pretrained_category_embeddings'] = \
                    poi_dataset.get_category_embeddings()
            except Exception as e:
                logging.warning(f"无法加载预训练嵌入: {e}")
                config['model']['pretrained_category_embeddings'] = None

    # 创建模型
    logging.info("创建模型...")
    # 将 gnpr_v2 配置合并到 model 配置中，供 create_model 检测是否使用简化编码器
    model_config = {**config['model'], 'gnpr_v2': gnpr_v2_config}
    model = create_model(model_config)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型参数: {num_params:,} (可训练: {num_trainable:,})")

    # 创建训练器
    trainer_config = {**config['training'], **config['paths']}
    trainer = SemanticIDTrainer(model, trainer_config, device)

    # 训练
    logging.info("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume
    )

    # 评估
    logging.info("评估模型...")
    geo_map = getattr(poi_dataset, 'geo_map', None)
    eval_results = trainer.evaluate_semantic_ids(test_loader, geo_map=geo_map)

    logging.info("评估结果:")
    logging.info(f"  唯一 ID 数: {eval_results['unique_ids']}")
    logging.info(f"  总 ID 数: {eval_results['total_ids']}")
    logging.info(f"  冲突率: {eval_results['collision_rate']:.4f}")

    if 'avg_collision@city' in eval_results:
        logging.info(f"  城市平均冲突率: {eval_results['avg_collision@city']:.4f}")
        logging.info(f"  城市最大冲突率: {eval_results['max_collision@city']:.4f}")

    for level, stats in eval_results['codebook_stats'].items():
        logging.info(f"  {level}: 利用率 = {stats['utilization']:.2%}")

    # 生成所有 Semantic ID
    logging.info("生成 Semantic ID...")
    from torch.utils.data import DataLoader

    # 根据数据类型选择 collate_fn
    if gnpr_v2_config.get('use_gnpr_v2_features', False):
        from semantic_id.dataset import gnpr_collate_fn
        collate_fn = gnpr_collate_fn
    else:
        from semantic_id.dataset import poi_collate_fn
        collate_fn = poi_collate_fn

    full_loader = DataLoader(
        poi_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    output_path = Path(config['paths']['output_dir']) / 'semantic_ids.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pluscode_neighborhoods = {
        bid: (biz.get('plus_code_neighborhood', '') if isinstance(biz, dict) else '')
        for bid, biz in getattr(poi_dataset, 'businesses', {}).items()
    }

    semantic_ids = trainer.generate_semantic_ids(
        full_loader,
        output_path=str(output_path),
        pluscode_neighborhoods=pluscode_neighborhoods,
        suffix_mode='grid'
    )

    logging.info(f"生成 {len(semantic_ids)} 个 Semantic ID")
    logging.info(f"保存到: {output_path}")

    # 保存码本
    codebook_path = Path(config['paths']['checkpoint_dir']) / 'codebooks.pt'
    model.save_codebooks(str(codebook_path))
    logging.info(f"码本保存到: {codebook_path}")

    logging.info("训练完成!")


if __name__ == '__main__':
    main()
