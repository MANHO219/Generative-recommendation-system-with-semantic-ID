"""
Semantic ID 模型配置

包含默认配置和配置管理工具
"""

from typing import Dict, Any
from pathlib import Path
import json
import yaml


# 默认配置
DEFAULT_CONFIG = {
    # 数据配置
    'data': {
        'data_dir': './dataset/yelp/processed',
        'max_categories': 5,
        'include_checkins': True,
        'include_users': True
    },
    
    # 模型配置
    'model': {
        'num_categories': 500,           # 类别数量（会根据数据自动更新）
        'embedding_dim': 128,             # 主嵌入维度（增大以减少信息瓶颈，降低冲突率）
        'num_quantizers': 3,              # 残差量化层数
        'codebook_size': 128,       # 每层码本大小
        'category_dim': 64,               # 类别嵌入维度
        'spatial_dim': 64,                # 空间嵌入维度
        'temporal_dim': 32,               # 时间嵌入维度
        'attribute_dim': 32,              # 属性嵌入维度
        'hidden_layers': [512, 256, 128],      # 隐藏层配置（参考GNPR-SID）
        'commitment_cost': 0.25,           # VQ commitment loss 权重（参考GNPR-SID）
        'use_cosine': True,                # 量化匹配使用余弦相似度
        'use_ema': True,                   # 使用 EMA 更新码本（参考GNPR-SID V2，避免码本坍缩）
        'use_projection': True,            # 启用 projection-space cosine 匹配
        'projection_warmup_steps': 0, # warmup 期间回退到原始匹配
        'projection_detach_codebook': False,  # 可选稳定项：是否截断投影码本梯度
        'ema_decay': 0.95,                 # EMA 衰减率
        'dropout': 0.1,
        'use_pretrained_category': True   # 是否使用预训练类别嵌入
    },
    
    # 训练配置
    'training': {
        'learning_rate': 1e-3,            # 参考实现使用较高学习率
        'weight_decay': 1e-4,             # 参考GNPR-SID: 1e-4
        'epochs': 100,                     # 训练轮数 cpu环境建议20，gpu建议100
        'batch_size': 64,                 # 批大小 cpu环境建议16，gpu建议128
        'warmup_epochs': 5,
        'recon_weight': 1.0,              # 重构损失权重（参考实现默认1.0）
        'quant_weight': 0.5,              # 量化损失权重（参考GNPR-SID: 0.5）
        'align_weight': 0.0,              # 对齐损失权重（不使用，参考实现不需要）
        'diversity_weight': 0.0,          # 码本利用率损失权重（不使用，EMA机制自动维持利用率）
        # 注：GNPR-SID 使用 EMA + 死码重置机制自然维持码本利用率，无需显式 diversity loss
        # 我们使用 epoch >= 5 作为阈值（在 compute_loss 中控制）
        'diversity_warmup_epochs': 0,   # 多样性损失 warm-up 的 epoch 数（epoch >= 5 后生效）
        'diversity_weight_start': 0.0,   # 初始多样性权重
        'diversity_weight_end': 0.0,    # 最终多样性权重
        'use_amp': True,                  # 混合精度训练，支持检测启用
        'num_workers': 4,                 # GPU 环境建议 4-8，CPU 环境建议 0
        'save_every': 10                  # 每N个epoch保存检查点
    },
    
    # 路径配置
    'paths': {
        'log_dir': './logs/semantic_id',
        'checkpoint_dir': './checkpoints/semantic_id',
        'output_dir': './output'
    },
    
    # 评估配置
    'evaluation': {
        'eval_every': 5,
        'compute_metrics': True
    },

    # 地理分层配置
    'geo': {
        'geo_stratified_split': True,   # 阶段A开关：地理分层切分
        'min_city_poi_count': 200,      # 小城合并阈值
        'use_geo_features': False,      # 阶段B-1开关（默认关闭，不改变模型结构）
        'state_dim': 32,                # 州嵌入维度
        'city_dim': 64,                 # 城市嵌入维度
    },

    # GNPR-SID V2 特征配置
    'gnpr_v2': {
        'use_gnpr_v2_features': True,           # 启用 GNPR-SID V2 风格特征（方案B）
        'category_dim': 64,                     # 类别嵌入维度
        'spatial_dim': 3,                       # 球坐标维度（固定）
        'temporal_dim': 12,                     # Fourier 时间维度（固定）
        'attribute_dim': 64,                    # Attributes 文本嵌入维度（默认值，运行时可被真实维度覆盖）
        'use_attributes': True,                 # 是否启用 Attributes 文本嵌入
        'use_attribute_cache': True,            # 是否缓存 Attributes 嵌入到磁盘
        'use_attribute_projection': False,       # 是否将 attributes 嵌入投影到 attribute_dim
        'use_sentence_transformer': True,  # 是否使用 SentenceTransformer
        'sentence_model_name': 'all-MiniLM-L6-v2',
        'category_embeddings_path': None,       # 预训练类别嵌入路径（可选）
    }
}


def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取配置
    
    Args:
        config_path: 配置文件路径（JSON 或 YAML）
    
    Returns:
        config: 完整配置字典
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # 递归合并配置
            config = _merge_config(config, user_config)
    
    return config


def _merge_config(base: Dict, override: Dict) -> Dict:
    """递归合并配置"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config(config: Dict, parent_key: str = '') -> Dict[str, Any]:
    """
    将嵌套配置扁平化
    
    用于传递给模型和训练器
    """
    items = {}
    
    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key))
        else:
            items[key] = value
    
    return items


def get_model_config(config: Dict) -> Dict:
    """提取模型配置"""
    return config.get('model', {})


def get_training_config(config: Dict) -> Dict:
    """提取训练配置"""
    training = config.get('training', {})
    paths = config.get('paths', {})
    return {**training, **paths}


def save_config(config: Dict, path: str):
    """保存配置到文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(config, f, indent=2, ensure_ascii=False)


# 预定义配置模板
CONFIGS = {
    'small': {
        'model': {
            'embedding_dim': 128,
            'num_quantizers': 2,
            'codebook_size': 32,
            'hidden_layers': [128, 64]
        },
        'training': {
            'epochs': 50,
            'batch_size': 128
        }
    },
    
    'base': DEFAULT_CONFIG,
    
    'large': {
        'model': {
            'embedding_dim': 512,
            'num_quantizers': 4,
            'codebook_size': 128,
            'hidden_layers': [512, 256, 128]
        },
        'training': {
            'epochs': 200,
            'batch_size': 32
        }
    }
}


def get_preset_config(preset: str) -> Dict:
    """获取预设配置"""
    if preset not in CONFIGS:
        raise ValueError(f"未知预设: {preset}. 可用: {list(CONFIGS.keys())}")
    
    if preset == 'base':
        return DEFAULT_CONFIG.copy()
    
    return _merge_config(DEFAULT_CONFIG, CONFIGS[preset])
