"""
Semantic ID 生成模块

基于残差量化（Residual Quantization）的层次化语义索引构建系统

核心组件:
- encoder: 多源异构特征的语义对齐与融合
- quantizer: 残差量化器（级联码本）
- decoder: 向量重构与语义校验
- model: 完整的 RQVAE 模型
- dataset: 数据集加载器
- trainer: 训练器
- config: 配置管理
"""

# Model
from .model import SemanticIDModel, RQVAE, create_model, resolve_sid_collisions

# Encoder
from .encoder import (
    MultiSourceEncoder, 
    CategoryEncoder, 
    PlusCodeEncoder,
    TemporalEncoder,
    AttributeEncoder
)

# Quantizer
from .quantizer import ResidualQuantizer, VectorQuantizer, ProductQuantizer

# Decoder
from .decoder import SemanticDecoder, MLPDecoder, SemanticConsistencyChecker

# Dataset
from .dataset import (
    YelpPOIDataset, 
    YelpSequenceDataset,
    CategoryVocabulary,
    PlusCodeTokenizer,
    create_dataloaders,
    poi_collate_fn
)

# Trainer
from .trainer import SemanticIDTrainer, train_semantic_id_model

# Config
from .config import get_config, get_preset_config, save_config, DEFAULT_CONFIG

__all__ = [
    # Model
    'SemanticIDModel',
    'RQVAE',
    'create_model',
    'resolve_sid_collisions',
    
    # Encoder
    'MultiSourceEncoder',
    'CategoryEncoder',
    'PlusCodeEncoder',
    'TemporalEncoder',
    'AttributeEncoder',
    
    # Quantizer
    'ResidualQuantizer',
    'VectorQuantizer',
    'ProductQuantizer',
    
    # Decoder
    'SemanticDecoder',
    'MLPDecoder',
    'SemanticConsistencyChecker',
    
    # Dataset
    'YelpPOIDataset',
    'YelpSequenceDataset',
    'CategoryVocabulary',
    'PlusCodeTokenizer',
    'create_dataloaders',
    'poi_collate_fn',
    
    # Trainer
    'SemanticIDTrainer',
    'train_semantic_id_model',
    
    # Config
    'get_config',
    'get_preset_config',
    'save_config',
    'DEFAULT_CONFIG',
]
