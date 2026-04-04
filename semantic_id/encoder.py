"""
多源异构特征的语义对齐与融合编码器

实现功能:
1. 商户属性对齐（POI Alignment）: Categories, Stars, Is_open
2. 空间上下文嵌入（Spatial Context Alignment）: Plus Codes
3. 时间特征嵌入（Temporal Context）: 时段分布
4. 用户偏好映射（User Preference Integration）: 可选的协同信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple


class CategoryEncoder(nn.Module):
    """
    类别语义编码器
    
    将文本类别转换为语义向量，支持:
    1. 预训练嵌入（如 Sentence-BERT）
    2. 可学习的类别嵌入
    """
    
    def __init__(
        self,
        num_categories: int,
        embedding_dim: int = 64,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_pretrained: bool = False
    ):
        super().__init__()
        
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        
        if pretrained_embeddings is not None:
            # 使用预训练嵌入（如 all-MiniLM-L6-v2 生成的）
            pretrained_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=freeze_pretrained
            )
            # 投影到目标维度
            self.projection = nn.Linear(pretrained_dim, embedding_dim)
        else:
            # 可学习嵌入
            self.embedding = nn.Embedding(num_categories, embedding_dim)
            self.projection = None
            # Xavier初始化
            nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            category_ids: [batch_size] 单标签索引
                       或 [batch_size, num_categories] 多标签索引（每个位置是类别ID）
        
        Returns:
            category_emb: [batch_size, embedding_dim]
        """
        if category_ids.dim() == 1:
            # 单标签: 直接查表
            emb = self.embedding(category_ids)
        else:
            # 多标签索引: [batch_size, num_categories] 
            # 每个位置是类别ID（不是one-hot），需要查表后平均
            batch_size, num_cats = category_ids.shape
            
            # 创建mask过滤padding (假设0是padding)
            mask = (category_ids > 0).float()  # [batch_size, num_cats]
            
            # 查表获取所有类别嵌入
            emb = self.embedding(category_ids)  # [batch_size, num_cats, emb_dim]
            
            # 加权平均 (忽略padding)
            mask = mask.unsqueeze(-1)  # [batch_size, num_cats, 1]
            emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [batch_size, emb_dim]
        
        if self.projection is not None:
            emb = self.projection(emb)
        
        return emb


class PlusCodeEncoder(nn.Module):
    """
    Plus Code 字符级空间编码器
    
    将 Plus Code 的字符序列编码为空间嵌入:
    - 输入: [batch_size, max_len] 字符索引序列 (vocab_size=22)
    - 输出: [batch_size, embedding_dim]
    
    使用位置编码来捕获空间层级信息:
    - 字符 0-1: 区域 (400km)
    - 字符 2-3: 城市 (20km)
    - 字符 4-5: 街区 (1km)
    - 字符 6-7: 精确位置 (50m)
    - 字符 8-9: 更精确位置
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        vocab_size: int = 22,  # 20字符 + padding + unknown
        max_len: int = 10,
        char_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # 字符嵌入
        self.char_embedding = nn.Embedding(vocab_size, char_embedding_dim, padding_idx=0)
        
        # 位置编码（可学习）- 捕获空间层级信息
        self.position_embedding = nn.Embedding(max_len, char_embedding_dim)
        
        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(max_len * char_embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.char_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def forward(self, pluscode_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pluscode_indices: [batch_size, max_len] 字符索引序列
        
        Returns:
            spatial_emb: [batch_size, embedding_dim]
        """
        batch_size = pluscode_indices.shape[0]
        
        # 字符嵌入
        char_emb = self.char_embedding(pluscode_indices)  # [batch_size, max_len, char_emb_dim]
        
        # 添加位置编码
        positions = torch.arange(self.max_len, device=pluscode_indices.device)
        pos_emb = self.position_embedding(positions)  # [max_len, char_emb_dim]
        char_emb = char_emb + pos_emb.unsqueeze(0)  # 广播相加
        
        # 展平并融合
        combined = char_emb.view(batch_size, -1)  # [batch_size, max_len * char_emb_dim]
        spatial_emb = self.fusion(combined)
        
        return spatial_emb


class TemporalEncoder(nn.Module):
    """
    时间特征编码器
    
    编码POI的时间活跃模式:
    - 工作日/周末分布
    - 时段分布 (早/午/晚/深夜)
    - 小时级分布 (可选)
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        use_hourly: bool = False
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_hourly = use_hourly
        
        # 基础特征维度: 2(工作日/周末) + 4(时段) = 6
        base_dim = 6
        
        if use_hourly:
            base_dim += 24  # 添加24小时分布
        
        self.encoder = nn.Sequential(
            nn.Linear(base_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embedding_dim)
        )
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [batch_size, feature_dim]
                包含: weekday_ratio, weekend_ratio, morning_ratio, 
                      afternoon_ratio, evening_ratio, night_ratio
                可选: 24小时分布
        
        Returns:
            temporal_emb: [batch_size, embedding_dim]
        """
        return self.encoder(temporal_features)


class AttributeEncoder(nn.Module):
    """
    商户属性编码器
    
    编码静态属性:
    - 星级评分 (1-5)
    - 营业状态 (is_open)
    - 评论数量 (review_count)
    - 其他数值属性
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        num_star_levels: int = 11,  # 0.5星级间隔: 0,1,1.5,2,...,5 共11级
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # 星级嵌入
        self.star_embedding = nn.Embedding(num_star_levels, embedding_dim // 2)
        
        # 其他数值特征编码
        self.numerical_encoder = nn.Sequential(
            nn.Linear(3, 32),  # is_open, review_count_norm, checkin_count_norm
            nn.ReLU(),
            nn.Linear(32, embedding_dim // 2)
        )
        
        # 融合层
        self.fusion = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(
        self,
        star_idx: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            star_idx: [batch_size] 星级索引 (0-9)
            numerical_features: [batch_size, 3] 数值特征
        
        Returns:
            attr_emb: [batch_size, embedding_dim]
        """
        star_emb = self.star_embedding(star_idx)
        num_emb = self.numerical_encoder(numerical_features)
        
        combined = torch.cat([star_emb, num_emb], dim=-1)
        return self.fusion(combined)


class MultiSourceEncoder(nn.Module):
    """
    多源异构特征对齐编码器

    整合所有特征源，生成统一的对齐向量 V_aligned
    用于后续的残差量化

    Args:
        use_geo_features: 是否融入地理嵌入（默认 False，阶段 B-1 开关）
        num_states:       state_vocab 大小（use_geo_features=True 时需要）
        num_cities:       city_vocab  大小（use_geo_features=True 时需要）
        state_dim:        州嵌入维度
        city_dim:         城市嵌入维度（同时是 GeoEncoder 输出维度）
    """

    def __init__(
        self,
        num_categories: int,
        output_dim: int = 256,
        category_dim: int = 64,
        spatial_dim: int = 64,
        temporal_dim: int = 32,
        attribute_dim: int = 32,
        pretrained_category_embeddings: Optional[np.ndarray] = None,
        dropout: float = 0.1,
        use_geo_features: bool = False,
        num_states: int = 0,
        num_cities: int = 0,
        state_dim: int = 32,
        city_dim: int = 64
    ):
        super().__init__()

        self.output_dim       = output_dim
        self.use_geo_features = use_geo_features

        # 各模态编码器
        self.category_encoder = CategoryEncoder(
            num_categories=num_categories,
            embedding_dim=category_dim,
            pretrained_embeddings=pretrained_category_embeddings
        )

        self.spatial_encoder = PlusCodeEncoder(
            embedding_dim=spatial_dim
        )

        self.temporal_encoder = TemporalEncoder(
            embedding_dim=temporal_dim
        )

        self.attribute_encoder = AttributeEncoder(
            embedding_dim=attribute_dim
        )

        # 可选地理编码器（阶段 B-1 开关）
        if use_geo_features and num_states > 0 and num_cities > 0:
            self.geo_encoder = GeoEncoder(
                num_states=num_states,
                num_cities=num_cities,
                state_dim=state_dim,
                city_dim=city_dim
            )
            geo_dim = city_dim
        else:
            self.geo_encoder = None
            geo_dim = 0

        # 特征维度
        total_dim = category_dim + spatial_dim + temporal_dim + attribute_dim + geo_dim

        # 多模态融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 可选: 用户偏好投影（协同信息）
        self.user_preference_projection = nn.Linear(output_dim, output_dim)
    
    def forward(
        self,
        category_ids: torch.Tensor,
        pluscode_indices: torch.Tensor,
        temporal_features: torch.Tensor,
        star_idx: torch.Tensor,
        numerical_features: torch.Tensor,
        user_preference: Optional[torch.Tensor] = None,
        return_combined: bool = False,
        state_id: Optional[torch.Tensor] = None,
        city_id:  Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            category_ids: [batch_size, num_categories] 类别one-hot
            pluscode_indices: [batch_size, 10] Plus Code字符索引
            temporal_features: [batch_size, 6] 时间特征
            star_idx: [batch_size] 星级索引
            numerical_features: [batch_size, 3] 数值属性
            user_preference: [batch_size, output_dim] 可选的用户偏好向量
            return_combined: 是否同时返回原始特征拼接（用于重构损失）
            state_id: [batch_size] 州索引（use_geo_features=True 时使用）
            city_id:  [batch_size] 城市桶索引（use_geo_features=True 时使用）

        Returns:
            aligned_vector: [batch_size, output_dim] 对齐后的融合向量
            或 (aligned_vector, combined): 如果 return_combined=True
        """
        # 编码各模态
        cat_emb     = self.category_encoder(category_ids)
        spatial_emb = self.spatial_encoder(pluscode_indices)
        temporal_emb = self.temporal_encoder(temporal_features)
        attr_emb    = self.attribute_encoder(star_idx, numerical_features)

        parts = [cat_emb, spatial_emb, temporal_emb, attr_emb]

        # 可选地理嵌入
        if self.geo_encoder is not None and state_id is not None and city_id is not None:
            geo_emb = self.geo_encoder(state_id, city_id)
            parts.append(geo_emb)

        # 拼接所有特征
        combined = torch.cat(parts, dim=-1)

        # 多模态融合
        aligned = self.fusion_layers(combined)

        # 可选: 融入用户偏好
        if user_preference is not None:
            user_proj = self.user_preference_projection(user_preference)
            aligned = aligned + 0.1 * user_proj  # 残差连接

        if return_combined:
            return aligned, combined
        return aligned
    
    def get_feature_dims(self) -> Dict[str, int]:
        """返回各模态的特征维度"""
        return {
            'category': self.category_encoder.embedding_dim,
            'spatial': self.spatial_encoder.embedding_dim,
            'temporal': self.temporal_encoder.embedding_dim,
            'attribute': self.attribute_encoder.embedding_dim,
            'output': self.output_dim
        }


class GeoEncoder(nn.Module):
    """
    地理层级编码器

    将州 (state) 和城市桶 (city bucket) 编码为地理嵌入向量，
    用于为 MultiSourceEncoder 提供可选的地理先验特征。

    Args:
        num_states: state_vocab 大小（含 PAD/UNK）
        num_cities: city_vocab 大小（含 PAD/UNK）
        state_dim:  州嵌入维度
        city_dim:   城市嵌入维度，同时也是输出维度
    """

    def __init__(
        self,
        num_states: int,
        num_cities: int,
        state_dim: int = 32,
        city_dim: int = 64
    ):
        super().__init__()
        self.state_dim = state_dim
        self.city_dim  = city_dim

        self.state_emb = nn.Embedding(num_states, state_dim, padding_idx=0)
        self.city_emb  = nn.Embedding(num_cities, city_dim,  padding_idx=0)
        self.fusion    = nn.Linear(state_dim + city_dim, city_dim)
        self.norm      = nn.LayerNorm(city_dim)

        nn.init.xavier_uniform_(self.state_emb.weight[1:])  # 跳过 PAD
        nn.init.xavier_uniform_(self.city_emb.weight[1:])

    def forward(
        self,
        state_id: torch.Tensor,
        city_id:  torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state_id: [batch_size] 州索引
            city_id:  [batch_size] 城市桶索引

        Returns:
            geo_emb: [batch_size, city_dim]
        """
        s_emb = self.state_emb(state_id)      # [B, state_dim]
        c_emb = self.city_emb(city_id)         # [B, city_dim]
        combined = torch.cat([s_emb, c_emb], dim=-1)  # [B, state_dim + city_dim]
        return self.norm(self.fusion(combined))
