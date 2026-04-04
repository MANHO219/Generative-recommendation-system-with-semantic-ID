"""
Semantic ID 生成模型 (RQVAE)

整合 Encoder-RQ-Decoder 的完整模型:
1. 多源特征对齐编码
2. 残差量化生成离散索引
3. 解码重构验证

核心类:
- RQVAE: 基础残差量化变分自编码器
- SemanticIDModel: 完整的Semantic ID生成模型（针对Yelp数据优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

from .encoder import MultiSourceEncoder, CategoryEncoder, PlusCodeEncoder
from .quantizer import ResidualQuantizer, VectorQuantizer
from .decoder import SemanticDecoder, SemanticConsistencyChecker


class RQVAE(nn.Module):
    """
    残差量化变分自编码器 (Residual Quantized VAE)
    
    基础模型架构:
    Input -> Encoder -> RQ -> Decoder -> Output
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        embedding_dim: 量化嵌入维度
        num_quantizers: 量化层数
        codebook_size: 码本大小
        commitment_cost: commitment loss 权重
        hidden_layers: 隐藏层配置
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        num_quantizers: int = 3,
        codebook_size: int = 64,
        commitment_cost: float = 0.25,
        hidden_layers: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        use_cosine: bool = False,
        kmeans_init: bool = True,
        use_ema: bool = True,
        use_projection: bool = False,
        projection_warmup_steps: int = 0,
        projection_detach_codebook: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # 编码器: Input -> Hidden
        encoder_dims = [input_dim] + hidden_layers + [embedding_dim]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if i < len(encoder_dims) - 2:
                encoder_layers.append(nn.LayerNorm(encoder_dims[i+1]))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 残差量化器
        self.quantizer = ResidualQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            use_cosine=use_cosine,
            kmeans_init=kmeans_init,
            use_ema=use_ema,
            use_projection=use_projection,
            projection_warmup_steps=projection_warmup_steps,
            projection_detach_codebook=projection_detach_codebook
        )
        
        # 解码器: Hidden -> Output
        decoder_dims = [embedding_dim] + hidden_layers[::-1] + [input_dim]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.LayerNorm(decoder_dims[i+1]))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)
        # 不需要输出激活：重构目标是 embedding 向量（有正有负）
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到连续空间"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从量化空间解码"""
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] 输入特征
            return_indices: 是否返回量化索引
        
        Returns:
            dict: {
                'reconstructed': 重构的输入,
                'quantized': 量化向量,
                'indices': Semantic ID 索引,
                'quant_loss': 量化损失
            }
        """
        # 编码
        z = self.encode(x)
        
        # 量化
        quantized, indices, quant_loss, _ = self.quantizer(z)
        
        # 解码
        reconstructed = self.decode(quantized)
        
        output = {
            'reconstructed': reconstructed,
            'quantized': quantized,
            'quant_loss': quant_loss
        }
        
        if return_indices:
            output['indices'] = indices
        
        return output
    
    @torch.no_grad()
    def get_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取 Semantic ID (推理模式)
        
        Args:
            x: [batch_size, input_dim]
        
        Returns:
            indices: [batch_size, num_quantizers] Semantic ID
        """
        z = self.encode(x)
        _, indices, _, _ = self.quantizer(z)
        return indices
    
    def compute_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
        recon_weight: float = 1.0,
        quant_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        Args:
            x: 原始输入
            output: 前向传播输出
            recon_weight: 重构损失权重
            quant_weight: 量化损失权重
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        # 重构损失 (MSE)
        recon_loss = F.mse_loss(output['reconstructed'], x)
        
        # 量化损失
        quant_loss = output['quant_loss']
        
        # 总损失
        total_loss = recon_weight * recon_loss + quant_weight * quant_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'quant_loss': quant_loss.item()
        }


class SemanticIDModel(nn.Module):
    """
    完整的 Semantic ID 生成模型
    
    针对 Yelp POI 数据优化的完整模型，整合:
    1. 多源特征编码 (类别、空间、时间、属性) → 拼接为稠密向量
    2. MLP Encoder: 将稠密向量压缩到 embedding_dim
    3. 残差量化
    4. MLP Decoder: 重构原始稠密向量
    
    架构遵循 GNPR-SID 论文:
    原始特征 → 拼接 → Encoder MLP → 编码向量 → RQ → 量化向量 → Decoder MLP → 重构特征
    
    Args:
        num_categories: 类别数量
        embedding_dim: 量化嵌入维度 (RQ 的工作维度)
        num_quantizers: 量化层数
        codebook_size: 码本大小
        pretrained_category_embeddings: 预训练类别嵌入
    """
    
    def __init__(
        self,
        num_categories: int,
        embedding_dim: int = 256,
        num_quantizers: int = 3,
        codebook_size: int = 64,
        input_dim: int = None,  # 输入特征维度（从 dataset 获取）
        hidden_layers: List[int] = [256, 128],
        commitment_cost: float = 0.25,
        dropout: float = 0.1,
        use_simple_encoder: bool = False,  # GNPR-SID V2: Linear(79→64)+LayerNorm+GELU
        **kwargs  # 忽略其他参数
    ):
        super().__init__()

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.use_simple_encoder = use_simple_encoder

        # 输入特征维度
        # GNPR-SID V2: 79 (64+3+12)
        # 原方案: num_categories + pluscode(10) + temporal(6) + star(1) + numerical(3)
        if input_dim is None:
            input_dim = 79 if use_simple_encoder else (num_categories + 10 + 6 + 1 + 3)
        self.input_dim = input_dim

        if use_simple_encoder:
            # GNPR-SID V2 风格：单层 Linear + LayerNorm + GELU
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, input_dim)
            )
        else:
            # 原方案：多层 MLP Encoder
            encoder_dims = [input_dim] + hidden_layers + [embedding_dim]
            encoder_layers = []
            for i in range(len(encoder_dims) - 1):
                encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
                if i < len(encoder_dims) - 2:
                    encoder_layers.append(nn.LayerNorm(encoder_dims[i+1]))
                    encoder_layers.append(nn.GELU())
                    encoder_layers.append(nn.Dropout(dropout))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_dims = [embedding_dim] + hidden_layers[::-1] + [input_dim]
            decoder_layers = []
            for i in range(len(decoder_dims) - 1):
                decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
                if i < len(decoder_dims) - 2:
                    decoder_layers.append(nn.LayerNorm(decoder_dims[i+1]))
                    decoder_layers.append(nn.GELU())
                    decoder_layers.append(nn.Dropout(dropout))
            self.decoder = nn.Sequential(*decoder_layers)

        # 残差量化器
        self.quantizer = ResidualQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            use_cosine=kwargs.get('use_cosine', False),
            kmeans_init=kwargs.get('kmeans_init', True),
            use_ema=kwargs.get('use_ema', True),
            use_projection=kwargs.get('use_projection', False),
            projection_warmup_steps=kwargs.get('projection_warmup_steps', 0),
            projection_detach_codebook=kwargs.get('projection_detach_codebook', False)
        )
    
    def forward(
        self,
        feature_vector: torch.Tensor,
        **kwargs  # 忽略其他参数（兼容旧接口）
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        架构遵循 GNPR-SID:
        1. feature_vector (原始特征拼接) → MLP Encoder → encoded (压缩)
        2. encoded → RQ → quantized
        3. quantized → MLP Decoder → reconstructed (无激活)
        4. Loss = MSE(reconstructed, feature_vector) + quant_loss
        
        Args:
            feature_vector: [batch_size, input_dim] 拼接后的原始特征向量
        
        Returns:
            dict: {
                'input': 输入特征（重构目标）,
                'encoded': 编码后的压缩向量,
                'quantized': 量化向量,
                'indices': Semantic ID,
                'reconstructed': 重构的特征,
                'quant_loss': 量化损失
            }
        """
        # Step 1: MLP Encoder - 压缩到 embedding_dim
        encoded = self.encoder(feature_vector)
        
        # Step 2: 残差量化
        quantized, indices, quant_loss, all_quantized = self.quantizer(encoded)
        
        # Step 3: MLP Decoder - 重构原始特征（无激活，embedding 有正有负）
        reconstructed = self.decoder(quantized)
        
        return {
            'input': feature_vector,      # 输入特征（重构目标）
            'encoded': encoded,           # 编码后的压缩向量
            'quantized': quantized,       # 量化后的向量
            'indices': indices,           # Semantic ID
            'reconstructed': reconstructed, # 重构的特征
            'quant_loss': quant_loss,     # 量化损失
            'all_quantized': all_quantized
        }
    
    @torch.no_grad()
    def get_semantic_ids(
        self,
        feature_vector: torch.Tensor,
        as_string: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        获取 Semantic ID (推理模式)
        
        Args:
            feature_vector: [batch_size, input_dim] 拼接后的原始特征向量
            as_string: 是否返回字符串格式
        
        Returns:
            indices: [batch_size, num_quantizers] 或 字符串列表
        """
        self.eval()
        
        encoded = self.encoder(feature_vector)
        _, indices, _, _ = self.quantizer(encoded)
        
        if as_string:
            return self.quantizer.indices_to_string(indices)
        
        return indices
    
    def compute_diversity_loss(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多样性损失（Diversity Loss）- 硬性算术距离约束 (Eq.12)
        
        针对长尾分布数据优化，使用硬性 L1 约束强制均匀分布。
        
        注意：不再使用额外的紧凑性约束 (Eq.13)，因为：
        1. VQ-VAE 的 commitment loss (Eq.10) 已提供足够的紧凑性
        2. 对于长尾数据（如 Restaurants 52k），保留适度松散性有助于
           LLM 区分细粒度特征（价格、口味等）
        3. 避免 L_utilize 和 L_compactness 的优化冲突
        
        硬性约束优势:
        - 强制扩容：打散头部类别(如 Restaurants 52k)的拥堵
        - 保护稀有数据：确保长尾类别(如 TX 4个)有独立表示空间
        - 消除梯度消失：L1 比熵损失梯度更稳定
        
        Args:
            indices: [batch_size, num_quantizers] 量化索引
        
        Returns:
            utilization_loss: 码本利用率损失 (L1)
        """
        batch_size = indices.size(0)
        num_quantizers = indices.size(1)
        
        utilization_losses = []
        
        # 目标：每个码字应被选择 N/K 次
        target_count = batch_size / self.codebook_size
        
        for q in range(num_quantizers):
            q_indices = indices[:, q]  # [batch_size]
            
            # 计算每个码字被选择的次数
            counts = torch.zeros(self.codebook_size, device=indices.device)
            counts.scatter_add_(0, q_indices, torch.ones_like(q_indices, dtype=torch.float))
            
            # L1 距离：|count_i - N/K|
            # 归一化到 [0, 1] 范围，除以 batch_size
            l1_distance = torch.abs(counts - target_count).sum() / batch_size
            
            utilization_losses.append(l1_distance)
        
        # 平均所有量化层的损失
        utilization_loss = torch.stack(utilization_losses).mean()
        
        return utilization_loss
    
    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        recon_weight: float = 1.0,
        quant_weight: float = 1.0,
        align_weight: float = 0.1,
        diversity_weight: float = 0.1,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        
        核心思想（参考 GNPR-SID 实现）：
        1. 重构损失: 重构原始特征拼接 (combined)，不是 aligned
        2. 量化损失: codebook_loss + commitment_loss
        3. 多样性损失: 延迟启用，等模型稳定后再加入（epoch >= 5）
        
        Args:
            output: 前向传播输出
            recon_weight: 重构损失权重
            quant_weight: 量化损失权重
            align_weight: 对齐损失权重（现在是 encoded 和 quantized 的相似性）
            diversity_weight: 码本利用率损失权重
            epoch: 当前 epoch 索引（用于延迟启用 diversity loss）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        # 重构损失: 重构原始特征拼接 (input)
        # 参考 GNPR-SID: loss_recon = MSE(decoded_output, original_input)
        recon_loss = F.mse_loss(output['reconstructed'], output['input'])
        
        # 量化损失 (包含 codebook_loss + commitment_loss)
        quant_loss = output['quant_loss']
        
        # 对齐一致性损失: 量化后仍保持与编码向量的相似性
        align_loss = 1 - F.cosine_similarity(
            output['quantized'], 
            output['encoded'],  # 改为 encoded
            dim=-1
        ).mean()
        
        # 利用率损失 - 参考实现中 epoch >= 1000 才启用
        # 在我们的设置中使用较小的阈值 (epoch >= 5)，让模型先学会重构
        if epoch >= 5:
            utilization_loss = self.compute_diversity_loss(indices=output['indices'])
        else:
            utilization_loss = torch.tensor(0.0, device=output['encoded'].device)
        
        # 总损失
        total_loss = (
            recon_weight * recon_loss + 
            quant_weight * quant_loss + 
            align_weight * align_loss +
            diversity_weight * utilization_loss
        )
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'quant_loss': quant_loss.item(),
            'align_loss': align_loss.item(),
            'utilization_loss': utilization_loss.item() if isinstance(utilization_loss, torch.Tensor) else 0.0
        }
    
    @torch.no_grad()
    def analyze_codebook_usage(
        self,
        indices: torch.Tensor
    ) -> Dict[str, Any]:
        """分析码本使用情况"""
        usage = self.quantizer.get_codebook_usage(indices)
        
        stats = {}
        for i, u in enumerate(usage):
            active_codes = (u > 0).sum().item()
            entropy = -(u * torch.log(u + 1e-10)).sum().item()
            max_usage = u.max().item()
            
            stats[f'level_{i}'] = {
                'active_codes': active_codes,
                'usage_entropy': entropy,
                'max_usage': max_usage,
                'utilization': active_codes / self.codebook_size
            }
        
        return stats
    
    @torch.no_grad()
    def get_codebooks(self) -> List[torch.Tensor]:
        """获取所有码本权重"""
        return self.quantizer.get_all_codebooks()
    
    def save_codebooks(self, path: str):
        """保存码本到文件"""
        codebooks = self.get_codebooks()
        torch.save({
            f'codebook_{i}': cb for i, cb in enumerate(codebooks)
        }, path)
    
    def load_codebooks(self, path: str):
        """从文件加载码本"""
        state = torch.load(path)
        for i, quantizer in enumerate(self.quantizer.quantizers):
            if f'codebook_{i}' in state:
                quantizer.embedding.weight.data.copy_(state[f'codebook_{i}'])


def resolve_sid_collisions(
    semantic_ids: Dict[str, str],
    pluscode_neighborhoods: Optional[Dict[str, str]] = None,
    suffix_mode: str = "grid"
) -> Dict[str, str]:
    """
    解决 Semantic ID 冲突
    
    当多个 POI 映射到相同的 SID 时，附加消歧后缀保证唯一性。
    - grid: 使用 plus_code_neighborhood 的最后 2 位，格式为 [GG] / [GG_n]
    - index: 兼容旧格式，使用 <d_n>
    
    Args:
        semantic_ids: {business_id: semantic_id_string}
        pluscode_neighborhoods: {business_id: plus_code_neighborhood}
        suffix_mode: 后缀模式，"grid" 或 "index"
    
    Returns:
        resolved_ids: {business_id: unique_semantic_id_string}
    """
    from collections import defaultdict

    if suffix_mode not in {"grid", "index"}:
        raise ValueError(
            f"Unsupported suffix_mode={suffix_mode}. Expected 'grid' or 'index'."
        )
    
    # 按 SID 分组
    sid_to_pois = defaultdict(list)
    for bid, sid in semantic_ids.items():
        sid_to_pois[sid].append(bid)
    
    # 解决冲突
    resolved_ids = {}
    collision_count = 0
    
    for sid, pois in sid_to_pois.items():
        if len(pois) == 1:
            # 无冲突
            resolved_ids[pois[0]] = sid
        else:
            # 有冲突，附加唯一标识符
            collision_count += len(pois)
            if suffix_mode == "grid":
                neighborhoods = pluscode_neighborhoods or {}
                grid_to_pois = defaultdict(list)

                for bid in pois:
                    neighborhood = (neighborhoods.get(bid, "") or "").strip()
                    grid_suffix = neighborhood[-2:].upper() if len(neighborhood) >= 2 else "UNK"
                    grid_to_pois[grid_suffix].append(bid)

                for grid_suffix, grouped_bids in grid_to_pois.items():
                    for i, bid in enumerate(grouped_bids):
                        suffix = f"[{grid_suffix}]" if i == 0 else f"[{grid_suffix}_{i}]"
                        resolved_ids[bid] = f"{sid}{suffix}"
            else:
                for i, bid in enumerate(pois):
                    resolved_ids[bid] = f"{sid}<d_{i}>"
    
    if collision_count > 0:
        print(
            f"解决了 {collision_count} 个 SID 冲突 "
            f"(涉及 {len([p for p in sid_to_pois.values() if len(p) > 1])} 个重复 SID, "
            f"suffix_mode={suffix_mode})"
        )
    
    return resolved_ids


def create_model(config: Dict) -> SemanticIDModel:
    """
    根据配置创建模型

    Args:
        config: 配置字典（可包含顶层 gnpr_v2 子字典）

    Returns:
        model: SemanticIDModel 实例
    """
    # 检测是否使用 GNPR-SID V2 风格
    gnpr_v2_config = config.get('gnpr_v2', {})
    use_gnpr_v2 = gnpr_v2_config.get('use_gnpr_v2_features', False)

    if use_gnpr_v2:
        use_attributes = gnpr_v2_config.get('use_attributes', False)
        input_dim = (
            gnpr_v2_config.get('category_dim', 64) +
            gnpr_v2_config.get('spatial_dim', 3) +
            gnpr_v2_config.get('temporal_dim', 12) +
            (gnpr_v2_config.get('attribute_dim', 0) if use_attributes else 0)
        )  # 默认 79 或 143（启用 attributes 时）
        embedding_dim = 64
    else:
        input_dim = config.get('input_dim', None)
        embedding_dim = config.get('embedding_dim', 256)

    return SemanticIDModel(
        num_categories=config.get('num_categories', 500),
        embedding_dim=embedding_dim,
        num_quantizers=config.get('num_quantizers', 3),
        codebook_size=config.get('codebook_size', 64),
        input_dim=input_dim,
        hidden_layers=config.get('hidden_layers', [256, 128]),
        commitment_cost=config.get('commitment_cost', 0.25),
        dropout=config.get('dropout', 0.1),
        use_simple_encoder=use_gnpr_v2,
        pretrained_category_embeddings=config.get('pretrained_category_embeddings', None)
    )
