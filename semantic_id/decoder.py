"""
语义解码器 (Semantic Decoder)

实现向量重构与语义校验:
1. 从 Semantic ID 索引重构商户表征
2. 一致性检验: 验证相同前缀的项目具有高度相似性
3. 逆向映射: 从离散ID还原连续特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class MLPDecoder(nn.Module):
    """
    MLP 解码器
    
    将量化后的向量解码还原为原始特征空间
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [256, 512],
        output_dim: int = 256,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 选择激活函数
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        # 构建解码层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # 非最后一层
                if use_layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, input_dim] 量化向量
        
        Returns:
            reconstructed: [batch_size, output_dim] 重构向量
        """
        return self.decoder(z)


class SemanticDecoder(nn.Module):
    """
    语义解码器
    
    完整的解码模块，包含:
    1. 从索引重构量化向量
    2. 解码到原始特征空间
    3. 语义一致性校验
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dims: List[int] = [256, 512],
        output_dim: int = 256,
        num_quantizers: int = 3,
        codebook_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # MLP 解码器
        self.mlp_decoder = MLPDecoder(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(output_dim)
        
        # 移除 Sigmoid 激活，避免梯度消失和值域压缩
        # 使用线性输出以保持特征表达能力
    
    def forward(
        self,
        quantized: torch.Tensor,
        apply_activation: bool = False
    ) -> torch.Tensor:
        """
        Args:
            quantized: [batch_size, embedding_dim] 量化后的向量
            apply_activation: 是否应用输出激活（已弃用，保留接口兼容性）
        
        Returns:
            reconstructed: [batch_size, output_dim] 重构的特征向量
        """
        decoded = self.mlp_decoder(quantized)
        decoded = self.output_norm(decoded)
        
        # 不再使用 Sigmoid，保持线性输出
        return decoded
    
    def decode_from_indices(
        self,
        indices: torch.Tensor,
        codebooks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        从 Semantic ID 索引直接重构向量
        
        Args:
            indices: [batch_size, num_quantizers] 索引序列
            codebooks: 码本权重列表
        
        Returns:
            reconstructed: [batch_size, output_dim]
        """
        batch_size = indices.shape[0]
        device = indices.device
        
        # 累加各层码本向量
        quantized_sum = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        for i, codebook in enumerate(codebooks):
            # codebook: [codebook_size, embedding_dim]
            # indices[:, i]: [batch_size]
            quantized = F.embedding(indices[:, i], codebook)
            quantized_sum = quantized_sum + quantized
        
        # 解码
        return self.forward(quantized_sum)


class SemanticConsistencyChecker(nn.Module):
    """
    语义一致性校验器
    
    验证:
    1. 相同 Semantic ID 前缀的项目具有高度相似性
    2. 量化过程未丢失关键业务信息
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_quantizers: int = 3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
    
    @torch.no_grad()
    def check_prefix_similarity(
        self,
        indices: torch.Tensor,
        features: torch.Tensor,
        prefix_length: int = 1
    ) -> Dict[str, float]:
        """
        检查相同前缀的项目特征相似性
        
        Args:
            indices: [N, num_quantizers] Semantic ID
            features: [N, embedding_dim] 原始特征
            prefix_length: 前缀长度 (1, 2, 或 3)
        
        Returns:
            stats: 统计信息字典
        """
        # 提取前缀
        prefix = indices[:, :prefix_length]
        
        # 将前缀转为字符串键
        prefix_strings = [
            '-'.join(map(str, p.tolist())) 
            for p in prefix
        ]
        
        # 按前缀分组
        from collections import defaultdict
        groups = defaultdict(list)
        for i, ps in enumerate(prefix_strings):
            groups[ps].append(i)
        
        # 计算组内相似性
        intra_similarities = []
        inter_similarities = []
        
        for group_key, member_indices in groups.items():
            if len(member_indices) < 2:
                continue
            
            group_features = features[member_indices]
            
            # 组内余弦相似度
            group_norm = F.normalize(group_features, dim=-1)
            sim_matrix = torch.mm(group_norm, group_norm.t())
            
            # 取上三角 (不含对角线)
            triu_indices = torch.triu_indices(len(member_indices), len(member_indices), offset=1)
            intra_sim = sim_matrix[triu_indices[0], triu_indices[1]]
            intra_similarities.extend(intra_sim.tolist())
        
        # 计算组间相似性 (随机采样)
        all_indices = list(range(len(indices)))
        import random
        for _ in range(min(1000, len(indices))):
            i, j = random.sample(all_indices, 2)
            if prefix_strings[i] != prefix_strings[j]:
                sim = F.cosine_similarity(
                    features[i].unsqueeze(0),
                    features[j].unsqueeze(0)
                )
                inter_similarities.append(sim.item())
        
        return {
            'prefix_length': prefix_length,
            'num_groups': len(groups),
            'avg_group_size': len(indices) / len(groups) if groups else 0,
            'intra_similarity_mean': np.mean(intra_similarities) if intra_similarities else 0,
            'intra_similarity_std': np.std(intra_similarities) if intra_similarities else 0,
            'inter_similarity_mean': np.mean(inter_similarities) if inter_similarities else 0,
            'inter_similarity_std': np.std(inter_similarities) if inter_similarities else 0,
            'similarity_gap': (
                np.mean(intra_similarities) - np.mean(inter_similarities)
            ) if intra_similarities and inter_similarities else 0
        }
    
    @torch.no_grad()
    def compute_reconstruction_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算重构质量指标
        
        Args:
            original: [N, dim] 原始特征
            reconstructed: [N, dim] 重构特征
        
        Returns:
            metrics: 质量指标字典
        """
        # MSE
        mse = F.mse_loss(reconstructed, original).item()
        
        # MAE
        mae = F.l1_loss(reconstructed, original).item()
        
        # 余弦相似度
        cos_sim = F.cosine_similarity(original, reconstructed, dim=-1).mean().item()
        
        # R^2
        ss_res = ((original - reconstructed) ** 2).sum().item()
        ss_tot = ((original - original.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        
        return {
            'mse': mse,
            'mae': mae,
            'cosine_similarity': cos_sim,
            'r2_score': r2
        }


# 引入 numpy (用于统计)
import numpy as np
