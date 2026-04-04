"""
残差量化器 (Residual Quantizer)

实现基于级联码本的分层编码机制:
1. 一级粗粒度量化: 捕捉宏观特征（地理大区、核心业态）
2. 残差递归细化: 逐层量化残差，捕捉细节
3. 序列化输出: 生成定长离散序列 (i, j, k)

核心组件:
- VectorQuantizer: 单层向量量化器
- ResidualQuantizer: 多层残差量化器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from einops import rearrange


def kmeans(
    data: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
    use_cosine: bool = False
) -> torch.Tensor:
    """
    K-means 聚类用于码本初始化
    
    Args:
        data: [N, D] 数据点
        num_clusters: 聚类数量
        num_iters: 迭代次数
        use_cosine: 是否使用余弦相似度
    
    Returns:
        centroids: [num_clusters, D] 聚类中心
    """
    N, D = data.shape
    device = data.device
    
    # 随机选择初始中心
    indices = torch.randperm(N)[:num_clusters]
    centroids = data[indices].clone()
    
    for _ in range(num_iters):
        if use_cosine:
            # 余弦相似度
            data_norm = F.normalize(data, dim=-1)
            cent_norm = F.normalize(centroids, dim=-1)
            sim = torch.mm(data_norm, cent_norm.t())
            assignments = sim.argmax(dim=-1)
        else:
            # 欧氏距离
            dists = torch.cdist(data, centroids)
            assignments = dists.argmin(dim=-1)
        
        # 更新中心
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(num_clusters, device=device)
        
        for i in range(num_clusters):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(dim=0)
                counts[i] = mask.sum()
        
        # 处理空簇
        empty_mask = counts == 0
        if empty_mask.any():
            # 用随机数据点填充空簇
            empty_indices = torch.where(empty_mask)[0]
            num_empty = len(empty_indices)
            # 允许重复采样，确保有足够的数据点
            random_indices = torch.randint(0, N, (num_empty,), device=device)
            new_centroids[empty_indices] = data[random_indices]
        
        centroids = new_centroids
    
    return centroids


class VectorQuantizer(nn.Module):
    """
    向量量化器 (单层)
    
    将连续向量映射到最近的码本向量，使用 Straight-Through Estimator 反向传播
    
    Args:
        num_embeddings: 码本大小 (分支因子 b)
        embedding_dim: 嵌入维度
        commitment_cost: commitment loss 权重 (β)
        use_cosine: 是否使用余弦相似度
        kmeans_init: 是否使用 K-means 初始化
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        use_cosine: bool = False,
        kmeans_init: bool = True,
        kmeans_iters: int = 20,
        use_ema: bool = True,        # 使用 EMA 更新
        ema_decay: float = 0.95,     # EMA 衰减率（参考GNPR-SID）
        ema_eps: float = 1e-5,       # EMA epsilon
        use_projection: bool = False,  # 是否启用投影空间匹配
        projection_warmup_steps: int = 0,  # 投影匹配 warmup 步数
        projection_detach_codebook: bool = False  # 是否截断投影码本梯度
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_cosine = use_cosine
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.use_projection = use_projection
        self.projection_warmup_steps = projection_warmup_steps
        self.projection_detach_codebook = projection_detach_codebook
        self.register_buffer('_global_step', torch.zeros((), dtype=torch.long))
        
        # 码本嵌入
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if self.use_projection:
            self.codebook_projection = nn.Linear(embedding_dim, embedding_dim)
            nn.init.normal_(self.codebook_projection.weight, std=embedding_dim ** -0.5)
            if self.codebook_projection.bias is not None:
                nn.init.zeros_(self.codebook_projection.bias)
        
        # 初始化
        if not kmeans_init:
            # 均匀初始化
            self.embedding.weight.data.uniform_(
                -1.0 / num_embeddings,
                1.0 / num_embeddings
            )
            self._initialized = True
        else:
            self.embedding.weight.data.zero_()
            self._initialized = False
            # K-means 初始化缓冲区
            self._init_buffer = []
            self._init_samples_needed = num_embeddings * 2  # 至少需要 2*K 个样本
        
        # EMA 更新相关（参考 GNPR-SID V2）
        if use_ema:
            self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
            # EMA 模式下码本不需要梯度
            self.embedding.weight.requires_grad_(False)
        else:
            self.register_buffer('_ema_cluster_size', None)
            self.register_buffer('_ema_w', None)
    
    @torch.no_grad()
    def _init_embeddings(self, data: torch.Tensor):
        """使用 K-means 初始化码本（累积多个 batch）"""
        if self._initialized:
            return
        
        # 累积数据
        self._init_buffer.append(data.detach().cpu())
        accumulated_size = sum(buf.size(0) for buf in self._init_buffer)
        
        # 如果累积的样本足够，进行初始化
        if accumulated_size >= self._init_samples_needed:
            print(f"Initializing codebook with K-means ({self.num_embeddings} clusters, {accumulated_size} samples)...")
            all_data = torch.cat(self._init_buffer, dim=0).to(data.device)
            
            centroids = kmeans(
                all_data,
                self.num_embeddings,
                self.kmeans_iters,
                self.use_cosine
            )
            self.embedding.weight.data.copy_(centroids)
            if self.use_ema:
                self._ema_w.data.copy_(centroids)
            self._initialized = True
            
            # 清理缓冲区
            del self._init_buffer
            del self._init_samples_needed
    
    def forward(
        self,
        z: torch.Tensor,
        return_distances: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch_size, embedding_dim] 输入向量
            return_distances: 是否返回距离矩阵
        
        Returns:
            quantized: [batch_size, embedding_dim] 量化后的向量
            indices: [batch_size] 码本索引
            loss: 量化损失
        """
        # 初始化 (如果需要)
        if not self._initialized and self.training:
            self._init_embeddings(z)

        use_proj_now = (
            self.use_projection
            and self._initialized
            and (self._global_step.item() >= self.projection_warmup_steps)
        )

        if use_proj_now:
            z_match = self.codebook_projection(z)
            e_match = self.codebook_projection(self.embedding.weight)
            if self.projection_detach_codebook:
                e_match = e_match.detach()

            sim = torch.mm(
                F.normalize(z_match, dim=-1),
                F.normalize(e_match, dim=-1).t()
            )
            indices = sim.argmax(dim=-1)
            distances = 1.0 - sim

            # 量化向量来自原始码本，避免投影空间与重构空间强耦合
            e_raw = F.embedding(indices, self.embedding.weight)
            dot_product = torch.sum(z * e_raw, dim=-1, keepdim=True)
            norm_sq = torch.sum(e_raw * e_raw, dim=-1, keepdim=True)
            quantized = dot_product / (norm_sq + 1e-8) * e_raw

            commitment_loss = 1 - F.cosine_similarity(quantized.detach(), z, dim=-1)
            loss = self.commitment_cost * commitment_loss.mean()
        else:
            # 计算距离
            if self.use_cosine:
                # 余弦相似度
                z_norm = F.normalize(z, dim=-1)
                emb_norm = F.normalize(self.embedding.weight, dim=-1)
                distances = 1 - torch.mm(z_norm, emb_norm.t())
            else:
                # 欧氏距离 ||z - e||^2 = ||z||^2 - 2*z·e + ||e||^2
                distances = (
                    z.pow(2).sum(dim=-1, keepdim=True)
                    - 2 * torch.mm(z, self.embedding.weight.t())
                    + self.embedding.weight.pow(2).sum(dim=-1)
                )

            # 找最近的码本向量
            indices = distances.argmin(dim=-1)

            # 获取量化向量
            quantized = self.embedding(indices)

            # === 损失计算 ===
            if self.use_ema:
                # EMA 模式：只使用 commitment loss（参考GNPR-SID V2）
                commitment_loss = F.mse_loss(z, quantized.detach())
                loss = self.commitment_cost * commitment_loss
            else:
                # 标准模式：codebook loss + commitment loss
                commitment_loss = F.mse_loss(z, quantized.detach())
                codebook_loss = F.mse_loss(quantized, z.detach())
                loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-Through Estimator
        quantized = z + (quantized - z).detach()
        
        # === EMA 更新码本（仅训练时）===
        if self.use_ema and self.training and self._initialized:
            self._ema_update(z, indices)

        if self.training:
            self._global_step.add_(1)
        
        if return_distances:
            return quantized, indices, loss, distances
        
        return quantized, indices, loss
    
    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """
        EMA 更新码本（参考 GNPR-SID V2）
        
        Args:
            z: [B, D] 输入向量
            indices: [B] 码本索引
        """
        B = z.size(0)
        
        # 1. 统计每个码字的使用次数
        one_hot = F.one_hot(indices, self.num_embeddings).float()  # [B, K]
        cluster_size = one_hot.sum(dim=0)  # [K]
        
        # 2. EMA 更新 cluster_size
        self._ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        
        # 3. 累积每个簇的向量和
        dw = torch.zeros_like(self._ema_w)
        dw.index_add_(0, indices, z.to(device=self._ema_w.device, dtype=self._ema_w.dtype))
        
        # 4. EMA 更新 ema_w
        self._ema_w.mul_(self.ema_decay).add_(
            dw, alpha=1 - self.ema_decay
        )
        
        # 5. 更新 embedding 权重
        n = self._ema_cluster_size.unsqueeze(1).clamp(min=self.ema_eps)
        self.embedding.weight.data.copy_(self._ema_w / n)
        
        # 6. 死码重置（参考GNPR-SID V2）
        avg_usage = self._ema_cluster_size.mean()
        dead_threshold = avg_usage * 0.1  # 使用率低于平均10%即重置
        dead_indices = torch.where(self._ema_cluster_size < dead_threshold)[0]
        num_dead = dead_indices.numel()
        
        if num_dead > 0:
            if B == 0:
                return  # 避免空 batch
            
            # 用随机样本重置死码
            if B >= num_dead:
                sample_indices = torch.randperm(B, device=z.device)[:num_dead]
            else:
                sample_indices = torch.randint(0, B, (num_dead,), device=z.device)
            
            replace_samples = z[sample_indices].to(
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype
            )
            self.embedding.weight.data[dead_indices] = replace_samples
            self._ema_cluster_size[dead_indices] = 1.0
            self._ema_w[dead_indices] = replace_samples.to(
                device=self._ema_w.device,
                dtype=self._ema_w.dtype
            )
    
    @torch.no_grad()
    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """统计码本使用率"""
        counts = torch.bincount(indices.flatten(), minlength=self.num_embeddings)
        return counts.float() / counts.sum()
    
    def get_codebook(self) -> torch.Tensor:
        """返回码本权重"""
        return self.embedding.weight.data


class ResidualQuantizer(nn.Module):
    """
    残差量化器 (多层级联)
    
    实现论文中的分层编码机制:
    - 一级粗粒度量化
    - 残差递归细化
    - 生成定长离散序列
    
    Args:
        num_quantizers: 量化层数 (深度 M)
        codebook_size: 每层码本大小 (分支因子 b)
        embedding_dim: 嵌入维度
        commitment_cost: commitment loss 权重
        shared_codebook: 是否共享码本
    """
    
    def __init__(
        self,
        num_quantizers: int = 3,
        codebook_size: int = 64,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        use_cosine: bool = False,
        kmeans_init: bool = True,
        shared_codebook: bool = False,
        dropout: float = 0.0,
        use_ema: bool = True,         # 使用 EMA 更新（参考GNPR-SID）
        use_projection: bool = False,  # 是否启用投影空间匹配
        projection_warmup_steps: int = 0,  # 投影匹配 warmup 步数
        projection_detach_codebook: bool = False  # 是否截断投影码本梯度
    ):
        super().__init__()
        
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.shared_codebook = shared_codebook
        
        # 构建量化器
        if shared_codebook:
            # 共享码本
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    codebook_size, embedding_dim, commitment_cost,
                    use_cosine=use_cosine, kmeans_init=kmeans_init,
                    use_ema=use_ema,
                    use_projection=use_projection,
                    projection_warmup_steps=projection_warmup_steps,
                    projection_detach_codebook=projection_detach_codebook
                )
                for _ in range(1)  # 只创建一个
            ])
        else:
            # 独立码本
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    codebook_size, embedding_dim, commitment_cost,
                    use_cosine=use_cosine, kmeans_init=(kmeans_init and i == 0),
                    use_ema=use_ema,
                    use_projection=use_projection,
                    projection_warmup_steps=projection_warmup_steps,
                    projection_detach_codebook=projection_detach_codebook
                )
                for i in range(num_quantizers)
            ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        z: torch.Tensor,
        return_all_codes: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            z: [batch_size, embedding_dim] 输入向量
            return_all_codes: 是否返回所有层的量化向量
        
        Returns:
            quantized: [batch_size, embedding_dim] 最终量化向量 (各层累加)
            indices: [batch_size, num_quantizers] 各层索引
            total_loss: 总量化损失
            all_quantized: 可选, 各层量化向量列表
        """
        batch_size = z.shape[0]
        device = z.device
        
        # 初始化
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        all_quantized = [] if return_all_codes else None
        total_loss = 0.0  # 用浮点数累加，每层的 loss 已有梯度
        
        for i in range(self.num_quantizers):
            # 选择量化器
            quantizer = self.quantizers[0] if self.shared_codebook else self.quantizers[i]

            # 量化当前残差
            quantized, indices, loss = quantizer(residual)
            
            # 累加
            quantized_sum = quantized_sum + quantized
            total_loss = total_loss + loss
            all_indices.append(indices)
            
            if return_all_codes:
                all_quantized.append(quantized)
            
            # 计算新残差
            residual = residual - quantized
            residual = self.dropout(residual)
        
        # 堆叠索引
        indices = torch.stack(all_indices, dim=-1)  # [batch_size, num_quantizers]
        
        # 平均损失
        total_loss = total_loss / self.num_quantizers
        
        return quantized_sum, indices, total_loss, all_quantized
    
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从索引序列重构向量
        
        Args:
            indices: [batch_size, num_quantizers] 索引序列
        
        Returns:
            reconstructed: [batch_size, embedding_dim] 重构向量
        """
        batch_size = indices.shape[0]
        device = indices.device
        
        reconstructed = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        for i in range(self.num_quantizers):
            quantizer = self.quantizers[0] if self.shared_codebook else self.quantizers[i]
            quantized = quantizer.embedding(indices[:, i])
            reconstructed = reconstructed + quantized
        
        return reconstructed
    
    def get_all_codebooks(self) -> List[torch.Tensor]:
        """获取所有码本"""
        if self.shared_codebook:
            return [self.quantizers[0].get_codebook()] * self.num_quantizers
        return [q.get_codebook() for q in self.quantizers]
    
    def get_codebook_usage(self, indices: torch.Tensor) -> List[torch.Tensor]:
        """统计各层码本使用率"""
        usage = []
        for i in range(self.num_quantizers):
            quantizer = self.quantizers[0] if self.shared_codebook else self.quantizers[i]
            usage.append(quantizer.get_codebook_usage(indices[:, i]))
        return usage
    
    @property
    def total_codes(self) -> int:
        """总编码空间大小"""
        return self.codebook_size ** self.num_quantizers
    
    def indices_to_string(self, indices: torch.Tensor, separator: str = "-") -> List[str]:
        """
        将索引转换为字符串形式的 Semantic ID
        
        Args:
            indices: [batch_size, num_quantizers]
            separator: 分隔符
        
        Returns:
            sid_strings: ["i-j-k", ...] 语义ID字符串列表
        """
        indices_np = indices.cpu().numpy()
        return [separator.join(map(str, idx)) for idx in indices_np]


class ProductQuantizer(nn.Module):
    """
    乘积量化器 (可选的替代方案)
    
    将向量分割成多个子空间，在每个子空间独立量化
    与残差量化互补
    """
    
    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 64,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        assert embedding_dim % num_codebooks == 0
        
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.sub_dim = embedding_dim // num_codebooks
        
        # 每个子空间的量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, self.sub_dim, commitment_cost)
            for _ in range(num_codebooks)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch_size, embedding_dim]
        
        Returns:
            quantized: [batch_size, embedding_dim]
            indices: [batch_size, num_codebooks]
            loss: 量化损失
        """
        # 分割
        z_split = z.chunk(self.num_codebooks, dim=-1)
        
        quantized_list = []
        indices_list = []
        total_loss = 0.0
        
        for i, (z_sub, quantizer) in enumerate(zip(z_split, self.quantizers)):
            q, idx, loss = quantizer(z_sub)
            quantized_list.append(q)
            indices_list.append(idx)
            total_loss += loss
        
        quantized = torch.cat(quantized_list, dim=-1)
        indices = torch.stack(indices_list, dim=-1)
        
        return quantized, indices, total_loss / self.num_codebooks
