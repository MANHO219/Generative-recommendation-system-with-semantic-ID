"""
Semantic ID 模型训练器

负责:
1. 训练循环管理
2. 损失计算与优化
3. 码本使用监控
4. 模型保存与恢复
5. Semantic ID 生成与导出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from collections import defaultdict
import time

from .model import SemanticIDModel, RQVAE, create_model
from .dataset import YelpPOIDataset, create_dataloaders


class SemanticIDTrainer:
    """
    Semantic ID 模型训练器
    
    Args:
        model: SemanticIDModel 实例
        config: 训练配置
        device: 训练设备
    """
    
    def __init__(
        self,
        model: SemanticIDModel,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.epochs = config.get('epochs', 100)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        
        # 损失权重（基础值）
        # 注意：不使用额外的 compactness_weight，因为 VQ-VAE 的 commitment loss 已提供足够紧凑性
        self.recon_weight = config.get('recon_weight', 1.0)
        self.quant_weight = config.get('quant_weight', 1.0)
        self.align_weight = config.get('align_weight', 0.1)
        self.diversity_weight = config.get('diversity_weight', 0.1)
        
        # 动态权重调整配置（针对长尾分布数据的 Warm-up 策略）
        # 多样性损失在训练初期权重较小，逐渐增大以打散头部类别
        self.diversity_warmup_epochs = config.get('diversity_warmup_epochs', 10)
        self.diversity_weight_start = config.get('diversity_weight_start', 0.01)
        self.diversity_weight_end = config.get('diversity_weight_end', self.diversity_weight)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # 日志
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'codebook_usage': []
        }
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        logging.info(f"Trainer 初始化完成，设备: {device}")
        logging.info(f"多样性损失 Warm-up: {self.diversity_weight_start} -> {self.diversity_weight_end} "
                     f"(前 {self.diversity_warmup_epochs} epochs)")
    
    def get_dynamic_diversity_weight(self, epoch: int) -> float:
        """
        计算动态多样性损失权重
        
        针对长尾分布数据的 Warm-up 策略：
        - 训练初期权重较小，允许模型先学习粗粒度聚类
        - 逐渐增大权重，强制打散头部类别的拥堵
        
        Args:
            epoch: 当前 epoch
        
        Returns:
            diversity_weight: 当前 epoch 的多样性损失权重
        """
        if epoch >= self.diversity_warmup_epochs:
            return self.diversity_weight_end
        
        # 线性插值
        progress = epoch / self.diversity_warmup_epochs
        weight = self.diversity_weight_start + progress * (self.diversity_weight_end - self.diversity_weight_start)
        
        return weight
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        # 获取当前 epoch 的动态多样性权重
        current_diversity_weight = self.get_dynamic_diversity_weight(epoch)
        
        total_loss = 0
        loss_components = defaultdict(float)
        all_indices = []
        
        # 重构质量指标
        total_mse = 0.0
        total_mae = 0.0
        total_cosine_sim = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} (div_w={current_diversity_weight:.4f})')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = self._to_device(batch)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    output = self.model(
                        feature_vector=batch['feature_vector']
                    )
                    loss, loss_dict = self.model.compute_loss(
                        output,
                        recon_weight=self.recon_weight,
                        quant_weight=self.quant_weight,
                        align_weight=self.align_weight,
                        diversity_weight=current_diversity_weight,
                        epoch=epoch
                    )
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    feature_vector=batch['feature_vector']
                )
                loss, loss_dict = self.model.compute_loss(
                    output,
                    recon_weight=self.recon_weight,
                    quant_weight=self.quant_weight,
                    align_weight=self.align_weight,
                    diversity_weight=current_diversity_weight,
                    epoch=epoch
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v
            
            # 收集 indices 用于分析码本
            all_indices.append(output['indices'].cpu())
            
            # 计算重构质量指标
            with torch.no_grad():
                input_vec = output['input']
                recon_vec = output['reconstructed']
                batch_size = input_vec.size(0)
                n_samples += batch_size
                
                # MSE: 每个元素的平方误差
                mse = F.mse_loss(recon_vec, input_vec, reduction='mean').item()
                total_mse += mse * batch_size
                
                # MAE: 每个元素的绝对误差
                mae = F.l1_loss(recon_vec, input_vec, reduction='mean').item()
                total_mae += mae * batch_size
                
                # 余弦相似度
                cos_sim = F.cosine_similarity(recon_vec, input_vec, dim=-1).sum().item()
                total_cosine_sim += cos_sim
                
                # 当前batch的平均指标
                batch_cos_sim = cos_sim / batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'quant': f"{loss_dict['quant_loss']:.4f}",
                'cos': f"{batch_cos_sim:.3f}"
            })
        
        # 平均损失
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        # 分析码本使用
        all_indices = torch.cat(all_indices, dim=0)
        codebook_stats = self.model.analyze_codebook_usage(all_indices)
        
        return {
            'loss': avg_loss,
            **avg_components,
            'codebook_stats': codebook_stats,
            # 重构质量指标
            'recon_mse': total_mse / n_samples,
            'recon_mae': total_mae / n_samples,
            'recon_cosine_sim': total_cosine_sim / n_samples
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0
        loss_components = defaultdict(float)
        all_indices = []
        
        # 重构质量指标
        total_mse = 0.0
        total_mae = 0.0
        total_cosine_sim = 0.0
        n_samples = 0
        
        for batch in val_loader:
            batch = self._to_device(batch)
            
            output = self.model(
                feature_vector=batch['feature_vector']
            )
            
            loss, loss_dict = self.model.compute_loss(
                output,
                recon_weight=self.recon_weight,
                quant_weight=self.quant_weight,
                align_weight=self.align_weight,
                diversity_weight=self.diversity_weight,
                epoch=9999  # 验证时始终启用diversity loss
            )
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v
            
            all_indices.append(output['indices'].cpu())
            
            # 计算重构质量指标
            input_vec = output['input']
            recon_vec = output['reconstructed']
            batch_size = input_vec.size(0)
            n_samples += batch_size
            
            mse = F.mse_loss(recon_vec, input_vec, reduction='mean').item()
            total_mse += mse * batch_size
            
            mae = F.l1_loss(recon_vec, input_vec, reduction='mean').item()
            total_mae += mae * batch_size
            
            cos_sim = F.cosine_similarity(recon_vec, input_vec, dim=-1).sum().item()
            total_cosine_sim += cos_sim
        
        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        all_indices = torch.cat(all_indices, dim=0)
        codebook_stats = self.model.analyze_codebook_usage(all_indices)
        
        return {
            'loss': avg_loss,
            **avg_components,
            'codebook_stats': codebook_stats,
            # 重构质量指标
            'recon_mse': total_mse / n_samples,
            'recon_mae': total_mae / n_samples,
            'recon_cosine_sim': total_cosine_sim / n_samples
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ):
        """
        完整训练流程
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logging.info(f"开始训练，共 {self.epochs} 个 epoch")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # 学习率 warmup
            if epoch < self.warmup_epochs:
                warmup_lr = self.learning_rate * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                self.scheduler.step()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            
            logging.info(
                f"Epoch {epoch}: "
                f"Train Loss = {train_metrics['loss']:.4f}, "
                f"Recon = {train_metrics['recon_loss']:.4f}, "
                f"Quant = {train_metrics['quant_loss']:.4f}, "
                f"Util = {train_metrics.get('utilization_loss', 0):.4f}"
            )
            logging.info(
                f"         "
                f"MSE = {train_metrics['recon_mse']:.6f}, "
                f"MAE = {train_metrics['recon_mae']:.6f}, "
                f"CosSim = {train_metrics['recon_cosine_sim']:.4f}"
            )
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                
                logging.info(
                    f"         Val Loss = {val_metrics['loss']:.4f}, "
                    f"MSE = {val_metrics['recon_mse']:.6f}, "
                    f"CosSim = {val_metrics['recon_cosine_sim']:.4f}"
                )
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # 定期保存
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # 记录码本使用
            self.history['codebook_usage'].append(train_metrics['codebook_stats'])
            
            # 日志码本利用率
            self._log_codebook_usage(train_metrics['codebook_stats'])
        
        # 训练完成，保存最终模型
        self.save_checkpoint('final_model.pt')
        self._save_history()
        
        logging.info("训练完成!")
    
    def _log_codebook_usage(self, stats: Dict):
        """记录码本使用情况"""
        for level, level_stats in stats.items():
            logging.info(
                f"  {level}: "
                f"活跃码字 = {level_stats['active_codes']}/{self.model.codebook_size}, "
                f"利用率 = {level_stats['utilization']:.2%}"
            )
    
    def _to_device(self, batch: Dict) -> Dict:
        """移动 batch 到设备"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logging.info(f"检查点保存到: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logging.info(f"从 {path} 恢复训练，当前 epoch = {self.current_epoch}")
    
    def _save_history(self):
        """保存训练历史"""
        history_path = self.log_dir / 'training_history.json'
        
        # 转换为可序列化格式
        serializable_history = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            # codebook_usage 需要特殊处理
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    @torch.no_grad()
    def generate_semantic_ids(
        self,
        data_loader: DataLoader,
        output_path: Optional[str] = None,
        resolve_collisions: bool = True,
        pluscode_neighborhoods: Optional[Dict[str, str]] = None,
        suffix_mode: str = "grid"
    ) -> Dict[str, str]:
        """
        为所有 POI 生成 Semantic ID
        
        Args:
            data_loader: 数据加载器
            output_path: 输出路径
            resolve_collisions: 是否解决 SID 冲突（添加唯一标识符）
            pluscode_neighborhoods: {business_id: plus_code_neighborhood}
            suffix_mode: 后缀模式，"grid" 或 "index"
        
        Returns:
            dict: {business_id: semantic_id_string}
        """
        from .model import resolve_sid_collisions
        
        self.model.eval()
        
        semantic_ids = {}
        
        for batch in tqdm(data_loader, desc="生成 Semantic ID"):
            batch = self._to_device(batch)
            business_ids = batch['business_ids']
            
            # 生成 Semantic ID
            ids = self.model.get_semantic_ids(
                feature_vector=batch['feature_vector'],
                as_string=True
            )
            
            for bid, sid in zip(business_ids, ids):
                semantic_ids[bid] = sid
        
        # 解决冲突（添加唯一标识符）
        if resolve_collisions:
            semantic_ids = resolve_sid_collisions(
                semantic_ids,
                pluscode_neighborhoods=pluscode_neighborhoods,
                suffix_mode=suffix_mode
            )
        
        # 保存
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_ids, f, ensure_ascii=False, indent=2)
            logging.info(f"Semantic ID 保存到: {output_path}")
        
        return semantic_ids
    
    @torch.no_grad()
    def evaluate_semantic_ids(
        self,
        data_loader: DataLoader,
        geo_map: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        评估 Semantic ID 质量

        包括:
        - 码本利用率
        - 冲突率 (相同 SID 的不同 POI)
        - 前缀相似性 (相似 POI 应有相同前缀)
        - [可选] 城市级别冲突率 (需要 geo_map)
        """
        self.model.eval()

        all_ids = []
        all_business_ids = []
        all_aligned = []

        for batch in data_loader:
            batch = self._to_device(batch)

            output = self.model(
                feature_vector=batch['feature_vector']
            )

            all_ids.append(output['indices'].cpu())
            all_business_ids.extend(batch['business_ids'])
            all_aligned.append(output['input'].cpu())  # 使用 input 作为特征表示

        all_ids = torch.cat(all_ids, dim=0)
        all_aligned = torch.cat(all_aligned, dim=0)

        # 1. 码本利用率
        codebook_stats = self.model.analyze_codebook_usage(all_ids)

        # 2. 冲突率
        id_strings = self.model.quantizer.indices_to_string(all_ids)
        unique_ids = set(id_strings)
        collision_rate = 1 - len(unique_ids) / len(id_strings)

        # 3. 前缀相似性分析
        prefix_similarity = self._compute_prefix_similarity(all_ids, all_aligned)

        metrics = {
            'codebook_stats': codebook_stats,
            'collision_rate': collision_rate,
            'unique_ids': len(unique_ids),
            'total_ids': len(id_strings),
            'prefix_similarity': prefix_similarity
        }

        # 4. 城市级冲突率（geo-aware 指标）
        if geo_map is not None:
            semantic_ids = dict(zip(all_business_ids, id_strings))
            city_groups: Dict[str, List[str]] = defaultdict(list)
            for biz_id, sid in semantic_ids.items():
                bucket = geo_map.get(biz_id, {}).get('bucket', '<UNK>')
                city_groups[bucket].append(sid)

            per_city_collision = {
                b: 1 - len(set(sids)) / len(sids)
                for b, sids in city_groups.items()
                if len(sids) > 0
            }

            if per_city_collision:
                metrics['avg_collision@city'] = float(
                    np.mean(list(per_city_collision.values()))
                )
                metrics['max_collision@city'] = float(
                    np.max(list(per_city_collision.values()))
                )
            else:
                metrics['avg_collision@city'] = 0.0
                metrics['max_collision@city'] = 0.0

        return metrics
    
    def _compute_prefix_similarity(
        self,
        indices: torch.Tensor,
        aligned: torch.Tensor
    ) -> Dict[str, float]:
        """计算前缀相似性"""
        n = len(indices)
        
        similarities = {}
        
        for prefix_len in range(1, indices.shape[1] + 1):
            # 按前缀分组
            prefix_groups = defaultdict(list)
            for i in range(n):
                prefix = tuple(indices[i, :prefix_len].tolist())
                prefix_groups[prefix].append(i)
            
            # 计算组内平均相似度
            group_sims = []
            for group_indices in prefix_groups.values():
                if len(group_indices) > 1:
                    group_vecs = aligned[group_indices]
                    # 计算组内所有对的余弦相似度
                    sim_matrix = torch.mm(
                        nn.functional.normalize(group_vecs, dim=1),
                        nn.functional.normalize(group_vecs, dim=1).T
                    )
                    # 取非对角线元素的平均
                    mask = ~torch.eye(len(group_indices), dtype=bool)
                    avg_sim = sim_matrix[mask].mean().item()
                    group_sims.append(avg_sim)
            
            if group_sims:
                similarities[f'prefix_{prefix_len}'] = np.mean(group_sims)
        
        return similarities


def train_semantic_id_model(
    data_dir: str,
    config: Optional[Dict] = None
):
    """
    训练 Semantic ID 模型的入口函数
    
    Args:
        data_dir: 处理后的 Yelp 数据目录
        config: 配置字典
    """
    # 默认配置
    default_config = {
        'num_categories': 500,
        'embedding_dim': 256,
        'num_quantizers': 3,
        'codebook_size': 64,
        'learning_rate': 1e-4,
        'epochs': 100,
        'batch_size': 64,
        'recon_weight': 1.0,
        'quant_weight': 1.0,
        'align_weight': 0.1,
        'use_amp': True,
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints'
    }
    
    if config:
        default_config.update(config)
    config = default_config
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用设备: {device}")
    
    # 数据
    train_loader, val_loader, test_loader, poi_dataset = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['batch_size']
    )
    
    # 更新类别数
    config['num_categories'] = len(poi_dataset.category_vocab)
    
    # 获取预训练类别嵌入 (可选)
    if config.get('use_pretrained_category', False):
        config['pretrained_category_embeddings'] = poi_dataset.get_category_embeddings()
    
    # 创建模型
    model = create_model(config)
    logging.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练器
    trainer = SemanticIDTrainer(model, config, device)
    
    # 训练
    trainer.train(train_loader, val_loader)
    
    # 评估
    eval_results = trainer.evaluate_semantic_ids(test_loader)
    logging.info(f"评估结果: {eval_results}")
    
    # 生成所有 Semantic ID
    all_loader = DataLoader(
        poi_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda b: {
            'business_ids': [x['business_id'] for x in b],
            'category_ids': torch.stack([x['category_ids'] for x in b]),
            'pluscode_indices': torch.stack([x['pluscode_indices'] for x in b]),
            'temporal_features': torch.stack([x['temporal_features'] for x in b]),
            'star_idx': torch.stack([x['star_idx'] for x in b]),
            'numerical_features': torch.stack([x['numerical_features'] for x in b]),
            'feature_vector': torch.stack([x['feature_vector'] for x in b])
        }
    )

    pluscode_neighborhoods = {
        bid: (biz.get('plus_code_neighborhood', '') if isinstance(biz, dict) else '')
        for bid, biz in poi_dataset.businesses.items()
    }
    
    semantic_ids = trainer.generate_semantic_ids(
        all_loader,
        output_path=Path(data_dir) / 'semantic_ids.json',
        pluscode_neighborhoods=pluscode_neighborhoods,
        suffix_mode='grid'
    )
    
    return model, semantic_ids
