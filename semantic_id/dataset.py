"""
Yelp POI 数据集加载器

从处理后的 JSON 文件加载数据并构建 PyTorch Dataset:
1. 加载 business_poi.json, checkin_poi.json, user_active.json
2. 构建类别词表和编码
3. 转换 Plus Code 为索引
4. 准备时间特征
"""

import json
import ast
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

try:
    from openlocationcode import openlocationcode as olc
except ImportError:
    olc = None
    print("警告: openlocationcode 未安装，Plus Code 功能将受限")


class PlusCodeTokenizer:
    """
    Plus Code 字符级分词器
    
    将 Plus Code 转换为字符索引序列:
    - 输入: "8FVC2222+22" (10位有效字符)
    - 输出: [idx_8, idx_F, idx_V, idx_C, idx_2, idx_2, idx_2, idx_2, idx_2, idx_2]
    
    字符表: 23456789CFGHJMPQRVWX (20个有效字符)
    vocab_size = 22 (20字符 + padding + unknown)
    """
    
    # Plus Code 字符集 (排除 AILO)
    ALPHABET = '23456789CFGHJMPQRVWX'
    CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 1-20, 0=padding
    
    def __init__(self, max_len: int = 10):
        self.max_len = max_len
        self.vocab_size = len(self.ALPHABET) + 2  # +1 padding, +1 unknown
        self.pad_idx = 0
        self.unk_idx = len(self.ALPHABET) + 1
    
    def encode(self, pluscode: str) -> List[int]:
        """
        编码 Plus Code 为字符索引列表
        
        Args:
            pluscode: 如 "8FVC2222+22"
        
        Returns:
            indices: [c0, c1, c2, ..., c9] 长度为 max_len
        """
        # 移除 '+' 符号和空格
        code = pluscode.replace('+', '').replace(' ', '')
        
        indices = []
        for i in range(self.max_len):
            if i < len(code):
                c = code[i]
                idx = self.CHAR_TO_IDX.get(c, self.unk_idx)
            else:
                idx = self.pad_idx  # Padding
            indices.append(idx)
        
        return indices
    
    def batch_encode(self, pluscodes: List[str]) -> torch.Tensor:
        """批量编码"""
        return torch.tensor([self.encode(pc) for pc in pluscodes], dtype=torch.long)


class CategoryVocabulary:
    """
    类别词表
    
    构建类别到索引的映射
    """
    
    def __init__(self, categories: Optional[List[str]] = None):
        self.category_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_category = {0: '<PAD>', 1: '<UNK>'}
        
        if categories:
            self.build(categories)
    
    def build(self, categories: List[str]):
        """从类别列表构建词表"""
        unique_categories = set()
        for cat in categories:
            if isinstance(cat, list):
                unique_categories.update(cat)
            else:
                unique_categories.add(cat)
        
        for cat in sorted(unique_categories):
            if cat not in self.category_to_idx:
                idx = len(self.category_to_idx)
                self.category_to_idx[cat] = idx
                self.idx_to_category[idx] = cat
    
    def encode(self, category: str) -> int:
        """类别编码"""
        return self.category_to_idx.get(category, 1)  # 1 = <UNK>
    
    def encode_multi(self, categories: List[str], max_len: int = 5) -> List[int]:
        """多类别编码（截断/填充）"""
        indices = [self.encode(c) for c in categories[:max_len]]
        while len(indices) < max_len:
            indices.append(0)  # Padding
        return indices
    
    def __len__(self):
        return len(self.category_to_idx)
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.category_to_idx, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.category_to_idx = json.load(f)
            self.idx_to_category = {v: k for k, v in self.category_to_idx.items()}


class YelpPOIDataset(Dataset):
    """
    Yelp POI 数据集
    
    加载处理后的数据并构建特征张量
    
    Args:
        data_dir: 处理后数据目录 (含 business_poi.json 等)
        max_categories: 最大类别数
        include_checkins: 是否包含签到数据
        include_users: 是否包含用户数据
    """
    
    def __init__(
        self,
        data_dir: str,
        max_categories: int = 5,
        include_checkins: bool = True,
        include_users: bool = True,
        category_vocab: Optional[CategoryVocabulary] = None
    ):
        self.data_dir = Path(data_dir)
        self.max_categories = max_categories
        self.include_checkins = include_checkins
        self.include_users = include_users
        
        # 加载数据
        self.businesses = self._load_json('business_poi.json')
        
        if include_checkins:
            self.checkins = self._load_checkins()
        else:
            self.checkins = {}
        
        if include_users:
            self.users = self._load_json('user_active.json')
        else:
            self.users = {}
        
        # 构建索引
        self.business_ids = list(self.businesses.keys())
        
        # 类别词表
        if category_vocab is not None:
            self.category_vocab = category_vocab
        else:
            self.category_vocab = CategoryVocabulary()
            all_categories = []
            for b in self.businesses.values():
                all_categories.extend(b.get('categories', []))
            self.category_vocab.build(all_categories)
        
        # Plus Code 分词器
        self.pluscode_tokenizer = PlusCodeTokenizer()

        # 地理分层词表
        self.geo_map, self.state_vocab, self.city_vocab = build_geo_partition(
            self.businesses
        )
        self.num_states = len(self.state_vocab)
        self.num_cities = len(self.city_vocab)

        logging.info(f"加载 {len(self.businesses)} 个 POI")
        logging.info(f"类别词表大小: {len(self.category_vocab)}")
        logging.info(f"地理词表: {self.num_states} 州, {self.num_cities} 城市桶")
    
    def _load_json(self, filename: str) -> Dict:
        """
        加载 JSON 文件
        
        支持两种格式:
        1. 标准 JSON: {"key": value, ...} 或 [...]
        2. JSON Lines: 每行一个 JSON 对象
        """
        path = self.data_dir / filename
        if not path.exists():
            logging.warning(f"文件不存在: {path}")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 尝试标准 JSON 解析
        try:
            data = json.loads(content)
            # 如果是列表，转换为字典（以 business_id 为 key）
            if isinstance(data, list):
                return {item.get('business_id', str(i)): item 
                        for i, item in enumerate(data) if isinstance(item, dict)}
            return data
        except json.JSONDecodeError:
            pass
        
        # 尝试 JSON Lines 格式（每行一个 JSON 对象）
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        # 使用 business_id 作为 key，如果没有则用行号
                        key = item.get('business_id', str(line_num))
                        data[key] = item
                except json.JSONDecodeError as e:
                    logging.warning(f"跳过无效行 {line_num}: {e}")
                    continue
        
        logging.info(f"从 {filename} 加载了 {len(data)} 条记录 (JSON Lines 格式)")
        return data
    
    def _load_checkins(self) -> Dict:
        """加载签到数据"""
        checkin_data = self._load_json('checkin_poi.json')
        
        # 如果是列表格式，转换为字典
        if isinstance(checkin_data, list):
            return {c['business_id']: c for c in checkin_data if 'business_id' in c}
        
        return checkin_data
    
    def __len__(self):
        return len(self.business_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个 POI 的特征
        
        Returns:
            dict: {
                'business_id': str,
                'category_ids': [max_categories],
                'pluscode_indices': [4],
                'temporal_features': [6],
                'star_idx': int,
                'numerical_features': [N]
            }
        """
        bid = self.business_ids[idx]
        business = self.businesses[bid]
        
        # 类别编码
        categories = business.get('categories', [])
        category_ids = self.category_vocab.encode_multi(categories, self.max_categories)
        
        # Plus Code 编码
        pluscode = business.get('pluscode', '')
        if pluscode:
            pluscode_indices = self.pluscode_tokenizer.encode(pluscode)
        else:
            # 从经纬度生成
            lat = business.get('latitude', 0)
            lng = business.get('longitude', 0)
            if olc and lat != 0 and lng != 0:
                pluscode = olc.encode(lat, lng, 10)
                pluscode_indices = self.pluscode_tokenizer.encode(pluscode)
            else:
                pluscode_indices = [0, 0, 0, 0]
        
        # 时间特征 (如果有签到数据)
        temporal_features = self._get_temporal_features(bid)
        
        # 星级
        stars = business.get('stars', 3.0)
        star_idx = int(min(max(stars * 2, 0), 10))  # 0-10 共11级
        
        # 数值特征
        numerical_features = self._get_numerical_features(business)
        
        # 构建完整的特征向量（用于直接拼接输入模型）
        # 参考 GNPR-SID: 直接拼接所有原始特征
        feature_vector = self._build_feature_vector(
            category_ids=category_ids,
            pluscode_indices=pluscode_indices,
            temporal_features=temporal_features,
            star_idx=star_idx,
            numerical_features=numerical_features
        )
        
        return {
            'business_id': bid,
            'category_ids': torch.tensor(category_ids, dtype=torch.long),
            'pluscode_indices': torch.tensor(pluscode_indices, dtype=torch.long),
            'temporal_features': torch.tensor(temporal_features, dtype=torch.float),
            'star_idx': torch.tensor(star_idx, dtype=torch.long),
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float),
            'feature_vector': torch.tensor(feature_vector, dtype=torch.float),  # 完整特征向量
            'state_id': torch.tensor(
                self.state_vocab.get(self.geo_map.get(bid, {}).get('state', ''), 1),
                dtype=torch.long
            ),
            'city_id': torch.tensor(
                self.city_vocab.get(self.geo_map.get(bid, {}).get('bucket', ''), 1),
                dtype=torch.long
            )
        }
    
    def _build_feature_vector(
        self,
        category_ids: List[int],
        pluscode_indices: List[int],
        temporal_features: List[float],
        star_idx: int,
        numerical_features: List[float]
    ) -> List[float]:
        """
        构建完整的特征向量（用于直接输入 MLP）
        
        参考 GNPR-SID 的数据格式：将所有特征拼接为一个稠密向量
        
        Returns:
            feature_vector: 拼接后的特征向量
        """
        features = []
        
        # 1. 类别 one-hot (num_categories 维)
        category_onehot = [0.0] * len(self.category_vocab)
        for cat_id in category_ids:
            if cat_id > 0:  # 跳过 padding
                category_onehot[cat_id] = 1.0
        features.extend(category_onehot)
        
        # 2. Plus Code 归一化 (10 维，每个字符归一化到 0-1)
        pluscode_norm = [idx / 22.0 for idx in pluscode_indices]  # vocab_size = 22
        features.extend(pluscode_norm)
        
        # 3. 时间特征 (6 维，已经是 0-1)
        features.extend(temporal_features)
        
        # 4. 星级归一化 (1 维)
        star_norm = star_idx / 10.0
        features.append(star_norm)
        
        # 5. 数值特征 (3 维，已归一化)
        features.extend(numerical_features)
        
        # Min-Max 归一化到 [0, 1]（保留相对差异，避免 L2 归一化导致的过度相似）
        # L2 归一化会让所有向量在单位球面上，导致编码空间利用率低
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # 处理全零向量（避免除零）
        feat_min = features_tensor.min()
        feat_max = features_tensor.max()
        
        if feat_max - feat_min > 1e-6:
            features_norm = (features_tensor - feat_min) / (feat_max - feat_min)
        else:
            features_norm = features_tensor  # 如果所有值相同，保持不变
        
        return features_norm.tolist()
    
    def get_feature_dim(self) -> int:
        """获取特征向量的维度"""
        # category_vocab_size + pluscode(10) + temporal(6) + star(1) + numerical(3)
        return len(self.category_vocab) + 10 + 6 + 1 + 3

    def get_geo_vocab(self) -> Tuple[Dict, Dict]:
        """返回 (state_vocab, city_vocab)"""
        return self.state_vocab, self.city_vocab
    
    def _get_temporal_features(self, business_id: str) -> List[float]:
        """
        获取时间分布特征
        
        Returns:
            [morning_ratio, afternoon_ratio, evening_ratio, night_ratio, 
             weekday_ratio, weekend_ratio]
        """
        if business_id not in self.checkins:
            return [0.25, 0.25, 0.25, 0.25, 0.7, 0.3]  # 默认均匀分布
        
        checkin = self.checkins[business_id]
        
        # 时段分布
        time_dist = checkin.get('time_distribution', {})
        morning = time_dist.get('morning', 0.25)
        afternoon = time_dist.get('afternoon', 0.25)
        evening = time_dist.get('evening', 0.25)
        night = time_dist.get('night', 0.25)
        
        # 工作日/周末
        weekday = checkin.get('weekday_ratio', 0.7)
        weekend = 1 - weekday
        
        return [morning, afternoon, evening, night, weekday, weekend]
    
    def _get_numerical_features(self, business: Dict) -> List[float]:
        """
        获取数值特征
        
        Returns:
            [review_count_log, avg_rating, is_open, ...]
        """
        review_count = business.get('review_count', 0)
        review_count_log = np.log1p(review_count) / 10.0  # 归一化
        
        stars = business.get('stars', 3.0) / 5.0  # 归一化到 0-1
        
        is_open = float(business.get('is_open', 1))
        
        # 可扩展更多属性
        return [review_count_log, stars, is_open]
    
    def get_category_embeddings(
        self, 
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    ) -> np.ndarray:
        """
        使用预训练模型获取类别语义嵌入
        
        Returns:
            embeddings: [vocab_size, embedding_dim]
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embedding_model)
            
            categories = [self.category_vocab.idx_to_category[i] 
                          for i in range(len(self.category_vocab))]
            
            embeddings = model.encode(categories, show_progress_bar=True)
            
            return embeddings
        
        except ImportError:
            logging.warning("sentence-transformers 未安装，使用随机嵌入")
            return np.random.randn(len(self.category_vocab), 384).astype(np.float32)


class YelpSequenceDataset(Dataset):
    """
    Yelp 用户访问序列数据集

    用于训练推荐模型，构建用户的 POI 访问序列

    Args:
        geo_filter:       若 True 则按城市桶过滤历史序列（同城优先）
        fallback_to_state: geo_filter 时，同城历史不足则回退到同州
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 50,
        min_seq_len: int = 5,
        poi_dataset: Optional[YelpPOIDataset] = None,
        geo_filter: bool = True,
        fallback_to_state: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.geo_filter = geo_filter
        self.fallback_to_state = fallback_to_state

        # POI 数据集
        if poi_dataset is not None:
            self.poi_dataset = poi_dataset
        else:
            self.poi_dataset = YelpPOIDataset(data_dir)

        # 构建 POI ID 到索引的映射
        self.poi_to_idx = {bid: i for i, bid in enumerate(self.poi_dataset.business_ids)}

        # geo_map（共用 POI 数据集中已构建好的）
        self.geo_map: Dict = getattr(self.poi_dataset, 'geo_map', {})

        # 加载用户序列
        self.sequences = self._load_sequences()

        logging.info(f"加载 {len(self.sequences)} 个用户序列")
    
    def _load_sequences(self) -> List[Dict]:
        """加载用户访问序列（支持 geo_filter）"""
        sequences = []

        # 从 review 数据构建序列
        review_path = self.data_dir / 'review_filtered.json'

        if not review_path.exists():
            logging.warning("review_filtered.json 不存在，尝试从 user_active.json 构建")
            return sequences

        # 按用户聚合评论
        user_reviews: Dict[str, List[Dict]] = defaultdict(list)

        with open(review_path, 'r', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line.strip())
                user_id     = review.get('user_id')
                business_id = review.get('business_id')
                date        = review.get('date', '')

                if business_id in self.poi_to_idx:
                    user_reviews[user_id].append({
                        'business_id': business_id,
                        'poi_idx':     self.poi_to_idx[business_id],
                        'date':        date
                    })

        # 构建序列
        for user_id, reviews in user_reviews.items():
            if len(reviews) < self.min_seq_len:
                continue

            # 按日期排序
            reviews.sort(key=lambda x: x['date'])

            if not self.geo_filter or not self.geo_map:
                # 原逻辑：不过滤，直接截断
                if len(reviews) > self.max_seq_len:
                    reviews = reviews[-self.max_seq_len:]
                sequences.append({
                    'user_id':      user_id,
                    'poi_sequence': [r['poi_idx']     for r in reviews],
                    'business_ids': [r['business_id'] for r in reviews]
                })
            else:
                # geo_filter 模式：以最后一个 POI 为 target，过滤历史
                target_review  = reviews[-1]
                target_bid     = target_review['business_id']
                target_time    = target_review['date']
                target_bucket  = self.geo_map.get(target_bid, {}).get('bucket', '')
                target_state   = self.geo_map.get(target_bid, {}).get('state', '')

                history = reviews[:-1]

                # 1. 同城历史（时间 < target_time 已由 reviews 排序保证）
                same_city = [
                    r for r in history
                    if self.geo_map.get(r['business_id'], {}).get('bucket', '') == target_bucket
                ]

                # 2. 不足时回退到同州
                if len(same_city) < self.min_seq_len and self.fallback_to_state:
                    same_city = [
                        r for r in history
                        if self.geo_map.get(r['business_id'], {}).get('state', '') == target_state
                    ]

                # 3. 仍不足则跳过
                if len(same_city) < self.min_seq_len:
                    continue

                # 截断 + 加回 target
                if len(same_city) > self.max_seq_len - 1:
                    same_city = same_city[-(self.max_seq_len - 1):]
                geo_seq = same_city + [target_review]

                sequences.append({
                    'user_id':      user_id,
                    'poi_sequence': [r['poi_idx']     for r in geo_seq],
                    'business_ids': [r['business_id'] for r in geo_seq]
                })

        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取用户序列
        
        Returns:
            dict: {
                'user_id': str,
                'input_seq': [seq_len-1],
                'target': [1] (最后一个),
                'seq_len': int
            }
        """
        seq_data = self.sequences[idx]
        poi_seq = seq_data['poi_sequence']
        
        # 输入序列 (除最后一个)
        input_seq = poi_seq[:-1]
        target = poi_seq[-1]
        
        # Padding
        padded_seq = input_seq + [0] * (self.max_seq_len - 1 - len(input_seq))
        
        return {
            'user_id': seq_data['user_id'],
            'input_seq': torch.tensor(padded_seq, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_len': torch.tensor(len(input_seq), dtype=torch.long)
        }


def build_geo_partition(
    business_list: Dict,
    min_city_poi: int = 200
) -> Tuple[Dict, Dict, Dict]:
    """
    按 state/city 统计 POI 数，小城市合并至 {state}_other 桶

    Args:
        business_list: {business_id: business_dict} 来自 business_poi.json
        min_city_poi:  POI 数小于此值的城市归入 {state}:_other 桶

    Returns:
        geo_map:    {business_id: {'state': str, 'city': str, 'bucket': str}}
        state_vocab:{state: idx}   0=PAD, 1=UNK, 2+ 为各州
        city_vocab: {bucket: idx}  0=PAD, 1=UNK, 2+ 为各桶
    """
    # 统计每个 state:city 组合的 POI 数
    city_counts: Dict[str, int] = defaultdict(int)
    for biz in business_list.values():
        state = biz.get('state', '') or ''
        city  = biz.get('city', '')  or ''
        key   = f"{state}:{city}"
        city_counts[key] += 1

    # 构建 geo_map：小城市合并至 {state}:_other
    geo_map: Dict[str, Dict] = {}
    for bid, biz in business_list.items():
        state  = biz.get('state', '') or ''
        city   = biz.get('city', '')  or ''
        key    = f"{state}:{city}"
        if city_counts[key] >= min_city_poi:
            bucket = key
        else:
            bucket = f"{state}:_other"
        geo_map[bid] = {'state': state, 'city': city, 'bucket': bucket}

    # 建 state_vocab
    all_states  = sorted({v['state']  for v in geo_map.values()})
    state_vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
    for s in all_states:
        if s not in state_vocab:
            state_vocab[s] = len(state_vocab)

    # 建 city_vocab（以 bucket 为单位）
    all_buckets = sorted({v['bucket'] for v in geo_map.values()})
    city_vocab: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
    for b in all_buckets:
        if b not in city_vocab:
            city_vocab[b] = len(city_vocab)

    return geo_map, state_vocab, city_vocab


def geo_stratified_split(
    dataset,
    geo_map: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    对每个城市桶内的索引按比例切 train/val/test

    确保每个桶内比例一致，避免某城市只出现在 test。

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)

    # 按 bucket 分组索引
    bucket_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        bid    = dataset.business_ids[idx]
        bucket = geo_map.get(bid, {}).get('bucket', '<UNK>')
        bucket_to_indices[bucket].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for bucket, indices in bucket_to_indices.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # 保证 test 至少 1 个（当 n >= 3）
        if n >= 3:
            n_test = n - n_train - n_val
            if n_test < 1:
                n_val  = max(1, n_val - 1)
                n_test = n - n_train - n_val
        else:
            # 极小城市：全给 train
            n_val  = 0
            n_test = 0

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    # 最终 shuffle 一下各集合（打乱桶顺序）
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    geo_stratified: bool = True,
    min_city_poi: int = 200
) -> Tuple[DataLoader, DataLoader, DataLoader, YelpPOIDataset]:
    """
    创建训练/验证/测试数据加载器

    Args:
        geo_stratified: 若 True 则使用地理分层切分（推荐），否则退回 random_split
        min_city_poi:   POI 数小于此值的城市归入 {state}:_other 桶

    Returns:
        train_loader, val_loader, test_loader, poi_dataset
    """
    # 创建 POI 数据集
    poi_dataset = YelpPOIDataset(data_dir)

    # 划分数据集
    if geo_stratified and hasattr(poi_dataset, 'geo_map') and poi_dataset.geo_map:
        train_indices, val_indices, test_indices = geo_stratified_split(
            poi_dataset, poi_dataset.geo_map,
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        train_set = torch.utils.data.Subset(poi_dataset, train_indices)
        val_set   = torch.utils.data.Subset(poi_dataset, val_indices)
        test_set  = torch.utils.data.Subset(poi_dataset, test_indices)
        logging.info(
            f"地理分层切分: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
        )
    else:
        n_total = len(poi_dataset)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)
        n_test  = n_total - n_train - n_val
        train_set, val_set, test_set = torch.utils.data.random_split(
            poi_dataset, [n_train, n_val, n_test]
        )
        logging.info(
            f"随机切分: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
        )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=poi_collate_fn
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=poi_collate_fn
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=poi_collate_fn
    )
    
    return train_loader, val_loader, test_loader, poi_dataset


def poi_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    POI 数据批处理函数
    """
    business_ids = [item['business_id'] for item in batch]

    result = {
        'business_ids': business_ids,
        'category_ids': torch.stack([item['category_ids'] for item in batch]),
        'pluscode_indices': torch.stack([item['pluscode_indices'] for item in batch]),
        'temporal_features': torch.stack([item['temporal_features'] for item in batch]),
        'star_idx': torch.stack([item['star_idx'] for item in batch]),
        'numerical_features': torch.stack([item['numerical_features'] for item in batch]),
        'feature_vector': torch.stack([item['feature_vector'] for item in batch])  # 完整特征向量
    }

    # 地理字段（可能不存在于 Subset 返回的老格式 batch 中，做安全判断）
    if 'state_id' in batch[0]:
        result['state_id'] = torch.stack([item['state_id'] for item in batch])
    if 'city_id' in batch[0]:
        result['city_id'] = torch.stack([item['city_id'] for item in batch])

    return result


# ============================================================
# GNPR-SID V2 风格数据集（方案B）
# 特征构成: category_emb + spatial_3d + fourier_time (+ optional attr_emb)
# ============================================================

class GNPRSIDPOIDataset(Dataset):
    """
    GNPR-SID V2 风格 POI 数据集

    特征构成:
    - category_emb: 64维（可学习嵌入 或 SentenceTransformer）
    - spatial_3d:    3维（球坐标 lat/lon → 3D）
    - fourier_time: 12维（Fourier 时间特征，6频率 × sin/cos）
    - attr_emb: 可选属性文本嵌入（默认64维，最终以实际编码维度为准）
    """

    def __init__(
        self,
        data_dir: str,
        category_embeddings_path: str = None,
        use_sentence_transformer: bool = False,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        use_attributes: bool = False,
        attribute_dim: int = 64,
        use_attribute_cache: bool = True,
        use_attribute_projection: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.use_attributes = use_attributes
        self.use_sentence_transformer = use_sentence_transformer
        self.sentence_model_name = sentence_model_name
        self.use_attribute_cache = use_attribute_cache
        self.use_attribute_projection = use_attribute_projection

        logging.info("加载 business_poi.json...")
        self.businesses = self._load_jsonl('business_poi.json')
        self.business_ids = list(self.businesses.keys())
        logging.info(f"加载了 {len(self.business_ids)} 个 POI")

        logging.info("加载 review_poi.json 构建时间分布...")
        self.time_distribution = self._load_time_distribution()

        logging.info("构建类别词汇表...")
        self._build_category_vocab()

        self.category_dim = 64
        self.spatial_dim = 3
        self.temporal_dim = 12
        self.attribute_dim = attribute_dim if self.use_attributes else 0
        self.feature_dim = self.category_dim + self.spatial_dim + self.temporal_dim + self.attribute_dim

        # 地理分层词表（供 geo_stratified_split 使用）
        self.geo_map, self.state_vocab, self.city_vocab = build_geo_partition(
            self.businesses
        )

        # 类别嵌入
        self.category_embeddings = None
        if use_sentence_transformer:
            self._load_sentence_embeddings(sentence_model_name)
        elif category_embeddings_path:
            self._load_category_embeddings(category_embeddings_path)

        # Attributes 文本语义嵌入（可选）
        self.attributes_text: Optional[List[str]] = None
        self.attr_embeddings: Optional[np.ndarray] = None
        self.attr_embedding_source: Optional[str] = None
        if self.use_attributes:
            self.attributes_text = self._load_attributes_text()
            self._load_attributes_embeddings(
                model_name=sentence_model_name,
                use_cache=use_attribute_cache
            )
            self.attribute_dim = int(self.attr_embeddings.shape[1])
            self.feature_dim = self.category_dim + self.spatial_dim + self.temporal_dim + self.attribute_dim
            logging.info(
                f"Attributes 已启用: source={self.attr_embedding_source}, "
                f"shape={self.attr_embeddings.shape}, feature_dim={self.feature_dim}"
            )

    # ---- 数据加载 ----

    def _load_jsonl(self, filename: str) -> Dict:
        path = self.data_dir / filename
        if not path.exists():
            logging.warning(f"文件不存在: {path}")
            return {}
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        key = item.get('business_id', str(line_num))
                        data[key] = item
                except json.JSONDecodeError:
                    continue
        logging.info(f"从 {filename} 加载了 {len(data)} 条记录")
        return data

    def _load_time_distribution(self) -> Dict[str, Dict[int, int]]:
        """从 review_poi.json 聚合每个 POI 的小时访问分布 {hour: count}"""
        time_dist: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        review_path = self.data_dir / 'review_poi.json'
        if not review_path.exists():
            return {}
        with open(review_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    bid = item.get('business_id')
                    date_str = item.get('date', '')
                    if bid and date_str:
                        try:
                            hour = int(date_str[11:13])
                            time_dist[bid][hour] += 1
                        except (ValueError, IndexError):
                            pass
                except json.JSONDecodeError:
                    continue
        logging.info(f"构建了 {len(time_dist)} 个 POI 的时间分布")
        return {bid: dict(dist) for bid, dist in time_dist.items()}

    def _build_category_vocab(self):
        """构建类别词汇表"""
        category_set = set()
        for biz in self.businesses.values():
            cats = biz.get('categories', '') or ''
            for cat in cats.split(', '):
                cat = cat.strip()
                if cat:
                    category_set.add(cat)
        self.categories = sorted(list(category_set))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

        # 兼容 train.py 读取 category_vocab 的调用
        self.category_vocab = self.category_to_idx

        logging.info(f"类别词汇表大小: {len(self.categories)}")

    # ---- 特征提取 ----

    def _get_spatial_3d(self, biz: Dict) -> np.ndarray:
        """lat/lon → 3D 球坐标"""
        lat = float(biz.get('latitude') or 0.0)
        lon = float(biz.get('longitude') or 0.0)
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        x = np.cos(lat_r) * np.cos(lon_r)
        y = np.cos(lat_r) * np.sin(lon_r)
        z = np.sin(lat_r)
        return np.array([x, y, z], dtype=np.float32)

    def _get_fourier_time(self, bid: str) -> np.ndarray:
        """Fourier 时间特征（12维：6频率 × sin/cos）"""
        dist = self.time_distribution.get(bid, {})
        total = sum(dist.values()) or 1
        # 加权平均小时
        mean_hour = sum(h * c for h, c in dist.items()) / total if dist else 12.0
        features = []
        for k in range(1, 7):
            features.append(np.sin(2 * np.pi * k * mean_hour / 24))
            features.append(np.cos(2 * np.pi * k * mean_hour / 24))
        return np.array(features, dtype=np.float32)

    def _get_category_embedding(self, biz: Dict) -> np.ndarray:
        """类别嵌入（64维）"""
        cats = [c.strip() for c in (biz.get('categories') or '').split(', ') if c.strip()]
        if self.category_embeddings is not None:
            # 使用预训练嵌入，取类别平均
            vecs = []
            for cat in cats:
                idx = self.category_to_idx.get(cat)
                if idx is not None and idx < len(self.category_embeddings):
                    vecs.append(self.category_embeddings[idx])
            if vecs:
                return np.mean(vecs, axis=0).astype(np.float32)
        # 回退：用 idx 的 one-hot 均值压缩到 64 维（简单哈希）
        vec = np.zeros(self.category_dim, dtype=np.float32)
        for cat in cats:
            idx = self.category_to_idx.get(cat, 0)
            vec[idx % self.category_dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ---- 预训练嵌入加载 ----

    def _load_sentence_embeddings(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            logging.info(f"加载 SentenceTransformer: {model_name}")
            model = SentenceTransformer(model_name)
            embeddings = model.encode(self.categories, show_progress_bar=True)
            if embeddings.shape[1] != self.category_dim:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.category_dim)
                embeddings = pca.fit_transform(embeddings)
            self.category_embeddings = embeddings.astype(np.float32)
            logging.info(f"SentenceTransformer 嵌入形状: {self.category_embeddings.shape}")
        except ImportError:
            logging.warning("sentence_transformers 未安装，回退到哈希嵌入")

    def _load_category_embeddings(self, path: str):
        try:
            data = np.load(path)
            self.category_embeddings = data.astype(np.float32)
            logging.info(f"加载预训练类别嵌入: {self.category_embeddings.shape}")
        except Exception as e:
            logging.warning(f"加载类别嵌入失败: {e}")

    # ---- Attributes 文本嵌入 ----

    def _load_attributes_text(self) -> List[str]:
        return [self._attributes_to_text(self.businesses[bid]) for bid in self.business_ids]

    def _cache_safe_name(self, name: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in name)

    def _build_attributes_cache_paths(self, model_name: str) -> Tuple[Path, Path]:
        text_blob = '\n'.join(self.attributes_text or [])
        text_hash = hashlib.sha1(text_blob.encode('utf-8')).hexdigest()
        model_key = self._cache_safe_name(model_name)
        cache_dir = self.data_dir / '.cache' / 'semantic_id'
        cache_dir.mkdir(parents=True, exist_ok=True)
        target_dim = int(self.attribute_dim or 64)
        proj_key = f"d{target_dim}" if self.use_attribute_projection else "raw"
        base = f"attr_emb_{model_key}_{proj_key}_{text_hash[:16]}"
        return cache_dir / f"{base}.npy", cache_dir / f"{base}.json"

    def _load_attributes_embeddings(self, model_name: str, use_cache: bool = True):
        emb_path, meta_path = self._build_attributes_cache_paths(model_name)
        target_dim = int(self.attribute_dim or 64)

        if use_cache and emb_path.exists() and meta_path.exists():
            try:
                cached = np.load(emb_path).astype(np.float32)
                if cached.shape[0] == len(self.business_ids):
                    self.attr_embeddings = cached
                    self.attr_embedding_source = 'cache'
                    logging.info(f"加载 attributes 缓存嵌入: {cached.shape} ({emb_path})")
                    return
                logging.warning("attributes 缓存条数不匹配，重算嵌入")
            except Exception as e:
                logging.warning(f"读取 attributes 缓存失败，重算嵌入: {e}")

        embeddings = None
        if self.use_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"加载 Attributes SentenceTransformer: {model_name}")
                model = SentenceTransformer(model_name)
                embeddings = model.encode(self.attributes_text, show_progress_bar=True)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if self.use_attribute_projection and embeddings.shape[1] != target_dim:
                    embeddings = self._project_embeddings(embeddings, target_dim)
                self.attr_embedding_source = 'st_embedding'
            except Exception as e:
                logging.warning(f"Attributes SentenceTransformer 不可用，回退哈希嵌入: {e}")

        if embeddings is None:
            embeddings = self._build_fallback_attribute_embeddings(self.attributes_text, target_dim)
            self.attr_embedding_source = 'fallback'

        self.attr_embeddings = embeddings.astype(np.float32)

        if use_cache:
            try:
                np.save(emb_path, self.attr_embeddings)
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'num_poi': len(self.business_ids),
                            'model_name': model_name,
                            'embedding_dim': int(self.attr_embeddings.shape[1]),
                            'target_dim': target_dim,
                            'use_projection': self.use_attribute_projection,
                            'source': self.attr_embedding_source
                        },
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
            except Exception as e:
                logging.warning(f"保存 attributes 缓存失败: {e}")

    def _project_embeddings(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        input_dim = int(embeddings.shape[1])
        if input_dim == target_dim:
            return embeddings.astype(np.float32)

        # 优先使用 PCA，保持与 category embedding 路径一致
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim)
            projected = pca.fit_transform(embeddings)
            logging.info(f"Attributes PCA 投影: {input_dim} -> {target_dim}")
            return projected.astype(np.float32)
        except Exception as e:
            logging.warning(f"Attributes PCA 不可用，使用随机投影: {e}")

        # 回退：固定随机种子，保证可复现
        rng = np.random.RandomState(42)
        proj_matrix = rng.normal(0.0, 1.0 / np.sqrt(target_dim), size=(input_dim, target_dim)).astype(np.float32)
        projected = embeddings @ proj_matrix
        return projected.astype(np.float32)

    def _build_fallback_attribute_embeddings(self, texts: List[str], dim: int) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().replace(',', ' ').split():
                idx = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % dim
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors

    def _attributes_to_text(self, biz: Dict) -> str:
        attrs = biz.get('attributes') or {}
        categories = (biz.get('categories') or '').lower()

        if not attrs:
            if any(k in categories for k in ['restaurant', 'food', 'cafe', 'bar']):
                return "missing attributes for food business"
            if categories:
                return "non-food category with sparse attributes"
            return "unknown attributes format"

        parts = []
        price = attrs.get('RestaurantsPriceRange2', '')
        if price:
            parts.append(f"Price range {price}")

        wifi = self._clean_str(attrs.get('WiFi', ''))
        if wifi and wifi not in ('none', 'no'):
            parts.append(f"WiFi {wifi}")

        alcohol = self._clean_str(attrs.get('Alcohol', ''))
        if alcohol and alcohol not in ('none', 'no'):
            parts.append(f"Alcohol {alcohol}")

        noise = self._clean_str(attrs.get('NoiseLevel', ''))
        if noise:
            parts.append(f"Noise level {noise}")

        for attr, key in [
            ('OutdoorSeating', 'Outdoor seating'),
            ('RestaurantsTakeOut', 'Takeout available'),
            ('RestaurantsDelivery', 'Delivery available'),
            ('HasTV', 'Has TV'),
            ('GoodForKids', 'Good for kids'),
            ('RestaurantsReservations', 'Reservations accepted'),
            ('Caters', 'Caters'),
            ('BusinessAcceptsCreditCards', 'Credit cards accepted'),
            ('HappyHour', 'Happy hour'),
        ]:
            if self._is_true(attrs.get(attr, '')):
                parts.append(key)

        for nested_key in ['Ambience', 'GoodForMeal', 'BusinessParking', 'Music']:
            true_keys = self._extract_true_keys(attrs.get(nested_key, ''))
            if true_keys:
                parts.append(f"{nested_key}: {', '.join(true_keys)}")

        if parts:
            return ', '.join(parts)

        if any(k in categories for k in ['restaurant', 'food', 'cafe', 'bar']):
            return "missing attributes for food business"
        if categories:
            return "non-food category with sparse attributes"
        return "unknown attributes format"

    def _clean_str(self, val) -> str:
        if val is None:
            return ''
        s = str(val).strip()
        s = s.replace("u'", "'").replace('u"', '"')
        s = s.strip("'\"")
        return s.lower()

    def _is_true(self, val) -> bool:
        return self._clean_str(val) in ('true', '1', 'yes')

    def _extract_true_keys(self, nested_val: Any) -> List[str]:
        if nested_val is None:
            return []
        s = str(nested_val).strip()
        if not s or s.lower() == 'none':
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return [k for k, v in parsed.items() if self._is_true(v)]
        except Exception:
            return []
        return []

    # ---- Dataset 接口 ----

    def __len__(self):
        return len(self.business_ids)

    def __getitem__(self, idx: int) -> Dict:
        bid = self.business_ids[idx]
        biz = self.businesses[bid]

        cat_emb = self._get_category_embedding(biz)       # [64]
        spatial = self._get_spatial_3d(biz)               # [3]
        temporal = self._get_fourier_time(bid)            # [12]
        parts = [cat_emb, spatial, temporal]
        if self.use_attributes and self.attr_embeddings is not None:
            attr_emb = self.attr_embeddings[idx]
            parts.append(attr_emb)

        feature_vector = np.concatenate(parts)
        if feature_vector.shape[0] != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: got={feature_vector.shape[0]}, expected={self.feature_dim}"
            )

        return {
            'business_id': bid,
            'feature_vector': torch.tensor(feature_vector, dtype=torch.float32)
        }


def gnpr_collate_fn(batch: List[Dict]) -> Dict:
    """GNPR-SID V2 批处理函数"""
    return {
        'business_ids': [item['business_id'] for item in batch],
        'feature_vector': torch.stack([item['feature_vector'] for item in batch])
    }


def create_gnpr_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    geo_stratified: bool = False,
    min_city_poi: int = 200,
    category_embeddings_path: str = None,
    use_sentence_transformer: bool = False,
    sentence_model_name: str = 'all-MiniLM-L6-v2',
    use_attributes: bool = False,
    attribute_dim: int = 64,
    use_attribute_cache: bool = True,
    use_attribute_projection: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, 'GNPRSIDPOIDataset']:
    """创建 GNPR-SID V2 风格数据加载器"""
    dataset = GNPRSIDPOIDataset(
        data_dir=data_dir,
        category_embeddings_path=category_embeddings_path,
        use_sentence_transformer=use_sentence_transformer,
        sentence_model_name=sentence_model_name,
        use_attributes=use_attributes,
        attribute_dim=attribute_dim,
        use_attribute_cache=use_attribute_cache,
        use_attribute_projection=use_attribute_projection
    )

    n = len(dataset)

    if geo_stratified and hasattr(dataset, 'geo_map') and dataset.geo_map:
        train_idx, val_idx, test_idx = geo_stratified_split(
            dataset, dataset.geo_map,
            train_ratio=train_ratio, val_ratio=val_ratio
        )
        logging.info(f"地理分层切分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        import random
        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        logging.info(f"随机切分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set   = torch.utils.data.Subset(dataset, val_idx)
    test_set  = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=gnpr_collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=gnpr_collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=gnpr_collate_fn)

    return train_loader, val_loader, test_loader, dataset
