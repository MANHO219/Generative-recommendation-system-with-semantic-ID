"""
LLM 微调数据集

构建指令微调格式的训练数据
"""

import json
import os
import random
import re
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path

try:
    from .config import DATA_CONFIG, PROMPT_TEMPLATE
except ImportError:
    from config import DATA_CONFIG, PROMPT_TEMPLATE


def _maybe_apply_strict_kcore() -> None:
    if not DATA_CONFIG.get('enable_strict_kcore', False):
        return

    try:
        from semantic_id.kcore import prepare_k_core_data_dir
    except Exception as err:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        try:
            from semantic_id.kcore import prepare_k_core_data_dir
        except Exception as inner_err:
            raise RuntimeError(f'无法导入 strict k-core 模块: {inner_err}') from inner_err

    source_dir = DATA_CONFIG['dataset_dir']
    k_value = int(DATA_CONFIG.get('k_core', 5))
    use_cache = bool(DATA_CONFIG.get('k_core_use_cache', True))
    output_dir = DATA_CONFIG.get('k_core_output_dir')

    filtered_dir, stats = prepare_k_core_data_dir(
        data_dir=source_dir,
        k=k_value,
        output_dir=output_dir,
        use_cache=use_cache,
    )
    DATA_CONFIG['dataset_dir'] = filtered_dir
    logging.info(
        'Strict k-core enabled: k=%s users=%s items=%s interactions=%s dir=%s',
        stats.get('k'),
        stats.get('users'),
        stats.get('items'),
        stats.get('interactions'),
        filtered_dir,
    )


def _write_json_atomic(path: Path, data: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> None:
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with open(temp_path, 'w') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    os.replace(temp_path, path)


def _is_angle_bracket_sid(sid: str) -> bool:
    return bool(re.fullmatch(r'(<[a-d]_\d+>){3,4}', sid.strip()))


def _parse_sid_hyphen_grid(sid: str) -> tuple[List[int], str | None]:
    match = re.fullmatch(r'(\d+)-(\d+)-(\d+)(?:\[([^\]]+)\])?', sid.strip())
    if not match:
        raise ValueError(f'Unsupported SID format: {sid}')
    levels = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    return levels, match.group(4)


def _extract_d_index(disambig: str | None) -> int | None:
    if disambig is None:
        return None
    trailing_index = re.search(r'_(\d+)$', disambig)
    if trailing_index:
        return int(trailing_index.group(1))
    pure_number = re.fullmatch(r'\d+', disambig)
    if pure_number:
        return int(disambig)
    return 0


def _to_angle_bracket_sid(sid: str) -> str:
    sid = sid.strip()
    if _is_angle_bracket_sid(sid):
        return sid
    values, disambig = _parse_sid_hyphen_grid(sid)
    base_sid = ''.join(f'<{label}_{value}>' for label, value in zip(['a', 'b', 'c'], values))
    d_index = _extract_d_index(disambig)
    if d_index is None:
        return base_sid
    return f'{base_sid}<d_{d_index}>'


class LLMFinetuneDataset:
    """LLM 微调数据集"""
    
    def __init__(
        self,
        samples: List[Dict[str, Any]] = None,
        dataset_dir: str = None,
        semantic_ids_path: str = None,
        split: str = 'train',
        cache_dir: str = None
    ):
        """
        Args:
            samples: 直接传入样本列表 (如果已加载)
            dataset_dir: Yelp 数据集目录 (用于从头构建)
            semantic_ids_path: Semantic ID JSON 路径
            split: 数据集划分 (train/val/test)
            cache_dir: 缓存目录
        """
        self.split = split
        
        if samples is not None:
            # 模式 1: 直接使用传入的样本
            self.samples = samples
        elif cache_dir and self._check_cache(cache_dir, split):
            # 模式 2: 从缓存加载
            self.samples = self._load_from_cache(cache_dir, split)
        else:
            # 模式 3: 不支持直接在 __init__ 中进行全量构建，
            # 必须通过外部的 prepare_datasets 统一构建并缓存，
            # 以避免由 random shuffle 导致的数据泄露或不一致。
            raise ValueError(
                f"No cache found for split '{split}' and no samples provided. "
                "Please run prepare_datasets() to generate and cache the data first."
            )

    def _check_cache(self, cache_dir: str, split: str) -> bool:
        path = Path(cache_dir) / f"{split}_samples.json"
        return path.exists()

    def _load_from_cache(self, cache_dir: str, split: str) -> List[Dict[str, Any]]:
        path = Path(cache_dir) / f"{split}_samples.json"
        print(f"Loading {split} samples from cache: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _ensure_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValueError(f'Unsupported datetime value: {value}')

    def _get_time_description(self, dt: datetime) -> str:
        """获取时间描述"""
        hour = dt.hour
        if 6 <= hour < 12:
            return f"{dt.strftime('%A')}, Morning ({hour}:00)"
        elif 12 <= hour < 18:
            return f"{dt.strftime('%A')}, Afternoon ({hour}:00)"
        elif 18 <= hour < 22:
            return f"{dt.strftime('%A')}, Evening ({hour}:00)"
        else:
            return f"{dt.strftime('%A')}, Night ({hour}:00)"
            
    def _get_day_type(self, dt: datetime) -> str:
        """获取日期类型"""
        return 'Weekend' if dt.weekday() >= 5 else 'Weekday'
        
    def _get_pluscode(self, lat: float, lon: float) -> str:
        """获取 Plus Code（简化版，使用坐标区间）"""
        # 这里使用简化版本，实际应使用 openlocationcode 库
        lat_code = f"{int(lat * 100) % 100:02d}"
        lon_code = f"{int(abs(lon) * 100) % 100:02d}"
        return f"{lat_code}{lon_code}+XX"
        
    def _format_history_items(self, history: List[Dict]) -> str:
        """格式化历史访问记录"""
        items = []
        for idx, visit in enumerate(history, 1):
            item = (
                f"{idx}. {visit['business_name']} "
                f"({visit['categories'][:50]}) - "
                f"Rated {visit['stars']:.1f} stars - "
                f"{visit['date'].strftime('%Y-%m-%d')}"
            )
            items.append(item)
        return '\n'.join(items)
        
    def _get_favorite_categories(self, history: List[Dict], top_k: int = 3) -> str:
        """获取用户最喜欢的类别"""
        category_counts = {}
        for visit in history:
            cats = visit['categories'].split(', ') if visit['categories'] else []
            for cat in cats:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                
        top_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ', '.join([cat for cat, _ in top_cats]) if top_cats else 'N/A'
        
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """主路径：SID+时间历史格式"""
        if {'instruction', 'input', 'output'}.issubset(sample.keys()):
            return sample['input']

        history = sample['history']
        target = sample['target']
        target_date = self._ensure_datetime(target['date'])
        sid_template = PROMPT_TEMPLATE['sid_time_history']

        history_items = []
        for visit in history:
            visit_date = self._ensure_datetime(visit['date'])
            history_items.append(
                sid_template['history_item'].format(
                    time=visit_date.strftime('%Y-%m-%d %H:%M:%S'),
                    sid=visit['sid'],
                )
            )

        history_text = ', '.join(history_items)
        user_id = sample['user_id']
        return (
            f"{sid_template['history_prefix'].format(user_id=user_id)}{history_text}.\n"
            f"{sid_template['query_suffix'].format(target_time=target_date.strftime('%Y-%m-%d %H:%M:%S'), user_id=user_id)}"
        )
        
    def format_instruction(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """格式化为 Llama 3.1 Chat 格式"""
        if {'instruction', 'input', 'output'}.issubset(sample.keys()):
            target_sid = _to_angle_bracket_sid(sample['output'])
            return {
                'messages': [
                    {'role': 'system', 'content': PROMPT_TEMPLATE['system']},
                    {'role': 'user', 'content': sample['input']},
                    {'role': 'assistant', 'content': target_sid}
                ]
            }

        prompt = self.format_prompt(sample)
        target_sid = _to_angle_bracket_sid(sample['target_sid'])
        
        return {
            'messages': [
                {'role': 'system', 'content': PROMPT_TEMPLATE['system']},
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': target_sid}
            ]
        }
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self.format_instruction(sample)


class DatasetBuilder:
    """Helper class to build and cache datasets"""
    def __init__(self, dataset_dir, semantic_ids_path, cache_dir):
        self.dataset_dir = Path(dataset_dir)
        self.semantic_ids_path = semantic_ids_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def build_and_save(self):
        gnpr_paths = DATA_CONFIG.get('gnpr_data_paths', {})
        if gnpr_paths and all(gnpr_paths.get(split) for split in ['train', 'val', 'test']):
            return self._build_from_gnpr_json(gnpr_paths)

        preprocess_pipeline = DATA_CONFIG.get('preprocess_pipeline', 'legacy')
        if preprocess_pipeline == 'yelp_session':
            return self._build_with_session_pipeline()
        if preprocess_pipeline != 'legacy':
            raise ValueError(f"Unsupported preprocess_pipeline: {preprocess_pipeline}")


        print("Starting dataset construction...")
        
        # Load raw data
        print("Loading raw data...")
        businesses_df = pd.read_json(self.dataset_dir / 'business_poi.json', lines=True)
        users_df = pd.read_json(self.dataset_dir / 'user_active.json', lines=True)
        print("Loading reviews (this may take a while)...")
        reviews_df = pd.read_json(self.dataset_dir / 'review_poi.json', lines=True)
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])
        
        with open(self.semantic_ids_path, 'r') as f:
            semantic_ids = json.load(f)

        # Optimize lookups
        print("Indexing data...")
        businesses_dict = businesses_df.set_index('business_id').to_dict('index')
        users_dict = users_df.set_index('user_id').to_dict('index')
        
        # Build samples
        print("Processing user histories and generating samples...")
        sliding_samples = []
        last_item_samples = []
        max_history = DATA_CONFIG['max_history_length']
        min_user_interactions = int(DATA_CONFIG.get('min_user_interactions', 2))
        min_user_interactions = max(2, min_user_interactions)
        test_mode = DATA_CONFIG.get('test_mode', 'sliding')
        export_last_item_test = bool(DATA_CONFIG.get('export_last_item_test', True))

        if test_mode not in {'sliding', 'last_item'}:
            raise ValueError(f"Unsupported test_mode: {test_mode}")
        
        grouped = reviews_df.groupby('user_id')
        total_groups = len(grouped)
        count = 0
        kept_user_count = 0
        
        for user_id, group in grouped:
            count += 1
            if count % 5000 == 0:
                print(f"  Processed {count}/{total_groups} users")

            # Sort by date
            group = group.sort_values('date')
            
            # Build visits sequence
            visits = []
            for _, row in group.iterrows():
                business_id = row['business_id']
                if business_id not in businesses_dict:
                    continue
                if business_id not in semantic_ids:
                    continue
                
                business_info = businesses_dict[business_id]
                visits.append({
                    'business_id': business_id,
                    'business_name': business_info['name'],
                    'categories': business_info.get('categories', ''),
                    'stars': row['stars'],
                    'date': row['date'].isoformat(),  # Serialize for JSON
                    'latitude': business_info['latitude'],
                    'longitude': business_info['longitude'],
                    'sid': _to_angle_bracket_sid(semantic_ids[business_id]),
                })

            if len(visits) < min_user_interactions:
                continue
            kept_user_count += 1

            # Get user info
            if user_id not in users_dict:
                continue
            user_info = users_dict[user_id]
            user_info_dict = {
                'review_count': int(user_info['review_count']),
                'average_stars': float(user_info['average_stars']),
            }
            
            # Generate sliding window samples
            user_samples = []
            for i in range(1, len(visits)):
                history = visits[max(0, i-max_history):i]
                target = visits[i]
                
                target_business_id = target['business_id']
                if target_business_id not in semantic_ids:
                    continue
                    
                sample = {
                    'user_id': user_id,
                    'user_info': user_info_dict,
                    'history': history,
                    'target': target,
                    'target_sid': _to_angle_bracket_sid(semantic_ids[target_business_id])
                }
                user_samples.append(sample)

            if not user_samples:
                continue

            last_item_samples.append(user_samples[-1])
            if test_mode == 'last_item':
                sliding_samples.extend(user_samples[:-1])
            else:
                sliding_samples.extend(user_samples)

        print(f"Generated {len(sliding_samples)} sliding samples.")
        print(f"Generated {len(last_item_samples)} per-user last-item samples.")
        print(
            f"Users kept after min interactions filter: {kept_user_count}/{total_groups} "
            f"(min_user_interactions={min_user_interactions})"
        )

        # Deterministic shuffle and split
        random.seed(42)  # Ensure reproducibility
        random.shuffle(sliding_samples)

        if test_mode == 'last_item':
            train_val_ratio = float(DATA_CONFIG['train_split']) + float(DATA_CONFIG['val_split'])
            if train_val_ratio <= 0:
                raise ValueError('train_split + val_split must be > 0 when test_mode=last_item')

            normalized_train_ratio = float(DATA_CONFIG['train_split']) / train_val_ratio
            train_size = int(len(sliding_samples) * normalized_train_ratio)

            splits = {
                'train': sliding_samples[:train_size],
                'val': sliding_samples[train_size:],
                'test': last_item_samples,
            }
            print(
                f"Split mode: last_item | train={len(splits['train'])} val={len(splits['val'])} "
                f"test(last_item)={len(splits['test'])}"
            )
        else:
            train_size = int(len(sliding_samples) * DATA_CONFIG['train_split'])
            val_size = int(len(sliding_samples) * DATA_CONFIG['val_split'])

            splits = {
                'train': sliding_samples[:train_size],
                'val': sliding_samples[train_size:train_size+val_size],
                'test': sliding_samples[train_size+val_size:]
            }
            if export_last_item_test:
                splits['test_last_item'] = last_item_samples
            print(
                f"Split mode: sliding | train={len(splits['train'])} val={len(splits['val'])} "
                f"test(sliding)={len(splits['test'])}"
            )
        
        # Save cache
        for split_name, split_samples in splits.items():
            path = self.cache_dir / f"{split_name}_samples.json"
            print(f"Saving {split_name} cache to {path} ({len(split_samples)} samples)...")
            _write_json_atomic(path, split_samples)
        
        print("Dataset preparation complete.")
        return splits

    def _build_with_session_pipeline(self) -> Dict[str, List[Dict[str, Any]]]:
        print("Starting dataset construction with Yelp session pipeline...")

        print("Loading raw data...")
        businesses_df = pd.read_json(self.dataset_dir / 'business_poi.json', lines=True)
        users_df = pd.read_json(self.dataset_dir / 'user_active.json', lines=True)
        print("Loading reviews (this may take a while)...")
        reviews_df = pd.read_json(self.dataset_dir / 'review_poi.json', lines=True)
        reviews_df['date'] = pd.to_datetime(reviews_df['date'])

        print("Applying session pipeline on Yelp reviews...")
        reviews_df = self._apply_session_pipeline_to_reviews(reviews_df)

        with open(self.semantic_ids_path, 'r', encoding='utf-8') as f:
            semantic_ids = json.load(f)

        print("Indexing data...")
        businesses_dict = businesses_df.set_index('business_id').to_dict('index')
        users_dict = users_df.set_index('user_id').to_dict('index')

        max_history = int(DATA_CONFIG['max_history_length'])
        min_user_interactions = int(DATA_CONFIG.get('min_user_interactions', 2))
        min_user_interactions = max(2, min_user_interactions)

        grouped = reviews_df.sort_values(['user_id', 'pseudo_session_trajectory_id', 'date']).groupby(['user_id', 'pseudo_session_trajectory_id'])
        total_groups = len(grouped)
        kept_user_count = 0

        splits: Dict[str, List[Dict[str, Any]]] = {
            'train': [],
            'val': [],
            'test': [],
        }

        for (user_id, trajectory_id), group in grouped:
            group = group.sort_values('date')

            visits = []
            for _, row in group.iterrows():
                split_tag = str(row.get('SplitTag', 'ignore'))
                if split_tag not in {'train', 'validation', 'test'}:
                    continue

                business_id = row['business_id']
                if business_id not in businesses_dict:
                    continue
                if business_id not in semantic_ids:
                    continue

                business_info = businesses_dict[business_id]
                visits.append({
                    'split_tag': split_tag,
                    'business_id': business_id,
                    'business_name': business_info['name'],
                    'categories': business_info.get('categories', ''),
                    'stars': row['stars'],
                    'date': row['date'].isoformat(),
                    'latitude': business_info['latitude'],
                    'longitude': business_info['longitude'],
                    'sid': _to_angle_bracket_sid(semantic_ids[business_id]),
                })

            # 每个 trajectory 内至少需要 2 个点才能形成 (history -> target)
            if len(visits) < 2:
                continue
            kept_user_count += 1

            if user_id not in users_dict:
                continue
            user_info = users_dict[user_id]
            user_info_dict = {
                'review_count': int(user_info['review_count']),
                'average_stars': float(user_info['average_stars']),
            }

            for i in range(1, len(visits)):
                history = visits[max(0, i - max_history):i]
                target = visits[i]
                target_split = target['split_tag']

                if not history:
                    continue

                sample = {
                    'user_id': user_id,
                    'user_info': user_info_dict,
                    'history': [{k: v for k, v in h.items() if k != 'split_tag'} for h in history],
                    'target': {k: v for k, v in target.items() if k != 'split_tag'},
                    'target_sid': target['sid'],
                }

                if target_split == 'train':
                    splits['train'].append(sample)
                elif target_split == 'validation':
                    splits['val'].append(sample)
                elif target_split == 'test':
                    splits['test'].append(sample)

        print(
            f"Users kept after min interactions filter: {kept_user_count}/{total_groups} "
            f"(min_user_interactions={min_user_interactions})"
        )
        print(
            f"Yelp-session split samples | train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )

        if DATA_CONFIG.get('test_mode', 'sliding') != 'sliding':
            print("Warning: test_mode is ignored under preprocess_pipeline='yelp_session'.")

        for split_name, split_samples in splits.items():
            path = self.cache_dir / f"{split_name}_samples.json"
            print(f"Saving {split_name} cache to {path} ({len(split_samples)} samples)...")
            _write_json_atomic(path, split_samples)

        print("Yelp session pipeline dataset preparation complete.")
        return splits

    def _apply_session_pipeline_to_reviews(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        temp = reviews_df.rename(columns={'user_id': 'UId', 'business_id': 'PId', 'date': 'Time'}).copy()
        temp = temp.sort_values(['UId', 'Time']).reset_index(drop=True)

        if bool(DATA_CONFIG.get('session_enable_filter_low_frequency', False)):
            temp = self._filter_low_frequency(
                df=temp,
                min_poi_freq=int(DATA_CONFIG.get('session_min_poi_freq', 10)),
                min_user_freq=int(DATA_CONFIG.get('session_min_user_freq', 10)),
            )

        temp = self._split_by_global_time(
            temp,
            train_ratio=float(DATA_CONFIG.get('train_split', 0.8)),
            val_ratio=float(DATA_CONFIG.get('val_split', 0.1)),
        )

        if bool(DATA_CONFIG.get('session_remove_isolated_24h', True)):
            temp = self._remove_isolated_checkins_24h(temp)

        temp = self._build_pseudo_sessions(
            temp,
            session_time_interval_min=int(DATA_CONFIG.get('session_time_interval_min', 60 * 24)),
        )

        if bool(DATA_CONFIG.get('session_ignore_singleton_sessions', True)):
            temp = self._ignore_singleton_sessions(temp)

        if bool(DATA_CONFIG.get('session_remove_unseen_user_poi', True)):
            temp = self._remove_unseen_user_poi(temp)

        temp = temp.rename(columns={'UId': 'user_id', 'PId': 'business_id', 'Time': 'date'})
        temp['date'] = pd.to_datetime(temp['date'])
        return temp.sort_values(['user_id', 'date']).reset_index(drop=True)
    def _lookup_semantic_sid(self, semantic_ids: Dict[str, str], poi_id: Any) -> Optional[str]:
        if poi_id in semantic_ids:
            return semantic_ids[poi_id]
        key_str = str(poi_id)
        if key_str in semantic_ids:
            return semantic_ids[key_str]
        try:
            key_int = int(poi_id)
            if key_int in semantic_ids:
                return semantic_ids[key_int]
            key_int_str = str(key_int)
            if key_int_str in semantic_ids:
                return semantic_ids[key_int_str]
        except (ValueError, TypeError):
            return None
        return None

    def _filter_low_frequency(self, df: pd.DataFrame, min_poi_freq: int, min_user_freq: int) -> pd.DataFrame:
        poi_count = df.groupby('PId')['UId'].count()
        keep_pois = set(poi_count[poi_count > min_poi_freq].index)
        df = df[df['PId'].isin(keep_pois)]

        user_count = df.groupby('UId')['PId'].count()
        keep_users = set(user_count[user_count > min_user_freq].index)
        df = df[df['UId'].isin(keep_users)]
        return df.sort_values(by=['UId', 'Time'], ascending=True).reset_index(drop=True)

    def _split_by_global_time(self, df: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
        ratio_sum = train_ratio + val_ratio
        if train_ratio <= 0 or val_ratio < 0 or ratio_sum >= 1:
            raise ValueError(
                f"Invalid split ratios for session pipeline: train_split={train_ratio}, val_split={val_ratio}. "
                "Require train_split > 0, val_split >= 0, and train_split + val_split < 1."
            )

        df = df.sort_values('Time').reset_index(drop=True)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * ratio_sum)
        df['SplitTag'] = 'train'
        df.loc[train_end:val_end - 1, 'SplitTag'] = 'validation'
        df.loc[val_end:, 'SplitTag'] = 'test'
        return df.sort_values(['UId', 'Time']).reset_index(drop=True)

    def _remove_isolated_checkins_24h(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=['UId', 'Time'], ascending=True).reset_index(drop=True)
        prev_t = df.groupby('UId')['Time'].shift(1)
        next_t = df.groupby('UId')['Time'].shift(-1)
        gap_prev = (df['Time'] - prev_t).dt.total_seconds().fillna(float('inf'))
        gap_next = (next_t - df['Time']).dt.total_seconds().fillna(float('inf'))
        isolated = (gap_prev > 86400) & (gap_next > 86400)
        return df[~isolated].reset_index(drop=True)

    def _build_pseudo_sessions(self, df: pd.DataFrame, session_time_interval_min: int) -> pd.DataFrame:
        df = df.sort_values(by=['UId', 'Time'], ascending=True).reset_index(drop=True)
        diffs = df.groupby('UId')['Time'].diff()
        diffs_min = diffs.dt.total_seconds() / 60.0
        new_session = diffs.isna() | (diffs_min > session_time_interval_min)
        df['pseudo_session_trajectory_id'] = new_session.cumsum().astype(int) - 1
        return df

    def _ignore_singleton_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df.groupby('pseudo_session_trajectory_id')['Time'].transform('count')
        df = df.copy()
        df.loc[counts == 1, 'SplitTag'] = 'ignore'
        return df

    def _remove_unseen_user_poi(self, df: pd.DataFrame) -> pd.DataFrame:
        train = df[df['SplitTag'] == 'train']
        val = df[df['SplitTag'] == 'validation']
        test = df[df['SplitTag'] == 'test']

        train_users = set(train['UId'])
        train_pois = set(train['PId'])

        val = val[val['UId'].isin(train_users) & val['PId'].isin(train_pois)].reset_index(drop=True)
        test = test[test['UId'].isin(train_users) & test['PId'].isin(train_pois)].reset_index(drop=True)

        ignored = df[df['SplitTag'] == 'ignore'].reset_index(drop=True)
        merged = pd.concat([train.reset_index(drop=True), val, test, ignored], axis=0)
        return merged.sort_values(['UId', 'Time']).reset_index(drop=True)

    def _build_from_gnpr_json(self, gnpr_paths: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        print('Loading GNPR instruction datasets...')
        splits = {}
        for split_name in ['train', 'val', 'test']:
            with open(gnpr_paths[split_name], 'r', encoding='utf-8') as f:
                loaded_samples = json.load(f)

            normalized = []
            for sample in loaded_samples:
                if not {'instruction', 'input', 'output'}.issubset(sample.keys()):
                    continue
                sample['output'] = _to_angle_bracket_sid(sample['output'])
                normalized.append(sample)

            path = self.cache_dir / f"{split_name}_samples.json"
            print(f"Saving {split_name} cache to {path} ({len(normalized)} samples)...")
            _write_json_atomic(path, normalized)
            splits[split_name] = normalized

        print('GNPR dataset preparation complete.')
        return splits


def prepare_datasets():
    """准备训练/验证/测试数据集"""
    _maybe_apply_strict_kcore()

    cache_dir = DATA_CONFIG.get('cache_dir')
    required_splits = ['train', 'val', 'test']
    
    # Check if cache exists
    schema = DATA_CONFIG.get('cache_schema', 'default')
    schema = f"{schema}|pipeline={DATA_CONFIG.get('preprocess_pipeline', 'legacy')}|test_mode={DATA_CONFIG.get('test_mode', 'sliding')}"
    if DATA_CONFIG.get('preprocess_pipeline', 'legacy') == 'yelp_session':
        schema = (
            f"{schema}|session_filter={DATA_CONFIG.get('session_enable_filter_low_frequency', False)}"
            f"|session_remove_isolated_24h={DATA_CONFIG.get('session_remove_isolated_24h', True)}"
            f"|session_ignore_singleton_sessions={DATA_CONFIG.get('session_ignore_singleton_sessions', True)}"
            f"|session_remove_unseen_user_poi={DATA_CONFIG.get('session_remove_unseen_user_poi', True)}"
            f"|session_gap_min={DATA_CONFIG.get('session_time_interval_min', 60 * 24)}"
        )
    schema_path = Path(cache_dir) / 'schema.txt' if cache_dir else None

    cache_exists = cache_dir and all(
        (Path(cache_dir) / f"{split}_samples.json").exists() 
        for split in required_splits
    )
    schema_matched = schema_path and schema_path.exists() and schema_path.read_text(encoding='utf-8').strip() == schema

    if cache_exists and not schema_matched:
        print('Cache schema changed. Clearing old caches (strategy A)...')
        for split in required_splits:
            path = Path(cache_dir) / f"{split}_samples.json"
            if path.exists():
                path.unlink()
        optional_paths = [
            Path(cache_dir) / 'test_last_item_samples.json',
            Path(cache_dir) / 'test_last_item_prompts.json',
        ]
        for path in optional_paths:
            if path.exists():
                path.unlink()
        cache_exists = False
    
    if cache_exists:
        print(f"Found cached datasets in {cache_dir}. Loading...")
        try:
            train_dataset = LLMFinetuneDataset(split='train', cache_dir=cache_dir)
            val_dataset = LLMFinetuneDataset(split='val', cache_dir=cache_dir)
            test_dataset = LLMFinetuneDataset(split='test', cache_dir=cache_dir)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as err:
            print(f"Cache load failed ({err}). Rebuilding dataset cache...")
            for split in required_splits:
                path = Path(cache_dir) / f"{split}_samples.json"
                if path.exists():
                    path.unlink()
            extra_path = Path(cache_dir) / 'test_last_item_samples.json'
            if extra_path.exists():
                extra_path.unlink()

            builder = DatasetBuilder(
                DATA_CONFIG['dataset_dir'],
                DATA_CONFIG['semantic_ids_path'],
                cache_dir
            )
            splits = builder.build_and_save()

            train_dataset = LLMFinetuneDataset(samples=splits['train'], split='train')
            val_dataset = LLMFinetuneDataset(samples=splits['val'], split='val')
            test_dataset = LLMFinetuneDataset(samples=splits['test'], split='test')

            if schema_path:
                schema_path.write_text(schema, encoding='utf-8')
    else:
        print("Cache not found or incomplete. Building dataset from scratch...")
        builder = DatasetBuilder(
            DATA_CONFIG['dataset_dir'],
            DATA_CONFIG['semantic_ids_path'],
            cache_dir
        )
        splits = builder.build_and_save()
        
        train_dataset = LLMFinetuneDataset(samples=splits['train'], split='train')
        val_dataset = LLMFinetuneDataset(samples=splits['val'], split='val')
        test_dataset = LLMFinetuneDataset(samples=splits['test'], split='test')

        if schema_path:
            schema_path.write_text(schema, encoding='utf-8')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if DATA_CONFIG.get('export_prompts_on_prepare', False):
        export_dir = Path(DATA_CONFIG.get('prompt_export_dir', cache_dir or 'output/dataset_cache'))
        export_dir.mkdir(parents=True, exist_ok=True)

        sid_template = PROMPT_TEMPLATE['sid_time_history']
        instruction_text = sid_template.get('instruction', '')

        def _to_prompt_record(dataset: LLMFinetuneDataset, sample: Dict[str, Any]) -> Dict[str, str]:
            if {'instruction', 'input', 'output'}.issubset(sample.keys()):
                return {
                    'instruction': sample['instruction'],
                    'input': sample['input'],
                    'output': _to_angle_bracket_sid(sample['output']),
                }

            return {
                'instruction': instruction_text,
                'input': dataset.format_prompt(sample),
                'output': _to_angle_bracket_sid(sample['target_sid']),
            }

        export_entries = [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]
        test_last_item_path = Path(cache_dir) / 'test_last_item_samples.json' if cache_dir else None
        if test_last_item_path and test_last_item_path.exists():
            with open(test_last_item_path, 'r', encoding='utf-8') as file:
                test_last_item_samples = json.load(file)
            export_entries.append(
                ('test_last_item', LLMFinetuneDataset(samples=test_last_item_samples, split='test'))
            )

        for split_name, dataset in export_entries:
            prompts = [_to_prompt_record(dataset, sample) for sample in dataset.samples]
            prompt_path = export_dir / f'{split_name}_prompts.json'
            _write_json_atomic(prompt_path, prompts, ensure_ascii=False, indent=2)
            print(f'已保存 {split_name}: {len(prompts)} 条到 {prompt_path}')
    
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # 测试数据集构建
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    
    # 查看样本
    print("\n=== Sample Instruction ===")
    sample = train_dataset[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))
