"""
LLM 微调配置

基于 QLoRA 的 Qwen3-8B-Instruct 指令微调配置
"""
from pathlib import Path

# config.py 所在目录的上一级即项目根目录（main/）
_BASE_DIR = Path(__file__).parent.parent

# 模型配置
MODEL_CONFIG = {
    'base_model': str(_BASE_DIR / 'models/JunHowie/Qwen3-8B-Instruct'),  # 使用本地模型
    'use_qlora': True,  # 使用 QLoRA
    'load_in_4bit': True,  # 4-bit 量化
    'bnb_4bit_quant_type': 'nf4',  # NormalFloat4
    'bnb_4bit_compute_dtype': 'bfloat16',
    'bnb_4bit_use_double_quant': True,  # 双重量化
}

# LoRA 配置（参考 GNPR-SID）
LORA_CONFIG = {
    'r': 16,  # 低秩矩阵的秩
    'lora_alpha': 32,  # LoRA 缩放因子
    'target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # 注意力层
        'gate_proj', 'up_proj', 'down_proj'  # FFN 层
    ],
    'lora_dropout': 0.05,
    'bias': 'none',
    'task_type': 'CAUSAL_LM'
}

# 训练配置
TRAINING_CONFIG = {
    'output_dir': str(_BASE_DIR / 'checkpoints/llm_sid'),
    'num_train_epochs': 3,
    'per_device_train_batch_size': 6,  # 4090 显存剩余 3GB，可提至 6
    'per_device_eval_batch_size': 6,
    'gradient_accumulation_steps': 3,  # 有效 batch_size = 6 * 3 = 18
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'logging_steps': 50,
    'eval_steps': 200,
    'save_steps': 200,
    'save_total_limit': 3,          # 最多保留 3 个 checkpoint（+best），节省磁盘
    'max_seq_length': 704,          # 覆盖 100% 样本（最长 689），比 768 节省 16% 计算量
    'bf16': True,  # BF16 混合精度
    'optim': 'paged_adamw_32bit',  # 分页优化器
    'gradient_checkpointing': True,  # 梯度检查点
}

# 数据配置
DATA_CONFIG = {
    'dataset_dir': str(_BASE_DIR / 'dataset/yelp/processed/Philadelphia'),  # 数据集目录
    'semantic_ids_path': str(_BASE_DIR / 'output/sid/PA_main_city/semantic_ids_v2.json'),
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'max_history_length': 50,  # 用户历史最多50条
    'cache_dir': str(_BASE_DIR / 'output/dataset_cache'),  # 缓存目录
    'cache_schema': 'sid_time_history_angle_bracket_gnpr',
    'export_prompts_on_prepare': True,
    'prompt_export_dir': str(_BASE_DIR / 'output/dataset_cache'),
    'default_prompt_mode': 'sid_time_history',
    'enable_legacy_prompt_mode': True,
    'sid_format_mode': 'angle_bracket',
    'gnpr_data_paths': {
        'train': None,
        'val': None,
        'test': None,
    },
}

# Prompt 模板
PROMPT_TEMPLATE = {
    'system': (
        "You are a POI next-visit prediction assistant. "
        "Predict only the next POI Semantic ID based on historical visit SIDs and timestamps. "
        "Output only one Semantic ID in angle-bracket format, such as "
        "<a_12><b_34><c_56> or <a_12><b_34><c_56><d_1>."
    ),

    'sid_time_history': {
        'instruction': (
            "Here is a record of a user's POI accesses, your task is based on the history "
            "to predict the POI that the user is likely to access at the specified time."
        ),
        'history_prefix': 'User_{user_id} checkin history: ',
        'history_item': '{time} visited {sid}',
        'query_suffix': 'When {target_time} user_{user_id} is likely to visit:',
    },

    'legacy_profile_context': {
        'user_template': (
            "### User Profile:\n"
            "- User ID: {user_id}\n"
            "- Active Level: {review_count} reviews\n"
            "- Average Rating: {average_stars:.1f} stars\n"
            "- Favorite Categories: {favorite_categories}\n"
        ),
        'context_template': (
            "### Spatiotemporal Context:\n"
            "- Current Location: Plus Code {pluscode}\n"
            "- Time: {time_description}\n"
            "- Day Type: {day_type}\n"
        ),
        'history_template': (
            "### Visit History (Recent {count} visits):\n"
            "{history_items}"
        ),
        'instruction': (
            "Based on the above information, predict the Semantic ID of the next POI "
            "the user will visit. Output only one Semantic ID in angle-bracket format."
        ),
    },
}
