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
    'dataset_dir': str(_BASE_DIR / 'dataset/yelp/processed'),
    'semantic_ids_path': str(_BASE_DIR / 'output/semantic_ids.json'),
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'max_history_length': 10,  # 用户历史最多10条
    'cache_dir': str(_BASE_DIR / 'output/dataset_cache'),  # 缓存目录
}

# Prompt 模板
PROMPT_TEMPLATE = {
    'system': (
        "You are an intelligent POI (Point of Interest) recommendation assistant. "
        "Your task is to predict the next POI that a user will visit based on "
        "their profile, visit history, and current spatiotemporal context. "
        "Generate the Semantic ID in the format: level0-level1-level2[GRID] "
        "(e.g., 12-34-56[GQ])."
    ),
    
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
        "the user will visit. Consider:\n"
        "1. User's historical preferences\n"
        "2. Spatiotemporal patterns (location, time, day)\n"
        "3. POI availability (must be open at the predicted time)\n\n"
        "Output only the Semantic ID in format: XX-XX-XX[GG]"
    )
}
