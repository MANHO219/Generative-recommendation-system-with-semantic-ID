"""
LLM 微调训练器

基于 QLoRA 的 Qwen-3-8B-Insturct 指令微调
"""

import torch
import json
import time
import math
import sys
import re
import argparse
import warnings
import inspect
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessorList,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from transformers import TrainerCallback

sys.path.append(str(Path(__file__).resolve().parents[1]))

from inference.constrained_decoding import TrieConstrainedLogitsProcessor
from inference.trie import TokenTrie

try:
    from .config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG
    from .dataset import prepare_datasets
except ImportError:
    from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG
    from dataset import prepare_datasets


def _is_angle_bracket_sid(sid: str) -> bool:
    return bool(re.fullmatch(r'(<[a-d]_\d+>){3,4}', sid.strip()))


def _to_angle_bracket_sid(sid: str) -> str:
    sid = sid.strip()
    if _is_angle_bracket_sid(sid):
        return sid
    match = re.fullmatch(r'(\d+)-(\d+)-(\d+)(?:(?:<d_(\d+)>)|\[([^\]]+)\])?', sid)
    if not match:
        return sid
    values = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    base_sid = ''.join(f'<{label}_{value}>' for label, value in zip(['a', 'b', 'c'], values))
    d_suffix = match.group(4)
    if d_suffix is not None:
        return f'{base_sid}<d_{int(d_suffix)}>'
    disambig = match.group(5)
    if disambig is None:
        return base_sid
    trailing_index = re.search(r'_(\d+)$', disambig)
    if trailing_index:
        return f'{base_sid}<d_{int(trailing_index.group(1))}>'
    pure_number = re.fullmatch(r'\d+', disambig)
    if pure_number:
        return f'{base_sid}<d_{int(disambig)}>'
    return f'{base_sid}<d_0>'


def _normalize_sid_text(value: str) -> str:
    text = value.strip()
    if not text:
        return text

    token_matches = re.findall(r'<[a-d]_\d+>', text)
    if len(token_matches) >= 3:
        return ''.join(token_matches[:4]) if len(token_matches) >= 4 else ''.join(token_matches[:3])

    angle_match = re.search(r'(<[a-d]_\d+>){3,4}', text)
    if angle_match:
        return angle_match.group(0)

    dash_match = re.search(r'(\d+-\d+-\d+(?:(?:<d_\d+>)|\[[^\]]+\])?)', text)
    if dash_match:
        return _to_angle_bracket_sid(dash_match.group(1))

    return _to_angle_bracket_sid(text)


def _apply_chat_template_no_think(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {
        'tokenize': False,
        'add_generation_prompt': add_generation_prompt,
    }
    signature = inspect.signature(tokenizer.apply_chat_template)
    if 'enable_thinking' in signature.parameters:
        kwargs['enable_thinking'] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    rendered = re.sub(r'<think>[\s\S]*?</think>\s*', '', rendered)
    rendered = rendered.replace('<think>', '').replace('</think>', '')
    return rendered


class GenerativeEvalCallback(TrainerCallback):
    """在每次 eval 结束后追加生成式评测指标（exact_match / hit@k / mrr@k）。"""

    def __init__(self, tokenizer, test_dataset, sid_set, trie, num_samples=30,
                 num_beams=5, top_k=5, max_new_tokens=20):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.sid_set = sid_set
        self.trie = trie
        self.num_samples = min(num_samples, len(test_dataset))
        self.num_beams = num_beams
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, _control, model=None, _logs=None, **_kwargs):
        if model is None:
            return

        model.eval()
        exact_match = hit = mrr_sum = ndcg_sum = valid_sid = 0
        latencies = []

        for idx in range(self.num_samples):
            sample = self.test_dataset[idx]
            messages = sample['messages']
            ground_truth = _normalize_sid_text(messages[-1]['content'])

            prompt = _apply_chat_template_no_think(
                self.tokenizer,
                messages[:-1],
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors='pt').to(model.device)
            prefix_len = inputs.input_ids.shape[1]

            processor = TrieConstrainedLogitsProcessor(
                trie=self.trie,
                prefix_length=prefix_len,
                eos_token_id=self.tokenizer.eos_token_id,
                allow_early_stop=True,
            )

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([processor]),
                )
            latencies.append((time.time() - t0) * 1000)

            candidates, seen = [], set()
            for beam_idx in range(outputs.shape[0]):
                pred = self.tokenizer.decode(
                    outputs[beam_idx][prefix_len:], skip_special_tokens=True).strip()
                normalized_pred = _normalize_sid_text(pred)
                if normalized_pred not in seen:
                    seen.add(normalized_pred)
                    candidates.append(normalized_pred)

            top1 = candidates[0] if candidates else ''
            top_k_cands = candidates[:self.top_k]

            if top1 == ground_truth:
                exact_match += 1
            if top1 in self.sid_set:
                valid_sid += 1
            if ground_truth in top_k_cands:
                hit += 1
                rank = top_k_cands.index(ground_truth) + 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)

        total = self.num_samples
        gen_metrics = {
            'gen_exact_match':       exact_match / total,
            f'gen_hit@{self.top_k}': hit / total,
            f'gen_mrr@{self.top_k}': mrr_sum / total,
            f'gen_ndcg@{self.top_k}': ndcg_sum / total,
            'gen_valid_sid_rate':    valid_sid / total,
            'gen_avg_latency_ms':    sum(latencies) / len(latencies),
        }

        # 打印到控制台
        step = state.global_step
        print(f"\n=== Generative Eval @ step {step} (n={total}) ===")
        for k, v in gen_metrics.items():
            print(f"  {k}: {v:.4f}")

        # 写入 TensorBoard
        if state.is_world_process_zero:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = f"{args.output_dir}/logs"
                writer = SummaryWriter(log_dir=log_dir, filename_suffix='.gen')
                for k, v in gen_metrics.items():
                    writer.add_scalar(f'eval/{k}', v, step)
                writer.close()
            except Exception:
                pass


def build_hf_dataset(base_dataset, tokenizer, max_samples=None):
    """\u5c06 LLMFinetuneDataset 转换为 HuggingFace Dataset（包含格式化后的文本字段）"""
    n = min(max_samples, len(base_dataset)) if max_samples else len(base_dataset)
    texts = []
    for i in range(n):
        sample = base_dataset[i]  # 返回 {messages: [...]}
        texts.append(
            _apply_chat_template_no_think(
                tokenizer,
                sample['messages'],
                add_generation_prompt=False,
            )
        )
    return Dataset.from_dict({'text': texts})


class LLMFinetune:
    """LLM 微调训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.accelerator = Accelerator()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载 tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # 加载模型
        self.model = self._load_model()
        
        # 准备数据集
        self.train_dataset, self.val_dataset, self.test_dataset = prepare_datasets()
        
    def _load_tokenizer(self):
        """加载 tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG['base_model'],
            trust_remote_code=True,
            use_fast=False  # 使用慢速 tokenizer 避免 tokenizers 版本兼容问题
        )
        
        # Llama 3 需要设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
        
    def _load_model(self):
        """加载模型并配置 QLoRA"""
        # BitsAndBytes 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=MODEL_CONFIG['load_in_4bit'],
            bnb_4bit_quant_type=MODEL_CONFIG['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, MODEL_CONFIG['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=MODEL_CONFIG['bnb_4bit_use_double_quant']
        )

        device_map = None
        if torch.cuda.is_available():
            # 4bit/8bit 量化训练时，模型加载设备必须与训练设备一致
            # 使用 current_device() 可避免 LOCAL_RANK 与单卡运行不一致导致的报错
            device_map = {"": torch.cuda.current_device()}
        
        requested_attn_impl = MODEL_CONFIG.get('attn_implementation', 'flash_attention_2')
        fallback_attn_impl = MODEL_CONFIG.get('fallback_attn_implementation', 'sdpa')

        def _load_with_attn(attn_impl: str):
            kwargs = {
                'quantization_config': bnb_config,
                'device_map': device_map,
                'trust_remote_code': True,
            }
            if attn_impl:
                kwargs['attn_implementation'] = attn_impl
            return AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG['base_model'],
                **kwargs,
            )

        try:
            model = _load_with_attn(requested_attn_impl)
            print(f"Attention backend: {requested_attn_impl}")
        except Exception as err:
            err_msg = str(err)
            is_flash_attn_issue = (
                requested_attn_impl == 'flash_attention_2'
                and (
                    'flash_attn' in err_msg
                    or 'undefined symbol' in err_msg
                    or 'cannot import name' in err_msg
                )
            )
            if not is_flash_attn_issue:
                raise

            warnings.warn(
                "flash_attention_2 load failed; fallback to "
                f"'{fallback_attn_impl}'. Original error: {err_msg}",
                RuntimeWarning,
            )
            model = _load_with_attn(fallback_attn_impl)
            print(f"Attention backend: {fallback_attn_impl} (fallback)")
        
        # 准备模型进行 k-bit 训练
        model = prepare_model_for_kbit_training(model)
        
        # LoRA 配置
        peft_config = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=LORA_CONFIG['target_modules'],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type=LORA_CONFIG['task_type']
        )
        
        # 应用 LoRA
        model = get_peft_model(model, peft_config)
        
        # 打印可训练参数
        model.print_trainable_parameters()
        
        return model
        
    def train(self):
        """开始训练"""
        print("Preparing HuggingFace datasets...")
        train_ds = build_hf_dataset(self.train_dataset, self.tokenizer, max_samples=25000)
        val_ds = build_hf_dataset(self.val_dataset, self.tokenizer, max_samples=5000)
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

        supported_sft_fields = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        max_seq_length = TRAINING_CONFIG.get('max_seq_length')
        max_length = TRAINING_CONFIG.get('max_length', max_seq_length)

        sft_config_kwargs = {
            'output_dir': TRAINING_CONFIG['output_dir'],
            'num_train_epochs': TRAINING_CONFIG['num_train_epochs'],
            'per_device_train_batch_size': TRAINING_CONFIG['per_device_train_batch_size'],
            'per_device_eval_batch_size': TRAINING_CONFIG['per_device_eval_batch_size'],
            'gradient_accumulation_steps': TRAINING_CONFIG['gradient_accumulation_steps'],
            'learning_rate': TRAINING_CONFIG['learning_rate'],
            'warmup_steps': TRAINING_CONFIG['warmup_steps'],
            'logging_steps': TRAINING_CONFIG['logging_steps'],
            'eval_steps': TRAINING_CONFIG['eval_steps'],
            'save_steps': TRAINING_CONFIG['save_steps'],
            'bf16': TRAINING_CONFIG['bf16'],
            'optim': TRAINING_CONFIG['optim'],
            'gradient_checkpointing': TRAINING_CONFIG['gradient_checkpointing'],
            'ddp_find_unused_parameters': False,
            'remove_unused_columns': False,
            'dataloader_num_workers': 4,
            'eval_strategy': 'steps',
            'save_strategy': 'steps',
            'load_best_model_at_end': True,
            'save_total_limit': TRAINING_CONFIG.get('save_total_limit', 3),
            'report_to': 'tensorboard',
            'logging_dir': f"{TRAINING_CONFIG['output_dir']}/logs",
            'dataset_text_field': 'text',
            'packing': True,
        }

        if 'max_seq_length' in supported_sft_fields:
            sft_config_kwargs['max_seq_length'] = max_seq_length
        elif 'max_length' in supported_sft_fields:
            sft_config_kwargs['max_length'] = max_length

        sft_config_kwargs = {
            key: value
            for key, value in sft_config_kwargs.items()
            if key in supported_sft_fields and value is not None
        }

        # 训练参数：对不同 trl 版本做字段兼容
        training_args = SFTConfig(**sft_config_kwargs)

        # 构建生成式评测 Callback
        with open(DATA_CONFIG['semantic_ids_path'], 'r', encoding='utf-8') as f:
            sid_map = json.load(f)
        sid_set = {_to_angle_bracket_sid(sid) for sid in sid_map.values()}
        trie = TokenTrie()
        for sid in sid_set:
            trie.add(self.tokenizer.encode(sid, add_special_tokens=False))

        gen_eval_cb = GenerativeEvalCallback(
            tokenizer=self.tokenizer,
            test_dataset=self.test_dataset,
            sid_set=sid_set,
            trie=trie,
            num_samples=TRAINING_CONFIG.get('mid_eval_samples', 30),
            num_beams=TRAINING_CONFIG.get('eval_num_beams', 5),
            top_k=TRAINING_CONFIG.get('eval_top_k', 5),
            max_new_tokens=TRAINING_CONFIG.get('eval_max_new_tokens', 20),
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,  # trl 0.16+ 改名为 processing_class
            callbacks=[gen_eval_cb],
        )
        
        # 开始训练
        print("\n=== Starting Training ===")
        trainer.train()
        
        # 保存最终模型
        final_path = f"{TRAINING_CONFIG['output_dir']}/final_model"
        trainer.save_model(final_path)
        print(f"\n=== Model saved to {final_path} ===")
        
        return trainer
        
    def evaluate(self, trainer: SFTTrainer = None):
        """评估模型"""
        if trainer is None:
            # 加载已保存的模型
            model_path = f"{TRAINING_CONFIG['output_dir']}/final_model"
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            
        # 评估
        test_ds = build_hf_dataset(self.test_dataset, self.tokenizer)
        
        print("\n=== Evaluating on Test Set ===")
        results = trainer.evaluate(eval_dataset=test_ds)
        print(f"Test Loss: {results['eval_loss']:.4f}")

        with open(DATA_CONFIG['semantic_ids_path'], 'r', encoding='utf-8') as file:
            sid_map = json.load(file)
        semantic_sid_set = {_to_angle_bracket_sid(sid) for sid in sid_map.values()}

        trie = TokenTrie()
        for sid in semantic_sid_set:
            token_ids = self.tokenizer.encode(sid, add_special_tokens=False)
            trie.add(token_ids)

        eval_generation_samples = min(
            len(self.test_dataset),
            TRAINING_CONFIG.get('eval_generation_samples', 200),
        )
        num_beams = TRAINING_CONFIG.get('eval_num_beams', 5)
        top_k = TRAINING_CONFIG.get('eval_top_k', 5)
        max_new_tokens = TRAINING_CONFIG.get('eval_max_new_tokens', 20)

        exact_match = 0
        hit_at_k = 0
        recall_at_k = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        valid_sid = 0
        latencies = []

        self.model.eval()
        for index in range(eval_generation_samples):
            sample = self.test_dataset[index]
            messages = sample['messages']
            prompt_messages = messages[:-1]
            ground_truth = _normalize_sid_text(messages[-1]['content'])

            prompt = _apply_chat_template_no_think(
                self.tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            prefix_len = inputs.input_ids.shape[1]

            processor = TrieConstrainedLogitsProcessor(
                trie=trie,
                prefix_length=prefix_len,
                eos_token_id=self.tokenizer.eos_token_id,
                allow_early_stop=True,
            )

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([processor]),
                )
            latencies.append((time.time() - start_time) * 1000)

            candidates = []
            seen = set()
            for beam_index in range(outputs.shape[0]):
                predicted = self.tokenizer.decode(
                    outputs[beam_index][prefix_len:],
                    skip_special_tokens=True,
                ).strip()
                normalized_pred = _normalize_sid_text(predicted)
                if normalized_pred not in seen:
                    seen.add(normalized_pred)
                    candidates.append(normalized_pred)

            top_1 = candidates[0] if candidates else ''
            top_k_candidates = candidates[:top_k]

            if top_1 == ground_truth:
                exact_match += 1
            if top_1 in semantic_sid_set:
                valid_sid += 1
            if ground_truth in top_k_candidates:
                hit_at_k += 1
                recall_at_k += 1
                rank = top_k_candidates.index(ground_truth) + 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)

        total = eval_generation_samples
        results['eval_exact_match_rate'] = exact_match / total
        results[f'eval_hit@{top_k}'] = hit_at_k / total
        results[f'eval_recall@{top_k}'] = recall_at_k / total
        results[f'eval_mrr@{top_k}'] = mrr_sum / total
        results[f'eval_ndcg@{top_k}'] = ndcg_sum / total
        results['eval_valid_sid_rate'] = valid_sid / total
        results['eval_avg_latency_ms'] = sum(latencies) / len(latencies)

        print("\n=== Generation Metrics ===")
        print(f"eval_samples: {total}")
        print(f"exact_match_rate: {results['eval_exact_match_rate']:.4f}")
        print(f"hit@{top_k}: {results[f'eval_hit@{top_k}']:.4f}")
        print(f"recall@{top_k}: {results[f'eval_recall@{top_k}']:.4f}")
        print(f"mrr@{top_k}: {results[f'eval_mrr@{top_k}']:.4f}")
        print(f"ndcg@{top_k}: {results[f'eval_ndcg@{top_k}']:.4f}")
        print(f"valid_sid_rate: {results['eval_valid_sid_rate']:.4f}")
        print(f"avg_latency_ms: {results['eval_avg_latency_ms']:.2f}")
        
        return results
        
    def generate_sample(self, sample_idx: int = 0):
        """生成样本预测"""
        sample = self.test_dataset[sample_idx]
        messages = sample['messages']
        
        # 只使用 system + user 消息
        prompt_messages = messages[:-1]
        ground_truth = messages[-1]['content']
        
        # 格式化 prompt
        prompt = _apply_chat_template_no_think(
            self.tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # Semantic ID 很短
                temperature=0.1,
                do_sample=True,
                top_p=0.9
            )
            
        # 解码
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的 Semantic ID（在 prompt 之后的部分）
        predicted = generated[len(prompt):].strip()
        
        print("\n=== Sample Generation ===")
        print(f"Prompt:\n{prompt}\n")
        print(f"Ground Truth: {ground_truth}")
        print(f"Predicted: {predicted}")
        
        return predicted, ground_truth


def main():
    """主训练流程"""
    parser = argparse.ArgumentParser(description='LLM finetune training entry')
    parser.add_argument('--strict_kcore', action='store_true', help='训练前执行严格 k-core 闭包过滤')
    parser.add_argument('--k_core', type=int, default=None, help='严格 k-core 的 k 值（默认取 config）')
    parser.add_argument('--k_core_output_dir', type=str, default=None, help='严格 k-core 输出目录')
    parser.add_argument('--k_core_no_cache', action='store_true', help='严格 k-core 不复用缓存，强制重算')
    parser.add_argument('--min_user_interactions', type=int, default=None, help='用户最小有效访问数阈值（默认取 config）')
    parser.add_argument('--dataset_dir', type=str, default=None, help='训练数据目录（可覆盖 config）')
    parser.add_argument('--semantic_ids_path', type=str, default=None, help='semantic_ids.json 路径（可覆盖 config）')
    parser.add_argument('--preprocess_pipeline', type=str, default=None, choices=['legacy', 'yelp_session'], help='数据预处理流程：legacy 或 yelp_session')
    parser.add_argument('--session_enable_filter_low_frequency', action='store_true', help='启用 Session 低频 POI/User 过滤')
    parser.add_argument('--session_min_poi_freq', type=int, default=None, help='Session POI 最小频次阈值（严格 > 阈值才保留）')
    parser.add_argument('--session_min_user_freq', type=int, default=None, help='Session User 最小频次阈值（严格 > 阈值才保留）')
    parser.add_argument('--no_session_remove_isolated_24h', action='store_true', help='关闭 Session 24h 双侧孤立访问剔除')
    parser.add_argument('--session_time_interval_min', type=int, default=None, help='Session 会话切分间隔（分钟）')
    parser.add_argument('--no_session_ignore_singleton_sessions', action='store_true', help='关闭 Session 单点会话忽略')
    parser.add_argument('--no_session_remove_unseen_user_poi', action='store_true', help='关闭 Session 冷启动过滤')
    parser.add_argument('--force_rebuild_cache', action='store_true', help='训练前删除 dataset cache 强制重建')
    args = parser.parse_args()

    if args.dataset_dir:
        DATA_CONFIG['dataset_dir'] = str(Path(args.dataset_dir).resolve())
    if args.semantic_ids_path:
        DATA_CONFIG['semantic_ids_path'] = str(Path(args.semantic_ids_path).resolve())
    if args.min_user_interactions is not None:
        DATA_CONFIG['min_user_interactions'] = max(2, int(args.min_user_interactions))
    if args.preprocess_pipeline is not None:
        DATA_CONFIG['preprocess_pipeline'] = args.preprocess_pipeline
    if args.session_enable_filter_low_frequency:
        DATA_CONFIG['session_enable_filter_low_frequency'] = True
    if args.session_min_poi_freq is not None:
        DATA_CONFIG['session_min_poi_freq'] = int(args.session_min_poi_freq)
    if args.session_min_user_freq is not None:
        DATA_CONFIG['session_min_user_freq'] = int(args.session_min_user_freq)
    if args.no_session_remove_isolated_24h:
        DATA_CONFIG['session_remove_isolated_24h'] = False
    if args.session_time_interval_min is not None:
        DATA_CONFIG['session_time_interval_min'] = int(args.session_time_interval_min)
    if args.no_session_ignore_singleton_sessions:
        DATA_CONFIG['session_ignore_singleton_sessions'] = False
    if args.no_session_remove_unseen_user_poi:
        DATA_CONFIG['session_remove_unseen_user_poi'] = False

    DATA_CONFIG['enable_strict_kcore'] = bool(args.strict_kcore)
    if args.k_core is not None:
        DATA_CONFIG['k_core'] = int(args.k_core)
    if args.k_core_output_dir:
        DATA_CONFIG['k_core_output_dir'] = str(Path(args.k_core_output_dir).resolve())
    if args.k_core_no_cache:
        DATA_CONFIG['k_core_use_cache'] = False

    if args.force_rebuild_cache:
        cache_dir = Path(DATA_CONFIG['cache_dir'])
        for filename in [
            'train_samples.json', 'val_samples.json', 'test_samples.json',
            'train_prompts.json', 'val_prompts.json', 'test_prompts.json', 'schema.txt',
        ]:
            path = cache_dir / filename
            if path.exists():
                path.unlink()

    # 初始化训练器
    llm_finetune = LLMFinetune()
    
    # 训练
    trainer = llm_finetune.train()
    
    # 评估
    llm_finetune.evaluate(trainer)
    
    # 生成样本
    for i in range(5):
        print(f"\n{'='*50}")
        llm_finetune.generate_sample(i)


if __name__ == '__main__':
    main()
