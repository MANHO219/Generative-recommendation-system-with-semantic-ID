"""
LLM 微调训练器

基于 QLoRA 的 Qwen-3-8B-Insturct 指令微调
"""

import torch
import json
import time
import sys
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

from config import MODEL_CONFIG, LORA_CONFIG, TRAINING_CONFIG, DATA_CONFIG
from dataset import prepare_datasets


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
        exact_match = hit = mrr_sum = valid_sid = 0
        latencies = []

        for idx in range(self.num_samples):
            sample = self.test_dataset[idx]
            messages = sample['messages']
            ground_truth = messages[-1]['content'].strip()

            prompt = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)
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
                if pred not in seen:
                    seen.add(pred)
                    candidates.append(pred)

            top1 = candidates[0] if candidates else ''
            top_k_cands = candidates[:self.top_k]

            if top1 == ground_truth:
                exact_match += 1
            if top1 in self.sid_set:
                valid_sid += 1
            if ground_truth in top_k_cands:
                hit += 1
                mrr_sum += 1.0 / (top_k_cands.index(ground_truth) + 1)

        total = self.num_samples
        gen_metrics = {
            'gen_exact_match':       exact_match / total,
            f'gen_hit@{self.top_k}': hit / total,
            f'gen_mrr@{self.top_k}': mrr_sum / total,
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


def build_hf_dataset(base_dataset, max_samples=None):
    """\u5c06 LLMFinetuneDataset 转换为 HuggingFace Dataset（包含格式化后的文本字段）"""
    n = min(max_samples, len(base_dataset)) if max_samples else len(base_dataset)
    texts = []
    for i in range(n):
        sample = base_dataset[i]  # 返回 {messages: [...]}
        # 使用简单的 ChatML 格式将 messages 序列化为文本
        parts = []
        for msg in sample['messages']:
            role = msg['role']
            content = msg['content']
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        texts.append('\n'.join(parts))
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
        
        # 加载模型（启用 Flash Attention 2 加速）
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['base_model'],
            quantization_config=bnb_config,
            device_map={"": f"cuda:{self.accelerator.local_process_index}"} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
        )
        
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
        train_ds = build_hf_dataset(self.train_dataset, max_samples=50000)
        val_ds = build_hf_dataset(self.val_dataset, max_samples=10000)
        print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
        
        # 训练参数：trl 0.16+ 使用 SFTConfig 统一管理所有参数
        training_args = SFTConfig(
            output_dir=TRAINING_CONFIG['output_dir'],
            num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
            per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
            per_device_eval_batch_size=TRAINING_CONFIG['per_device_eval_batch_size'],
            gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
            learning_rate=TRAINING_CONFIG['learning_rate'],
            warmup_steps=TRAINING_CONFIG['warmup_steps'],
            logging_steps=TRAINING_CONFIG['logging_steps'],
            eval_steps=TRAINING_CONFIG['eval_steps'],
            save_steps=TRAINING_CONFIG['save_steps'],
            bf16=TRAINING_CONFIG['bf16'],
            optim=TRAINING_CONFIG['optim'],
            gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing'],
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            eval_strategy='steps',
            save_strategy='steps',
            load_best_model_at_end=True,
            save_total_limit=TRAINING_CONFIG.get('save_total_limit', 3),
            report_to='tensorboard',
            logging_dir=f"{TRAINING_CONFIG['output_dir']}/logs",
            # SFT 专属参数（trl 0.16+ 移入 SFTConfig）
            max_seq_length=TRAINING_CONFIG['max_seq_length'],
            dataset_text_field='text',
            packing=True,
        )

        # 构建生成式评测 Callback
        with open(DATA_CONFIG['semantic_ids_path'], 'r', encoding='utf-8') as f:
            sid_map = json.load(f)
        sid_set = set(sid_map.values())
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
        test_ds = build_hf_dataset(self.test_dataset)
        
        print("\n=== Evaluating on Test Set ===")
        results = trainer.evaluate(eval_dataset=test_ds)
        print(f"Test Loss: {results['eval_loss']:.4f}")

        with open(DATA_CONFIG['semantic_ids_path'], 'r', encoding='utf-8') as file:
            sid_map = json.load(file)
        semantic_sid_set = set(sid_map.values())

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
        valid_sid = 0
        latencies = []

        self.model.eval()
        for index in range(eval_generation_samples):
            sample = self.test_dataset[index]
            messages = sample['messages']
            prompt_messages = messages[:-1]
            ground_truth = messages[-1]['content'].strip()

            prompt = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
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
                if predicted not in seen:
                    seen.add(predicted)
                    candidates.append(predicted)

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

        total = eval_generation_samples
        results['eval_exact_match_rate'] = exact_match / total
        results[f'eval_hit@{top_k}'] = hit_at_k / total
        results[f'eval_recall@{top_k}'] = recall_at_k / total
        results[f'eval_mrr@{top_k}'] = mrr_sum / total
        results['eval_valid_sid_rate'] = valid_sid / total
        results['eval_avg_latency_ms'] = sum(latencies) / len(latencies)

        print("\n=== Generation Metrics ===")
        print(f"eval_samples: {total}")
        print(f"exact_match_rate: {results['eval_exact_match_rate']:.4f}")
        print(f"hit@{top_k}: {results[f'eval_hit@{top_k}']:.4f}")
        print(f"recall@{top_k}: {results[f'eval_recall@{top_k}']:.4f}")
        print(f"mrr@{top_k}: {results[f'eval_mrr@{top_k}']:.4f}")
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
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
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
