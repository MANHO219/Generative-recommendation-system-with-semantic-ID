import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from constrained_decoding import TrieConstrainedLogitsProcessor
from trie import TokenTrie

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_finetune.config import PROMPT_TEMPLATE


def build_trie(tokenizer, semantic_ids: List[str]) -> TokenTrie:
    trie = TokenTrie()
    for sid in semantic_ids:
        token_ids = tokenizer.encode(sid, add_special_tokens=False)
        trie.add(token_ids)
    return trie


def load_semantic_ids(path: str) -> List[str]:
    with open(path, "r") as f:
        data = json.load(f)
    return list({sid for sid in data.values()})


def get_time_description(dt: datetime) -> str:
    hour = dt.hour
    if 6 <= hour < 12:
        return f"{dt.strftime('%A')}, Morning ({hour}:00)"
    if 12 <= hour < 18:
        return f"{dt.strftime('%A')}, Afternoon ({hour}:00)"
    if 18 <= hour < 22:
        return f"{dt.strftime('%A')}, Evening ({hour}:00)"
    return f"{dt.strftime('%A')}, Night ({hour}:00)"


def get_day_type(dt: datetime) -> str:
    return "Weekend" if dt.weekday() >= 5 else "Weekday"


def get_pluscode(lat: float, lon: float) -> str:
    lat_code = f"{int(lat * 100) % 100:02d}"
    lon_code = f"{int(abs(lon) * 100) % 100:02d}"
    return f"{lat_code}{lon_code}+XX"


def format_history_items(history: List[Dict[str, Any]]) -> str:
    items = []
    for idx, visit in enumerate(history, 1):
        item = (
            f"{idx}. {visit['business_name']} "
            f"({visit['categories'][:50]}) - "
            f"Rated {visit['stars']:.1f} stars - "
            f"{visit['date'].strftime('%Y-%m-%d')}"
        )
        items.append(item)
    return "\n".join(items)


def get_favorite_categories(history: List[Dict[str, Any]], top_k: int = 3) -> str:
    category_counts: Dict[str, int] = {}
    for visit in history:
        cats = visit.get("categories", "").split(", ") if visit.get("categories") else []
        for cat in cats:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    top_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ", ".join([cat for cat, _ in top_cats]) if top_cats else "N/A"


def format_prompt(sample: Dict[str, Any]) -> str:
    user_info = sample["user_info"]
    history = sample["history"]
    target = sample["target"]

    if isinstance(target["date"], str):
        target["date"] = datetime.fromisoformat(target["date"])
    for item in history:
        if isinstance(item["date"], str):
            item["date"] = datetime.fromisoformat(item["date"])

    user_prompt = PROMPT_TEMPLATE["user_template"].format(
        user_id=sample["user_id"][:8],
        review_count=user_info["review_count"],
        average_stars=user_info["average_stars"],
        favorite_categories=get_favorite_categories(history),
    )

    context_prompt = PROMPT_TEMPLATE["context_template"].format(
        pluscode=get_pluscode(target["latitude"], target["longitude"]),
        time_description=get_time_description(target["date"]),
        day_type=get_day_type(target["date"]),
    )

    history_prompt = PROMPT_TEMPLATE["history_template"].format(
        count=len(history),
        history_items=format_history_items(history),
    )

    return (
        f"{user_prompt}\n"
        f"{context_prompt}\n"
        f"{history_prompt}\n\n"
        f"{PROMPT_TEMPLATE['instruction']}"
    )


def build_chat_prompt(tokenizer, sample: Dict[str, Any]) -> str:
    user_prompt = format_prompt(sample)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": PROMPT_TEMPLATE["system"]},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--semantic_ids_path", type=str, required=True)
    parser.add_argument("--sample_json", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.sample_json:
        sample = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
    else:
        sample = {
            "user_id": "demo_user_123456",
            "user_info": {"review_count": 120, "average_stars": 4.1},
            "history": [
                {
                    "business_name": "Starbucks",
                    "categories": "Coffee & Tea, Cafes",
                    "stars": 4.0,
                    "date": "2024-11-01",
                },
                {
                    "business_name": "Burger King",
                    "categories": "Fast Food, Burgers",
                    "stars": 3.5,
                    "date": "2024-11-03",
                },
                {
                    "business_name": "McDonald's",
                    "categories": "Fast Food, Burgers",
                    "stars": 3.0,
                    "date": "2024-11-05",
                },
            ],
            "target": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "date": "2024-11-06",
            },
        }

    prompt = build_chat_prompt(tokenizer, sample)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    semantic_ids = load_semantic_ids(args.semantic_ids_path)
    trie = build_trie(tokenizer, semantic_ids)

    prefix_len = inputs.input_ids.shape[1]
    processor = TrieConstrainedLogitsProcessor(
        trie=trie,
        prefix_length=prefix_len,
        eos_token_id=tokenizer.eos_token_id,
        allow_early_stop=True,
    )

    logits_processors = LogitsProcessorList([processor])

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        logits_processor=logits_processors,
    )

    generated = tokenizer.decode(outputs[0][prefix_len:], skip_special_tokens=True)
    print(generated.strip())


if __name__ == "__main__":
    main()
