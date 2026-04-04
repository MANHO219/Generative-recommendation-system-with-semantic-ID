from typing import List, Optional

import torch
from transformers import LogitsProcessor

from inference.trie import TokenTrie


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        trie: TokenTrie,
        prefix_length: int,
        eos_token_id: Optional[int] = None,
        allow_early_stop: bool = True,
    ) -> None:
        self.trie = trie
        self.prefix_length = prefix_length
        self.eos_token_id = eos_token_id
        self.allow_early_stop = allow_early_stop

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]

        for batch_idx in range(batch_size):
            generated = input_ids[batch_idx, self.prefix_length :].tolist()
            node = self.trie.traverse(generated)

            if node is None:
                allowed = []
            else:
                allowed = list(node.children.keys())
                if self.allow_early_stop and node.is_end and self.eos_token_id is not None:
                    allowed.append(self.eos_token_id)

            if not allowed and self.eos_token_id is not None:
                allowed = [self.eos_token_id]

            mask = torch.full((vocab_size,), float("-inf"), device=scores.device)
            mask[allowed] = 0.0
            scores[batch_idx] = scores[batch_idx] + mask

        return scores
