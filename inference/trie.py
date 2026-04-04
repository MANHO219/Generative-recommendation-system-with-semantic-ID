from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class TokenTrie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def add(self, token_ids: Iterable[int]) -> None:
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.is_end = True

    def traverse(self, token_ids: Iterable[int]) -> Optional[TrieNode]:
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node

    def next_tokens(self, token_ids: Iterable[int]) -> List[int]:
        node = self.traverse(token_ids)
        if node is None:
            return []
        return list(node.children.keys())
