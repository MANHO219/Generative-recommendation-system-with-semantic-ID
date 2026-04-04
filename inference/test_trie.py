from trie import TokenTrie


def test_trie():
    trie = TokenTrie()
    trie.add([1, 2, 3])
    trie.add([1, 2, 4])

    assert trie.next_tokens([]) == [1]
    assert set(trie.next_tokens([1, 2])) == {3, 4}
    assert trie.next_tokens([9]) == []

    node = trie.traverse([1, 2, 3])
    assert node is not None and node.is_end


if __name__ == "__main__":
    test_trie()
    print("Trie tests passed.")
