# Write function to autocomplete a sentence given prefix

from trie import Trie


def autocomplete(trie: Trie, prefix: str):
    """
    TODO: add helpful comments
    """
    # Number of completions to return
    n = 3
    (contains, node) = trie.contains(trie.root, prefix)
    if contains:
        return trie.return_completions_from_node(node, prefix=prefix)[:n]
    # else:
        # return rnn.return_completions(...)
