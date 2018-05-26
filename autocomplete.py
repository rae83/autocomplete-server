# Write function to autocomplete a sentence given prefix

from preprocess import Trie


def autocomplete(trie: Trie, prefix: list):
    """
    TODO: add helpful comments
    """
    (contains, node) = trie.contains(trie.root, prefix)
    if contains:
        return trie.return_completions_from_node(node)
    # else:
        # return rnn.return_completions(...)
