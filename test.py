import string
import random
import unittest
from trie import Trie, TrieNode


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.common_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        self.ending_1 = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        self.ending_2 = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        self.string_1 = self.common_prefix + self.ending_1
        self.string_2 = self.common_prefix + self.ending_2
        self.not_string = self.common_prefix + ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        self.root = TrieNode("")
        self.trie = Trie(self.root)
        self.trie.add_sentence(self.root, self.string_1)
        self.trie.add_sentence(self.root, self.string_2)

    def test_contains(self):
        self.assertTrue(self.trie.contains(self.root, self.string_1)[0])
        self.assertTrue(self.trie.contains(self.root, self.string_2)[0])
        self.assertFalse(self.trie.contains(self.root, self.not_string)[0])

    def test_return_completions_from_node(self):
        node = self.trie.contains(self.root, self.common_prefix)[1]
        completions = self.trie.return_completions_from_node(node)
        # Although the unnittest method is misleadingly named, it actually checks if two arrays contain same elements
        self.assertCountEqual([self.ending_1, self.ending_2], completions)

if __name__ == '__main__':
    unittest.main()