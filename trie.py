# Write function to process dataset into usable data model
# Construct a prefix tree (trie) from the dataset

import os.path
import json
import pickle
from nltk.tokenize import sent_tokenize


class TrieNode(object):

    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.is_end_of_sentence = False


class Trie(object):

    def __init__(self, root: TrieNode):
        self.root = root

    def add_sentence(self, root: TrieNode, sentence: str):

        character = sentence[0]

        for child in root.children:

            # If the leading char in the string is already a child, no need to add a new node
            if character == child.char:

                if len(sentence) == 1:
                    child.is_end_of_sentence = True
                    return self

                return self.add_sentence(child, sentence[1:])

        node = TrieNode(character)

        if len(sentence) == 1:
            node.is_end_of_sentence = True
            root.children.append(node)
            return self

        root.children.append(node)
        return self.add_sentence(node, sentence[1:])

    def return_completions_from_node(self, node: TrieNode, prefix=""):

        def enumerate_sentences(node: TrieNode, sentence: str, sentences: list, prefix):

            if len(node.children) > 0:
                for child in node.children:
                    if child.is_end_of_sentence:
                        sentences.append(prefix + sentence + child.char)
                    if len(child.children) > 0:
                        enumerate_sentences(child, sentence + child.char, sentences, prefix)
                    else:
                        enumerate_sentences(child, sentence, sentences, prefix)

            return sentences

        if node == None:
            return []

        return enumerate_sentences(node, "", [], prefix)

    def contains(self, root: TrieNode, sentence: str):
        """
        Returns (True, last node visited) if a sentence exists in the trie, (False, None) otherwise
        """
        character = sentence[0]

        for child in root.children:
            if character == child.char:
                if len(sentence) == 1:
                    return (True, child)
                return self.contains(child, sentence[1:])

        return (False, None)


def extract_sentences_from_json(file_path: str):

    sentences = []

    with open(file_path) as json_data:
        data = json.load(json_data)

    for issues in data["Issues"]:
        for message in issues["Messages"]:
            text = message["Text"]
            sentences.append(text)

            # If the text contains multiple sentences, add each individual sentence to the dataset to be added to the trie
            sub_sentences = sent_tokenize(text)
            if len(sub_sentences) > 1:
                sentences.extend(sent_tokenize(text))

    return sentences
