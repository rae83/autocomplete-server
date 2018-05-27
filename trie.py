# Write function to process dataset into usable data model
# Construct a prefix tree (trie) from the dataset

import os.path
import json
import pickle
import sys
import nltk
nltk.download("punkt")
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
        Returns (True, last node visited) if a sentence exists in the trie. Returns (False, None) otherwise
        """
        character = sentence[0]

        for child in root.children:
            if character == child.char:
                if len(sentence) == 1:
                    return (True, child)
                return self.contains(child, sentence[1:])

        return (False, None)


def extract_sentences_from_json(file_path: str):
    """
    Opens the JSON file at a specified path, extracting just the sentence text from objects in the file.
    Return: list of strings.
    """
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


def save_sentences_to_file(sentences: list, file_path: str):
    with open(file_path, 'w') as file_handler:
        for sentence in sentences:
            file_handler.write(sentence + "\n")


def initialize_prefix_trie():
    # Initialize the prefix trie model for autocompletion
    trie_file_path = b"models/trie.obj"
    sentences_file_path = "models/sentences.txt"

    # If pickled file with trie exists, load the model. Else, create the model from the sentences
    if os.path.isfile(trie_file_path):
        file = open(trie_file_path, 'rb')
        trie = pickle.load(file)
        root = trie.root
    else:


        root = TrieNode('')
        trie = Trie(root)

        sentences = extract_sentences_from_json("sample_conversations.json")
        if not os.path.isfile(sentences_file_path):
            save_sentences_to_file(sentences, sentences_file_path)

        for sentence in sentences:
            trie.add_sentence(root, sentence)

        sys.setrecursionlimit(5000)
        filehandler = open(trie_file_path, "wb")
        pickle.dump(trie, filehandler)
        filehandler.close()
    
    return trie
