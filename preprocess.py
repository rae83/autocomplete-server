# Write function to process dataset into usable data model

# Construct a prefix tree (trie) from the dataset

import json
import pickle
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize


class TrieNode():

    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.is_end_of_sentence = False


class Trie():

    def __init__(self, root: TrieNode):
        self.root = root


    def add_sentence(self, root: TrieNode, sentence: list):

        character = sentence[0]

        for child in root.children:

            # if the leading char in the string is already a child, no need to add a new node
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
    

    def return_completions_from_node(self, node: TrieNode):

        def enumerate_sentences(node: TrieNode, sentence: str, sentences: list):
            
            if len(node.children) > 0:
                for child in node.children:
                    sentence += child.char
                    if child.is_end_of_sentence:
                        sentences.append(sentence)
                    enumerate_sentences(child, sentence, sentences)
            
            return sentences

        if node == None:
            return []

        return enumerate_sentences(node, "", [])


    def contains(self, root: TrieNode, sentence: list):
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

            # if the text contains multiple sentences, add each individual sentence to the dataset to be added to the trie
            sub_sentences = sent_tokenize(text)
            if len(sub_sentences) > 1:
                sentences.extend(sent_tokenize(text))

    return sentences


if __name__ == "__main__":

    # code below is currently just for sanity checks
    root = TrieNode('')
    trie = Trie(root)

    sentences = extract_sentences_from_json("sample_conversations.json")

    for sentence in sentences:
        trie.add_sentence(root, sentence)
    
    # TODO: pickle the trie so it can be reloaded instead of recomputed each time

    (boolean_1, node_1) = trie.contains(root, "Hi! I placed an order on your website and I can't find the tracking number. Can you help me")
    (boolean_2, node_2) = trie.contains(root, "xxxx") # NOTE: can't call return_completions_from_node if node is None
    (boolean_3, node_3) = trie.contains(root, "Hi! I placed an order on your NOT website and I can't find the tracking number. Can you help me find out where my package is?")

    print(boolean_1, node_1)
    print(boolean_2, node_2)
    print(boolean_3, node_3)

    print(trie.return_completions_from_node(node_1))
    print(trie.return_completions_from_node(node_2))
