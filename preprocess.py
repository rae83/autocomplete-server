# Write function to process dataset into usable data model

# Construct a prefix tree (trie) from the dataset

import json
import pickle


class TrieNode():

    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.end_of_sentence = False


class Trie():

    def __init__(self, root: TrieNode):
        self.root = root

    def add_sentence(self, root: TrieNode, sentence: list):

        character = sentence[0]

        for child in root.children:

            # if the leading char in the string is already a child, no need to add a new node
            if character == child.char:

                if len(sentence) == 1: # is this block correct?
                    child.end_of_sentence = True
                    return self

                return self.add_sentence(child, sentence[1:])

        node = TrieNode(character)

        if len(sentence) == 1:
            node.end_of_sentence = True
            root.children.append(node)
            return self

        root.children.append(node)
        return self.add_sentence(node, sentence[1:])

    def contains(self, root: TrieNode, sentence: list):

        character = sentence[0]

        for child in root.children:
            if character == child.char:
                if len(sentence) == 1:
                    return True
                return self.contains(child, sentence[1:])

        return False


def extract_sentences_from_json(file_path: str):

    sentences = []

    with open(file_path) as json_data:
        data = json.load(json_data)
    
    for issues in data["Issues"]:
        for message in issues["Messages"]:
            sentences.append(message["Text"])
    
    return sentences



if __name__ == "__main__":

    # code below is currently just for sanity checks
    root = TrieNode('')
    trie = Trie(root)

    sentences = extract_sentences_from_json("sample_conversations.json") # TODO: pickle this so it can be reloaded instead of recomputed each time
    
    for sentence in sentences:
        trie.add_sentence(root, sentence)

    print(trie.contains(root, "Hi! I placed an order on your website and I can't find the tracking number. Can you help me find out where my package is"))
    print(trie.contains(root, "Hi! I placed an order on your website and I can't find the tracking number. Can you help me find out where my package is?"))
    print(trie.contains(root, "Hi! I placed an order on your NOT website and I can't find the tracking number. Can you help me find out where my package is?"))
