# Write function to process dataset into usable data model

# Construct a prefix tree (trie) from the dataset


class TrieNode():

    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.end_of_sentence = False


class Trie():

    def __init__(self, root: TrieNode):
        self.root = root

    def create_trie_from_file(self):
        return

    def add_sentence(self, root: TrieNode, sentence: list):

        character = sentence[0]

        for child in root.children:

            # if the leading char in the string is already a child, no need to add a new node
            if character == child.char:
                return self.add_sentence(child, sentence[1:])

        node = TrieNode(character)

        if len(sentence) == 1:
            node.end_of_sentence = True
            return self

        root.children.append(node)
        return self.add_sentence(node, sentence[1:])


if __name__ == "__main__":

    root = TrieNode('')
    trie = Trie(root)

    for sentence in ["What is your account number?", "What is your address?", "What is your order number?"]:
        trie.add_sentence(root, sentence)
    
    print(trie)
