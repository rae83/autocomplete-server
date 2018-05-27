import tornado.ioloop
import tornado.web
import os
import pickle
from trie import Trie, TrieNode, extract_sentences_from_json, save_sentences_to_file


class autocomplete_handler(tornado.web.RequestHandler):

    def autocomplete(self, trie: Trie, prefix: str):
        """
        Check if prefix is in trie. If yes, then autocomplete using trie.  Else, use RNN model.
        n: Number of completions to return
        Return: list of strings, where each element is a possible completion.
        """
        n = 3
        (contains, node) = trie.contains(trie.root, prefix)
        if contains:
            return trie.return_completions_from_node(node, prefix=prefix)[:n]
        # else:
            # return rnn.return_completions(...)

    def get(self):
        """
        Parse args from URL and return autocompletions as JSON.
        """
        args = self.get_arguments("q")[0]
        response = {"Completions": self.autocomplete(trie, args)}
        self.write(response)
        self.write('\n')


def make_app():
    """
    Initialize server with one endpoint for sentence autocomplete.
    """
    return tornado.web.Application([
        (r"/autocomplete", autocomplete_handler),
    ])


if __name__ == "__main__":

    # Initialize the prefix trie model for autocompletion
    trie_file_path = b"models/trie.obj"
    sentences_file_path = "models/sentences.txt"

    # If pickled file with trie exists, load the model. Else, create the model from the sentences
    if os.path.isfile(trie_file_path):
        file = open(trie_file_path, 'rb')
        trie = pickle.load(file)
        root = trie.root
    else:
        import sys
        import nltk
        nltk.download("punkt")
        from nltk.tokenize import sent_tokenize

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

    # Start the server
    app = make_app()
    app.listen(13000)
    tornado.ioloop.IOLoop.current().start()
