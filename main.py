# Set up REST server with one endpoint for sentence autocomplete

import tornado.ioloop
import tornado.web
import os
import pickle
from preprocess import Trie, TrieNode, extract_sentences_from_json
from autocomplete import autocomplete


class autocomplete_handler(tornado.web.RequestHandler):
    """
    TODO: add helpful comments
    """

    def get(self):
        """
        TODO: add helpful comments
        """
        args = self.get_arguments("q")[0]
        response = {"Completions": autocomplete(trie, args)}
        self.write(response)
        self.write('\n')


def make_app():
    """
    Initialize the /autocomplete endpoint
    """
    return tornado.web.Application([
        (r"/autocomplete", autocomplete_handler),
    ])


if __name__ == "__main__":

    # Initialize the prefix trie model for autocompletion
    file_path = b"models/trie.obj"

    # If pickled file with trie exists, load the model. Else, create the model from the sentences
    if os.path.isfile(file_path):
        file = open(file_path, 'rb')
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

        for sentence in sentences:
            trie.add_sentence(root, sentence)

        sys.setrecursionlimit(5000)
        filehandler = open(file_path, "wb")
        pickle.dump(trie, filehandler)
        filehandler.close()

    # Start the server
    app = make_app()
    app.listen(13000)
    tornado.ioloop.IOLoop.current().start()
