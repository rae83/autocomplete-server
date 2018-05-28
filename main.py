import tornado.ioloop
import tornado.web
import os
import pickle
from keras.models import load_model
from trie import Trie, TrieNode, extract_sentences_from_json, save_sentences_to_file, initialize_prefix_trie
from rnn import generate_text


class autocomplete_handler(tornado.web.RequestHandler):

    def autocomplete(self, trie: Trie, model, prefix: str):
        """
        Check if prefix is in trie. If yes, then autocomplete using trie.  Else, use RNN model.
        n: Number of completions to return
        Return: list of strings, where each element is a possible completion.
        """
        n = 3
        (contains, node) = trie.contains(trie.root, prefix)
        if contains:
            return trie.return_completions_from_node(node, prefix=prefix)[:n]
        else:
            return generate_text(model, prefix)

    def get(self):
        """
        Parse args from URL and return autocompletions as JSON.
        """
        args = self.get_arguments("q")[0]
        response = {"Completions": self.autocomplete(trie, rnn, args)}
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

    trie = initialize_prefix_trie()
    rnn = load_model("rnn/checkpoints/model.ckpt")

    # Start the server
    app = make_app()
    app.listen(13000)
    tornado.ioloop.IOLoop.current().start()
