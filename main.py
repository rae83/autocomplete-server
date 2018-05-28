import tornado.ioloop
import tornado.web
import os
import pickle
from keras.models import load_model
from trie import Trie, TrieNode, extract_sentences_from_json, save_sentences_to_file, initialize_prefix_trie
from rnn.rnn import generate_text, build_inference_model


class autocomplete_handler(tornado.web.RequestHandler):

    def autocomplete(self, trie: Trie, model, prefix: str):
        """
        Check if prefix is in trie. If yes, then autocomplete using trie.  Else, use RNN(GRU) model.
        Args:
            trie (Trie): the Trie object for completing prefixes seen during training.
            model (Keras model): the RNN(GRU) Keras model for completing novel prefixes.
        Returns: 
            list of strings, where each element is a possible completion.
        """
        # n (int): Number of completions to return.
        n = 3
        (contains, node) = trie.contains(trie.root, prefix)
        if contains:
            return trie.return_completions_from_node(node, prefix=prefix)[:n]
        else:
            completions = []
            for _ in range(n):
                completions.append(generate_text(model, prefix))
            return completions

    def get(self):
        """
        Parse args from URL and return autocompletions as JSON.
        """
        args = self.get_arguments("q")[0]
        response = {"Completions": self.autocomplete(trie, inference_model, args)}
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

    # Load or create the prefix trie for autocompleting sequences seen in training
    trie = initialize_prefix_trie()

    # Load RNN(GRU) model for autocompleting seed sequences not seen before
    model = load_model("checkpoints/model.ckpt")
    inference_model = build_inference_model(model)
    inference_model.set_weights(model.get_weights())

    # Start the server
    app = make_app()
    app.listen(13000)
    tornado.ioloop.IOLoop.current().start()
