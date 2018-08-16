
---
#### The server
`python3 main.py` starts the server with an endpoint for autocompleting strings at `localhost:13000/autocomplete`.

You can retrieve your completions by:

        $: curl http://localhost:13000/autocomplete?q=What+is+y

        =>  {"Completions": ["What is your account number?", "What is your address?", "What is your order number?"]}

Requests are handled by the `autocomplete_handler`, which uses a prefix trie to return completions from strings seen during training, or an RNN(GRU) model to complete novel strings.

#### Models
All models are already created and saved, and will be loaded upon initialization of the server.

However, should you choose to recreate the models instead of loading the serialized objects, the prefix trie is constructed if necessary upon starting the server, and loaded from a pickled object if the trie has previously been constructed.

The RNN(GRU) can be trained (starting from a previous checkpoint) via the command `python3 rnn.py train --checkpoint=checkpoints/model.ckpt --restore=checkpoints/model.ckpt  --text=data/sentences.txt`.  If there are no prior checkpoints, omit the `--restore` flag.

Training the model over 64 epochs took about ~1 hour on CPU, and a fraction of that time on GPU


#### Credits and dependencies
Non-standard library packages used: `tornado`, `pickle`, `keras`, `numpy`, `nltk`, `h5py`

`main.py`, `trie.py` and `test.py` entirely written by me.  `rnn.py` and `utils.py` is my adaptation of [YuXuan Tay's Keras implementation](https://github.com/yxtay/char-rnn-text-generation) of [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn).  No need to reinvent the wheel :). The `logging.py` file, however, comes directly from YuXuan's repo.

