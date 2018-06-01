# ML Engineering ASAPP Challenge
### Ryan Enderby's Submission



---
#### The server
`python3 main.py` starts the server with an endpoint for autocompleting strings at `localhost:13000/autocomplete`.

Just like the prompt proposes, you can retrieve your completions by:

        $: curl http://localhost:13000/autocomplete?q=What+is+y

        =>  {"Completions": ["What is your account number?", "What is your address?", "What is your order number?"]}

Requests are handled by the `autocomplete_handler`, which uses a prefix trie to return completions from strings seen during training, or an RNN(GRU) model to complete novel strings.

#### Models
All models are already created and saved, and will be loaded upon initialization of the server.

However, should you choose to recreate the models instead of loading the serialized objects, the prefix trie is constructed if necessary upon starting the server, and loaded from a pickled object if the trie has previously been constructed.

The RNN(GRU) can be trained (starting from a previous checkpoint) via the command `python3 rnn.py train --checkpoint=checkpoints/model.ckpt --restore=checkpoints/model.ckpt  --text=../data/sentences.txt`.  If there are no prior checkpoints, omit the `--restore` flag.

Training the model over 64 epochs took about ~1 hour on CPU, and a fraction of that time on GPU.  Most of the time it generates coherent English remarks.  However, it will occasionally omit sentences with nonsense words, which can largely be attributed to (1) the size of the dataset and (2) the priority I gave to keeping the model lightweight.


#### Credits and dependencies
Non-standard library packages used: `tornado`, `pickle`, `keras`, `numpy`, `nltk`, `h5py`

`main.py`, `trie.py` and `test.py` entirely written by me.  `rnn.py` and `utils.py` is my adaptation of [YuXuan Tay's Keras implementation](https://github.com/yxtay/char-rnn-text-generation) of [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn).  No need to reinvent the wheel :) . I changed a number of the lines of code, text-generation logic, model hyper-paremeters, etc. Happy to explain my decisions in a phone call. The `logging.py` file, however, comes directly from YuXuan's repo.

#### Possible extensions

**Post-deployment learning:** Training shouldn't just be a one-time activity.  Instances where a user does not choose any autocomplete suggestions can be recorded and used for further training.  A separate server could accumulate new complete messages and run batch training on the RNN model, updating the weights of the deployed models periodically.  Similarly, messages can be added to the trie.  If trie size and memory become issues, least-recently-used completions or prefixes could be ejected after an amount of time.

**Privacy concerns:** Training data should be scrubbed of any personal information.  For example, phone numbers could be replaced by a tag indicating that it's a phone number ("< phone_number >"), and an occurrence of the tag in autocompleted text could signal the application to replace the tag with the phone number of the user to which the response is being sent.


## Follow-up questions
---

- How would you evaluate your autocomplete server? If you made another version, how would you compare the two to decide which is better?

In this case, there are a couple of dimensions across which to compare autocomplete servers.

1. **Model performance:** If the difference between the two versions involves changes to the RNN model, it would be essential to compare how well each version's model fits the data.  This can be done through the comparison of training and validation losses.  A good strategy would be cross validation (e.g. k-folds), such that you average losses over the various folds to identify which model performs better.  Validation loss is an especially useful metric as it helps identify which model may perform better on previously unseen data.  This is important to note because user input will vary greatly and there are no guarantees it will closely match training conversations.

2. **Speed:** A better fitting model could end up being less useful if it is perceptibly slower to users.  As such, a comparison should be made regarding the prediction times of each model on input sequences of the same length.

3. **User experience:** At the end of the day, this tool is successful if it helps the user.  A simple way to evaluate this dimension is through deploying both versions in an AB Test.  Users are randomly assigned either version A or version B, and some useful metric (or direct user feedback) is tracked and evaluated.  For example, did one version lower the average total time users spent chatting with a customer service agent, indicating that the version helped them accomplish their goal quicker?
    
---
- One way to improve the autocomplete server is to give topic-specific suggestions. How would you design an auto-categorization server? It should take a list of messages and return a TopicId. (Assume that every conversation in the training set has a TopicId).

There are a few ways you could go about this problem. One way is to treat it as a topic modeling problem, another is to treat it as a supervised-learning, classification problem.  Since you have a training set of labeled data, I would opt for the classification approach.  The server would receive a list of sentences, create feature vectors for each sentence (e.g. using a pretrained embedding model like Word2Vec and engineering other useful features - although, if speed is again important, you may need to forego features that require too much preprocessing), and run a classification model on the feature vector representations of the sentences. Since you are returning a TopicId for the *list*, each message in the list can be processed independently (e.g. fed through a model with softmax scores as the output), the outputs can be pooled / added, with the highest scoring TopicId assigned to the list.  This has clear benefits in terms of speed over a server that processes each sentence in the list sequentially.

---
- How would you evaluate if your auto-categorization server is good?

Similar to my response for the first questions, it is important to consider dimensions alongside model performance, such as speed and value to the user.  Further, since this is a user-facing, it is also important to consider *what kind of errors* the model makes.  As such, to assess the performance of the model that the auto-categorization server uses, you can look at the confusion matrix of training and test output.

---
- Processing hundreds of millions of conversations for your autocomplete and auto-categorize models could take a very long time. How could you distribute the processing across multiple machines?

Fortunately, since requests from different users are independent, the model servers are independent and entirely horizontally scalable, and instances can be spun up according to request load.  Both autocomplete and auto-categorize can be opened as API end points, with each request being pushed to a queueing mechanism.  The next available server will pull from the queue to process a new request every time it finishes processing an earlier request.
