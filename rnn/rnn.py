import os
import time

import numpy as np

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed, GRU
from keras.models import load_model, Sequential
from keras.optimizers import Adam

from utils import (batch_generator, encode_text, generate_seed, ID2CHAR, main,
                   sample_from_probs, VOCAB_SIZE)


# Adapted from implementation: https://github.com/yxtay/char-rnn-text-generation


def build_model(batch_size, seq_len, vocab_size=VOCAB_SIZE, embedding_size=32,
                rnn_size=128, num_layers=2, drop_rate=0.0,
                learning_rate=0.001, clip_norm=5.0):
    """
    build character embeddings LSTM text generation model.
    """
    model = Sequential()
    # Input shape: (batch_size, seq_len)
    model.add(Embedding(vocab_size, embedding_size,
                        batch_input_shape=(batch_size, seq_len)))
    model.add(Dropout(drop_rate))
    # Shape: (batch_size, seq_len, embedding_size)
    for _ in range(num_layers):
        model.add(GRU(rnn_size, return_sequences=True, stateful=True))
        model.add(Dropout(drop_rate))
    # Shape: (batch_size, seq_len, rnn_size)
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    # Output shape: (batch_size, seq_len, vocab_size)
    optimizer = Adam(learning_rate, clipnorm=clip_norm)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def build_inference_model(model, batch_size=1, seq_len=1):
    """
    build inference model from model config
    input shape modified to (1, 1)
    """
    config = model.get_config()
    # Edit batch_size and seq_len
    config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
    inference_model = Sequential.from_config(config)
    inference_model.trainable = False
    return inference_model


def generate_text(model, seed, length=512, top_n=10):
    """
    generates text of specified length from trained model
    with given seed character sequence.
    """
    generated = seed
    encoded = encode_text(seed)
    model.reset_states()

    for idx in encoded[:-1]:
        x = np.array([[idx]])
        # Input shape: (1, 1)
        # Set internal states
        model.predict(x)

    next_index = encoded[-1]
    for i in range(length):
        x = np.array([[next_index]])
        # Input shape: (1, 1)
        probs = model.predict(x)
        # Output shape: (1, 1, vocab_size)
        next_index = sample_from_probs(probs.squeeze(), top_n)
        # Append to sequence
        generated += ID2CHAR[next_index]
        if ID2CHAR[next_index] in [".", "!", "?", "\n"]:
            return generated

    return generated


def train_main(args):
    """
    trains model specfied in args.
    main method for train subcommand.
    """
    # Load text
    with open(args.text_path) as f:
        text = f.read()

    # Load or build model
    if args.restore:
        load_path = args.checkpoint_path if args.restore is True else args.restore
        model = load_model(load_path)
    else:
        model = build_model(batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            vocab_size=VOCAB_SIZE,
                            embedding_size=args.embedding_size,
                            rnn_size=args.rnn_size,
                            num_layers=args.num_layers,
                            drop_rate=args.drop_rate,
                            learning_rate=args.learning_rate,
                            clip_norm=args.clip_norm)

    # Make and clear checkpoint directory
    model.save(args.checkpoint_path)
    # Callbacks
    callbacks = [
        ModelCheckpoint(args.checkpoint_path, verbose=1, save_best_only=False)
    ]

    # Training start
    num_batches = (len(text) - 1) // (args.batch_size * args.seq_len)
    model.reset_states()
    model.fit_generator(batch_generator(encode_text(text), args.batch_size, args.seq_len, one_hot_labels=True),
                        num_batches, args.num_epochs, callbacks=callbacks)
    model.save_weights("model_weights.h5")
    return model


def generate_main(args):
    """
    generates text from trained model specified in args.
    main method for generate subcommand.
    """
    # Load learning model for config and weights
    model = load_model(args.checkpoint_path)
    # Build inference model and transfer weights
    inference_model = build_inference_model(model)
    inference_model.set_weights(model.get_weights())

    # Create seed if not specified
    if args.seed is None:
        with open(args.text_path) as f:
            text = f.read()
        seed = generate_seed(text)
    else:
        seed = args.seed

    return generate_text(inference_model, seed, args.length, args.top_n)


if __name__ == "__main__":
    main("Keras", train_main, generate_main)

    # python3 rnn.py train --checkpoint=checkpoints/model.ckpt --text=data/sentences.txt
    # python3 rnn.py generate --checkpoint=checkpoints/model.ckpt --seed="What is you"