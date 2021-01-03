import numpy as np
import os
import re
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, BatchNormalization, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# Root directory of project files
# DATA_PATH = Path(__file__).resolve().parent / "data" / "review_polarity"
DATA_PATH = Path(__file__).resolve().parent / "data"
LABEL_REGEX = "__label__[1|2] "

# *** CITATIONS ***
# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
# https://www.tensorflow.org/tutorials/text/text_classification_rnn

def retrieve_data():
    def read_file(file, encoding="utf-8"):
        with open(file, encoding=encoding) as f:
            dataset = f.read().split("\n")
            dataset.pop()
        return dataset

    def get_data(dataset):
        X = []
        y = []

        for review in dataset:
            label = re.search(LABEL_REGEX, review).group(0)
            X.append(review.replace(label, ""))
            sentiment = 0 if label == "__label__1 " else 1
            y.append(sentiment)

        return X, y

    train_dataset = read_file(DATA_PATH / "train.ft.txt")[:20000]
    test_dataset = read_file(DATA_PATH / "test.ft.txt")[:2000]

    X_train, y_train = get_data(train_dataset)
    X_test, y_test = get_data(test_dataset)



    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    # dataset = []
    # labels = []
    #
    # for root, dirs, files in os.walk(DATA_PATH, topdown=False):
    #     for dir in dirs:
    #         label = 0 if dir == "neg" else 1
    #         for file in os.listdir(DATA_PATH / dir):
    #             with open(DATA_PATH / dir / file) as f:
    #                 dataset.append(f.read())
    #                 labels.append(label)
    #
    # X = np.array(dataset)
    # y = np.array(labels)
    #
    # return {"X": X, "y": y}


def compute_results(data):
    #X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.2, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((data["X_train"], data["y_train"]))
    test_dataset = tf.data.Dataset.from_tensor_slices((data["X_test"], data["y_test"]))

    BUFFER_SIZE = 1000
    BATCH_SIZE = 100

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    VOCAB_SIZE = 1000
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = Sequential()
    model.add(encoder)
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=8, activity_regularizer=l2(0.001), mask_zero=True))
    model.add(Bidirectional(LSTM(8)))
    model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.001), activity_regularizer=l1(0.001)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=Adam(), metrics=['accuracy'])

    es_callback = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(train_dataset, validation_data=test_dataset, batch_size=BATCH_SIZE, epochs=10, callbacks=[es_callback])

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == "__main__":
    data = retrieve_data()
    results = compute_results(data)