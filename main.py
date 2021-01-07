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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from requests_html import HTMLSession

# Root directory of project files
DATA_PATH = Path(__file__).resolve().parent

LABEL_REGEX = "__label__[1|2] "
BUFFER_SIZE = 1000
BATCH_SIZE = 100
VOCAB_SIZE = 1000
TRAIN_SIZE = 1500000
TEST_SIZE = 300000
EPOCHS = 11

np.random.seed(3)
tf.random.set_seed(3)


# *** CITATIONS ***
# https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
# https://www.tensorflow.org/tutorials/text/text_classification_rnn

def plot_results(results, metric):
    plt.plot(results.history[metric])
    plt.plot(results.history[f"val_{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(DATA_PATH / f"figs/model_{metric}")
    plt.show()


def separate_data(dataset):
    x = []
    y = []

    for review in dataset:
        label = re.search(LABEL_REGEX, review).group(0)
        x.append(review.replace(label, ""))
        y.append(0 if label == "__label__1 " else 1)

    return x, y


def retrieve_data():
    with open(DATA_PATH / "data/train.ft.txt", encoding="utf-8") as tr, open(DATA_PATH / "data/test.ft.txt",
                                                                        encoding="utf-8") as te:
        train_dataset = tr.read().split("\n")[:TRAIN_SIZE]
        test_dataset = te.read().split("\n")[:TEST_SIZE]

    x_train, y_train = separate_data(train_dataset)
    x_test, y_test = separate_data(test_dataset)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


def build_model(train_dataset):
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = Sequential()
    model.add(encoder)
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=8, activity_regularizer=l2(0.001),
                        mask_zero=True))
    model.add(Bidirectional(LSTM(8)))
    model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.001), activity_regularizer=l2(0.001)))
    model.add(Dense(1, activation="sigmoid"))

    return model


def compute_results(train_dataset, test_dataset, model):
    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=Adam(), metrics=['accuracy'])

    es_callback = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(train_dataset, validation_data=test_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=[es_callback])

    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    #model.save(DATA_PATH + "my_model")

    return history


def get_article_body(link, css_selector):
    article_body = []
    session = HTMLSession()
    res = session.get(link)
    article = res.html.find(css_selector)
    for paragraph in article:
        article_body.append(paragraph.text)

    return article_body


if __name__ == "__main__":
    model = load_model(DATA_PATH / "my_model")

    # train_dataset, test_dataset = retrieve_data()
    # model = build_model(train_dataset)
    # results = compute_results(train_dataset, test_dataset, model)
    # plot_results(results, "accuracy")
    # plot_results(results, "loss")

    op_ed_links = ["https://www.nytimes.com/2021/01/06/opinion/protests-trump-disinformation.html",
                   "https://www.nytimes.com/2021/01/06/opinion/georgia-senate-election.html",
                   "https://www.nytimes.com/2021/01/05/opinion/trump-republicans-election.html"]

    op_eds = []

    for link in op_ed_links:
        op_ed = get_article_body(link, "p.css-axufdj.evys1bk0")
        op_eds.append(op_ed)

    np.save(DATA_PATH + "op_eds", op_eds)



