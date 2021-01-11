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
ROOT = Path(__file__).resolve().parent

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

def plot_model_results(results, metric):
    plt.plot(results.history[metric])
    plt.plot(results.history[f"val_{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(ROOT / f"figs/model_{metric}")
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
    with open(ROOT / "data/train.ft.txt", encoding="utf-8") as tr, open(ROOT / "data/test.ft.txt",
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

    # model.save(ROOT / "outputs/my_model")

    return history


def web_scrape_text():
    op_eds = []

    link_1 = "https://www.cbc.ca/news/canada/manitoba/teresa-moysey-united-church-opinion-manitoba-1.5856794"
    op_ed_1 = get_article_body(link_1, "div.story > span > p")
    op_eds.append(op_ed_1)

    link_2 = "https://www.cbc.ca/news/canada/manitoba/opinion-madhav-sinha-vaccination-plan-quality-1.5845930"
    op_ed_2 = get_article_body(link_2, "div.story > span > p")
    op_eds.append(op_ed_2)

    link_3 = "https://www.thestar.com/opinion/editorials/2021/01/07/curfew-would-be-dramatic-but-an-admission-of-failure-in-fight-against-covid-19.html"
    op_ed_3 = get_article_body(link_3, "div.c-article-body__content > p.text-block-container")
    op_eds.append(op_ed_3)

    link_4 = "https://www.cnn.com/2021/01/05/opinions/uk-delay-second-covid-vaccine-dose-moore/index.html"
    op_ed_4 = get_article_body(link_4, "div.zn-body__paragraph")
    op_eds.append(op_ed_4)

    np.save(ROOT / "data/op_eds.npy", op_eds)

    papers = []

    link_5 = "https://www.nejm.org/doi/10.1056/NEJMoa2035389"
    paper_1 = get_article_body(link_5, "p.f-body, p.f-body--sm")
    papers.append(paper_1)

    link_6 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2034545"
    paper_2 = get_article_body(link_6, "p.f-body, p.f-body--sm")
    papers.append(paper_2)

    link_7 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2016638"
    paper_3 = get_article_body(link_7, "p.f-body, p.f-body--sm")
    papers.append(paper_3)

    link_8 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2019375"
    paper_4 = get_article_body(link_8, "p.f-body, p.f-body--sm")
    papers.append(paper_4)

    np.save(ROOT / "data/papers.npy", papers)

    short_stories = []

    link_9 = "https://www.freechildrenstories.com/the-great-hill"
    story_1 = get_article_body(link_9, "div.sqs-block-content > p")
    short_stories.append(story_1)

    link_10 = "https://www.freechildrenstories.com/the-particular-way-of-the-odd-ms-mckay"
    story_2 = get_article_body(link_10, "div.sqs-block-content > p")
    short_stories.append(story_2)

    link_11 = "https://www.freechildrenstories.com/the-stellar-one-1"
    story_3 = get_article_body(link_11, "div.sqs-block-content > p")
    short_stories.append(story_3)

    link_12 = "https://www.freechildrenstories.com/king-michael"
    story_4 = get_article_body(link_12, "div.sqs-block-content > p")
    short_stories.append(story_4)

    np.save(ROOT / "data/short_stories.npy", short_stories)


def get_article_body(link, css_selector):
    session = HTMLSession()
    res = session.get(link)
    article = res.html.find(css_selector)
    article_body = [paragraph.text for paragraph in article]

    return article_body


def plot_article_sentiment(article, title):
  model = load_model(ROOT + "my_model")
  predictions = model.predict(article)
  plt.plot(predictions)
  plt.axhline(y=np.mean(predictions), color='r', linestyle='--')
  plt.title(f"Sentiment History: {title}")
  plt.ylabel("sentiment")
  plt.xlabel("paragraph")
  plt.legend(['Sentiment History', 'Mean'], loc='upper left')
  plt.show()


if __name__ == "__main__":
    train_dataset, test_dataset = retrieve_data()
    model = build_model(train_dataset)
    results = compute_results(train_dataset, test_dataset, model)
    plot_model_results(results, "accuracy")
    plot_model_results(results, "loss")

    web_scrape_text()

    for index, op_ed in enumerate(np.load(ROOT / "data/op_eds.npy", allow_pickle=True)):
      plot_article_sentiment(op_ed, f"Sample Opinion Editorial #{index + 1}")

    for index, paper in enumerate(np.load(ROOT / "data/papers.npy", allow_pickle=True)):
      plot_article_sentiment(paper, f"Sample Paper #{index + 1}")

    for index, story in enumerate(np.load(ROOT / "data/short_stories.npy", allow_pickle=True)):
      plot_article_sentiment(story, f"Sample Short Story #1{index + 1}")



