import numpy as np
import re
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from requests_html import HTMLSession
from typing import List, Tuple
import pandas as pd

# Root directory of project files
ROOT = Path(__file__).resolve().parent

# Each review in the dataset begins with __label__1 or __label__2, respectively indicating a negative or positive review
# We match each review against this regular expression to determine the review polarity
LABEL_REGEX = "__label__[1|2] "

BATCH_SIZE = 128
VOCAB_SIZE = 1000

# Number of training and testing samples
TRAIN_SIZE = 2000000
TEST_SIZE = 400000

EPOCHS = 25

np.random.seed(3)
tf.random.set_seed(3)


def plot_model_results(results: History, metric: str) -> None:
    """
    Plots the specified model metrics against epochs during training, then saves the resultant plot to the figs folder.
    :param results: A History object containing information about the model's metrics during the training process.
    :param metric:  A string representing which metric to plot. May be passed in as "accuracy" or "loss."
    """

    plt.plot(results.history[metric])
    plt.plot(results.history[f"val_{metric}"])
    plt.title(f"model {metric}")
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(ROOT / f"figs/model_{metric}")
    plt.show()


def separate_data(dataset: List[str]) -> Tuple[List[str], List[int]]:
    """
    Separates the dataset into input data (x) and targets (y).
    :param dataset: A list containing the raw data from test.ft.txt or train.ft.txt.
    :return: The tuple (x, y).
    """

    x = []
    y = []

    for review in dataset:
        # Extract the label from the beginning of the review
        label = re.search(LABEL_REGEX, review).group(0)
        # Add the review without the label to the input data
        x.append(review.replace(label, ""))
        # Add the value 0 to the target data if the review is negative, otherwise add 1
        y.append(0 if label == "__label__1 " else 1)

    return x, y


def retrieve_data() -> Tuple[PrefetchDataset, PrefetchDataset]:
    """
    Reads the contents of test.ft.txt and train.ft.txt into separate lists, and then prepares the data for training.
    :return: The training and testing datasets contained in a tuple.
    """

    with open(ROOT / "data/train.ft.txt", encoding="utf-8") as tr, open(ROOT / "data/test.ft.txt",
                                                                        encoding="utf-8") as te:
        # Read contents of train and test files and add individual reviews to corresponding list
        train_dataset = tr.read().split("\n")[:TRAIN_SIZE]
        test_dataset = te.read().split("\n")[:TEST_SIZE]

    # Extract reviews (x) and labels (y) from training and testing datasets
    x_train, y_train = separate_data(train_dataset)
    x_test, y_test = separate_data(test_dataset)

    # Input pipeline below built with the help of: https://www.tensorflow.org/tutorials/text/text_classification_rnn

    # Create Tensorflow datasets from slices of the reviews and labels
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle the data to ensure randomization of the samples
    # We use a buffer size equal to the size of the training dataset; this ensures perfect shuffling as per this answer:
    # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625
    train_dataset = train_dataset.shuffle(TRAIN_SIZE)

    # Combine elements of shuffled dataset into batches
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # Prefetch elements for next training iteration while current iteration takes place
    # An in-depth explanation of this step can be found at https://www.tensorflow.org/guide/data_performance#prefetching
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Repeat above steps for testing dataset (except shuffling)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


def build_model(train_dataset: PrefetchDataset) -> Sequential:
    """
    Initializes a Sequential model and adds text vectorization, word embedding, LSTM, and densely connected layers.
    :param train_dataset: The dataset to adapt the vocabulary on.
    :return: A Sequential object.
    """

    # Initialize the TextVectorization layer which assigns integers to each token
    encoder = TextVectorization(max_tokens=VOCAB_SIZE)

    # Set the vocabulary for the encoding layer. This will be used to initialize a lookup table of word embeddings.
    # The code for this and subsequent layers adapted from:
    # https://www.tensorflow.org/tutorials/text/text_classification_rnn#create_the_text_encoder
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = Sequential()
    model.add(encoder)
    # Next we add our word embedding layer which converts token indices into dense vectors
    model.add(Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=8, activity_regularizer=l2(0.001),
                        mask_zero=True))
    # Bidirectional wrapper for LSTM allows data to be processed forwards and backwards and then concatenated into
    # one output
    model.add(Bidirectional(LSTM(8)))
    # Densely connected layers with L2 regularization to reduce over-fitting
    model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.001), activity_regularizer=l2(0.001)))
    model.add(Dense(1, activation="sigmoid"))

    return model


def compute_results(train_dataset: PrefetchDataset, test_dataset: PrefetchDataset, model: Sequential) -> History:
    """
    Configures and trains the model, then evaluates its accuracy against the test dataset. The trained model is saved
    to the outputs folder.
    :param train_dataset: A PrefetchDataset that the model will be trained on.
    :param test_dataset: A PrefetchDataset that the model's accuracy will be evaluated against.
    :param model: The Sequential object to be trained.
    :return: A History object containing information about the model's metrics during the training process.
    """

    model.compile(loss=BinaryCrossentropy(from_logits=True), optimizer=Adam(), metrics=["accuracy"])

    # Here we introduce an early stopping callback function that will cease training once the validation loss
    # stops decreasing. This is to minimize over-fitting (i.e. reduce the difference between training loss
    # and validation loss).
    # Idea retrieved from https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/
    es_callback = EarlyStopping(monitor="val_loss", patience=3)

    # Train the model
    history = model.fit(train_dataset, validation_data=test_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        callbacks=[es_callback])

    # Get the loss values and metrics once evaluating the model against the test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    model.save(ROOT / "outputs/my_model")

    return history


def get_article_body(url: str, css_selector: str) -> List[str]:
    """
    Extracts the text data from the article body.
    :param url: The URL to the article.
    :param css_selector: Targets all matching HTML elements containing text in the article body.
    :return: A list of paragraphs from the article body.
    """

    session = HTMLSession()
    # Send a GET request to the specified URL
    res = session.get(url)
    # Retrieve all HTML elements with the matching CSS selector
    article = res.html.find(css_selector)
    # Extract the text content from each HTML element; ignore if text is just an empty string
    article_body = [paragraph.text for paragraph in article if paragraph.text != '']

    return article_body


def web_scrape_text() -> None:
    """
    Scrapes text from different categories of articles and saves the data into NumPy arrays. Each array is saved
    into the data folder.
    """

    # Initialize a two-dimensional list of opinion editorials
    op_eds = []

    link_1 = "https://www.cbc.ca/news/canada/manitoba/teresa-moysey-united-church-opinion-manitoba-1.5856794"
    op_ed_1 = get_article_body(link_1, "div.story > span > p")
    op_eds.append(op_ed_1)

    link_2 = "https://www.cbc.ca/news/canada/manitoba/opinion-madhav-sinha-vaccination-plan-quality-1.5845930"
    op_ed_2 = get_article_body(link_2, "div.story > span > p")
    op_eds.append(op_ed_2)

    link_3 = "https://www.cbc.ca/news/canada/saskatchewan/opinion-martha-neovard-new-year-unresolution-1.5863819"
    op_ed_3 = get_article_body(link_3, "div.story > span > p")
    op_eds.append(op_ed_3)

    link_4 = "https://www.cbc.ca/news/canada/saskatchewan/comedy-craig-silliphant-early-xmas-1.5808227"
    op_ed_4 = get_article_body(link_4, "div.story > span > p")
    op_eds.append(op_ed_4)

    link_5 = "https://vancouversun.com/opinion/editorials/editorial-trudeaus-lack-of-humility-is-bad-politics"
    op_ed_5 = get_article_body(link_5, "section.article-content > p")
    op_eds.append(op_ed_5)

    link_6 = "https://vancouversun.com/opinion/columnists/lilley-mengs-lifestyle-family-visit-in-vancouver-simply-outrageous/wcm/b4f18898-4f48-4b37-93be-c262d18f1e6c"
    op_ed_6 = get_article_body(link_6, "section.article-content > p")
    op_eds.append(op_ed_6)

    link_7 = "https://www.thestar.com/opinion/editorials/2021/01/07/curfew-would-be-dramatic-but-an-admission-of-failure-in-fight-against-covid-19.html"
    op_ed_7 = get_article_body(link_7, "div.c-article-body__content > p.text-block-container")
    op_eds.append(op_ed_7)

    link_8 = "https://www.cnn.com/2021/01/05/opinions/uk-delay-second-covid-vaccine-dose-moore/index.html"
    op_ed_8 = get_article_body(link_8, "div.zn-body__paragraph")
    op_eds.append(op_ed_8)

    np.save(ROOT / "outputs/op_eds", op_eds)

    # Initialize a two-dimensional list of scientific papers
    papers = []

    link_9 = "https://www.nejm.org/doi/10.1056/NEJMoa2035389"
    paper_1 = get_article_body(link_9, "p.f-body, p.f-body--sm")
    papers.append(paper_1)

    link_10 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2034545"
    paper_2 = get_article_body(link_10, "p.f-body, p.f-body--sm")
    papers.append(paper_2)

    link_11 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2016638"
    paper_3 = get_article_body(link_11, "p.f-body, p.f-body--sm")
    papers.append(paper_3)

    link_12 = "https://www.nejm.org/doi/full/10.1056/NEJMoa2019375"
    paper_4 = get_article_body(link_12, "p.f-body, p.f-body--sm")
    papers.append(paper_4)

    np.save(ROOT / "outputs/papers", papers)

    # Initialize a two-dimensional list of children's short stories
    short_stories = []

    link_13 = "https://www.freechildrenstories.com/the-great-hill"
    story_1 = get_article_body(link_13, "div.sqs-block-content > p")
    short_stories.append(story_1)

    link_14 = "https://www.freechildrenstories.com/the-particular-way-of-the-odd-ms-mckay"
    story_2 = get_article_body(link_14, "div.sqs-block-content > p")
    short_stories.append(story_2)

    link_15 = "https://www.freechildrenstories.com/the-stellar-one-1"
    story_3 = get_article_body(link_15, "div.sqs-block-content > p")
    short_stories.append(story_3)

    link_16 = "https://www.freechildrenstories.com/king-michael"
    story_4 = get_article_body(link_16, "div.sqs-block-content > p")
    short_stories.append(story_4)

    np.save(ROOT / "outputs/short_stories", short_stories)


def plot_article_sentiment(article: List[str], title: str, model: Sequential) -> None:
    """
    Plots the sentiment score (between 0 and 1; 0 indicating negative sentiment and 1 indicating positive sentiment)
    against paragraphs for a given article. The plots are saved to the figs folder.
    :param article: A list of paragraphs from the article body.
    :param title: The title of the plot.
    :param model: The trained model. Used to predict the sentiment of the article, paragraph-by-paragraph.
    """

    predictions = model.predict(article)
    plt.plot(predictions)
    plt.axhline(y=np.mean(predictions), color="r", linestyle="--")
    plt.title(f"Sentiment History: {title}")
    plt.ylabel("sentiment")
    plt.xlabel("paragraph")
    plt.legend(["Sentiment History", "Mean"], loc="upper left")
    df = pd.DataFrame(predictions)
    # Append the summary statistics to the plot
    plt.figtext(0.15, 0.15, df[0].describe().to_string())
    plt.savefig(ROOT / "figs" / "_".join(title.split(" ")).lower())
    plt.show()


# Main function
if __name__ == "__main__":
    model = load_model(ROOT / "outputs/my_model")

    # Uncomment these lines to re-train the model and re-plot metrics
    # train_dataset, test_dataset = retrieve_data()
    # model = build_model(train_dataset)
    # results = compute_results(train_dataset, test_dataset, model)
    # plot_model_results(results, "accuracy")
    # plot_model_results(results, "loss")

    # Uncomment the call to web_scrape_text() to re-scrape the text data and re-save it into NumPy arrays
    # web_scrape_text()

    # For each category of text, plot the sentiment history of each article of text
    for idx, op_ed in enumerate(np.load(ROOT / "outputs/op_eds.npy", allow_pickle=True)):
        plot_article_sentiment(op_ed, f"Sample Opinion Editorial #{idx + 1}", model)

    for idx, paper in enumerate(np.load(ROOT / "outputs/papers.npy", allow_pickle=True)):
        plot_article_sentiment(paper, f"Sample Paper #{idx + 1}", model)

    for idx, story in enumerate(np.load(ROOT / "outputs/short_stories.npy", allow_pickle=True)):
        plot_article_sentiment(story, f"Sample Short Story #{idx + 1}", model)