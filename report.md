# Q1. Summary
a) Obtained a dataset compatible with Natural Language Processing techniques 

b) Prepared the data for training a Keras Sequential model

c) Built a Sequential model containing text vectorization, word embedding, and LSTM layers for numerical processing of 
text data

d) Implemented regularization techniques to minimize over-fitting in the RNN

e) Plotted metrics such as training and testing loss and accuracy as functions of the number of epochs

f) Utilized the requests-html Python library to parse HTML from various web articles for the purposes of
sentiment analysis

g) Plotted a paragraph-by-paragraph sentiment history of each article considered, and did cross-comparisons between each
article category

# Q2. Dataset Description

## Dataset Description
This dataset was retrieved from Kaggle, an online data science community. The dataset can be downloaded here:
https://www.kaggle.com/bittlingmayer/amazonreviews

Contained in the dataset is a total of 4,000,000 Amazon reviews; 3,600,000 reviews in the train dataset and 400,000 
reviews in the test dataset. Each review is prepended by a label indicating the polarity. A label of __label__1 is
for 1 and 2-star reviews whereas a label of __label__2 is for 4 and 5-star reviews. Neutral reviews (i.e. 3-star 
reviews) were excluded from this dataset. As such, we are interested in training a model to perform binary
classification, where the output will be closer to 0 (i.e. negative sentiment) or a 1 (positive sentiment). The polarity
is evenly divided among the training and testing datasets; that is, there are 1,800,000 positive and negative reviews
in the training dataset, and 200,000 positive and negative reviews in the testing dataset, which adds up to the total of
4,000,000.

## AUC Values
|        Feature        |  AUC  |
|:----------------------|:-----:|
| Feature 1             | 0.790 |
| Feature 2             | 0.250 |
| ....                  | ...   |
| Feature 10            | 0.023 |

# Q3. Details
a) The dataset, retrieved from Kaggle, is a slightly modified version of the dataset from Xiang Zhang's 
Google Drive<sup>1</sup>, which is available in CSV format. Kaggle's version of this dataset is compatible with
the fastText library for text classification, which makes it easy to train a model within a few minutes. However, I 
opted to use the Keras API to construct the model architecture and perform training.

In this task, I simply extracted the two datasets into the data folder of the project's root directory and then read
their contents into Python lists for training and testing. I constrained the training dataset to 2,000,000 samples due to
memory limitations and performance considerations.

b) This step involved performing some minor preprocessing of the data and converting it into a format that is simple to
use on a Sequential model. After reading the data into training and testing lists, my first step was to isolate the 
input data into list x and class labels into list y so that they were separate:

```
x_train, y_train = separate_data(train_dataset)
x_test, y_test = separate_data(test_dataset)
```

I extracted a simple utility function for this purpose:

```
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
```

My next step was to reinitialize the training and testing datasets as instances of tf.Data.Dataset. The Tensorflow
dataset was created from slices of the input data (reviews) and labels (polarities):

```
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
```

The Tensorflow dataset API offers the user flexibility in applying dataset transformations for
preprocessing<sup>2</sup>. Because my dataset is relatively large, (~1.7GB) to ensure perfect shuffling, I decided to 
shuffle the dataset elements with a buffer size equal to the training size<sup>3</sup>:

`train_dataset = train_dataset.shuffle(TRAIN_SIZE)`

Then the shuffled elements are combined into batches as follows:

`train_dataset = train_dataset.batch(BATCH_SIZE)`

Lastly, we want to prefetch elements. At training step s, the input pipeline reads data for step s+1<sup>4</sup>.
Although this requires additional memory usage, it can increase throughput.

`train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)`

c) A machine learning model cannot be trained on plain text; it must operate on numerical input. Therefore, we need to
build our model such that each batch of strings are encoded as integers. The preprocessed text must be converted into
numeric feature vectors so that it can be fed into our neural network. To achieve this, we first add a TextVectorization
layer to our Sequential model:

`encoder = TextVectorization(max_tokens=VOCAB_SIZE)`

This layer has many capabilities, but here we only pass in `max_tokens` to limit the size of vocabulary. From here we
call the `adapt()` method on the encoder, which analyzes the dataset and returns a list of the `max_tokens` most
frequently used words:

`encoder.adapt(train_dataset.map(lambda text, label: text))`

We then initialize the Sequential object and add the first TextVectorization layer:

```
model = Sequential()
model.add(encoder)
```

This layer will essentially receive strings as input and encode each token (word) in the vocabulary by assigning
integers to them. These integers will be indices to a lookup table containing dense vectors for each word in the
vocabulary. This lookup table is initialized by the Embedding layer:

```
model.add(Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=8, activity_regularizer=l2(0.001),
                        mask_zero=True))
```
                        
The Embedding layer will receive the token indices and initialize the embedding vectors (i.e. the weights of the
embedding layer)<sup>5</sup>. The weights are then updated via Stochastic Gradient Descent in order to minimize loss.

An LSTM layer with a Bidirectional wrapper is then added to the model. An LSTM layer on its own will send outputs from
one timestep into their inputs on the subsequent timestep<sup>6</sup>. With the Bidirectional wrapper, however, the RNN
layers are duplicated such that the first layer receives input as-is, and the second layer receives a reversed copy as
input <sup>7</sup>. 

`model.add(Bidirectional(LSTM(8)))`

Finally, two densely connected layers are added for final processing and a classification (between 0 and 1) appears as
the final output:

```
model.add(Dense(8, activation="relu", kernel_regularizer=l2(0.001), activity_regularizer=l2(0.001)))
model.add(Dense(1, activation="sigmoid"))
```

d) Upon initially training the model, I noticed in the plots that the training loss continued decreasing for every
epoch, while the validation loss began steadily increasing. To minimize this over-fitting, I added l2 regularization
terms of 0.001 to the Embedding and Dense layers. I noticed that adding the regularization decreased the discrepancy
between training and validation loss, although the validation loss did continue to slightly increase. I made use of 
Keras' EarlyStopping callback function, which terminates training after a specified metric has stopped improving
(namely, loss)<sup>8</sup>. This ensures that the model's accuracy does not get worse with additional unnecessary
training. After adjusting the regularization penalties, decreasing the size of LSTM and Dense layers, and implementing
early stopping, I was able to achieve a training and validation accuracy of approximately 92.2%.

e) I used the matplotlib library to visually represent the model's accuracy and loss metrics against the number of
epochs. From the resultant plots, we see an abrupt change in the accuracy and loss during the first epoch, and then they
begin to gradually taper off. The validation loss began to steadily increase after the 16th epoch and into the 19th 
epoch, upon which the early stopping callback was activated, which rolled the model state back to the previous lowest
loss value. Both of these plots, model_accuracy.png and model_loss.png, can be found in the figs folder.

f) After training the model and plotting its metrics, my next step was to generate some text data via web scraping. I 
installed the requests-html library which has a number of features, including full JavaScript and async support.
For the purposes of sentiment analysis, all I needed was to extract the main body of text from different categories of
articles (opinion editorials, scientific papers, and children's short stories) and use the model to predict sentiments
for each paragraph of text in those articles. To accomplish this, I picked a predetermined number of articles for each 
category and saved the main bodies of text into three separate lists (one for each category). The code to achieve this
is as follows:

```
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

    # and so on
```

The `web_scrape_text()` function initializes the three lists, `op_eds`, `papers`, and `short_stories`. A GET request is 
made to each URL via the `get_article_body()` function, which takes in the URL and CSS selector containing the desired 
text. It then returns a list of all paragraphs in the body of the article. Although `web_scrape_text()` has some obvious
redundancy, I felt as though it was the easiest way to obtain small samples of text for sentiment analysis.

g) After scraping the text data and saving it, I looped through each article and plotted the sentiment history against
the number of paragraphs. For the opinion editorials, the general trend was a higher variance compared to that of 
the scientific papers and short stories. I selected the first four opinion editorials depending on whether the author 
was projecting an overall positive sentiment on the topic at hand, and the last four editorials were chosen if the
author's sentiment appeared to be negative. As expected, the model predicted a mean positive (>0.5) sentiment on the 
first four editorials and a mean negative (<0.5) sentiment on the last four. The standard deviations across the eight
opinion editorial samples were above 0.20, suggesting a relatively high variance. For future experiments, this could be
accounted for by ensuring that an equal number of paragraphs are considered per article; the op-eds generally have a
significantly fewer number of paragraphs which can make the effects of statistical outliers more apparent.

The scientific papers were all retrieved from the New England Journal of Medicine. The topic of each article was related 
to COVID-19 vaccination and case studies. From the plots, each of the scientific papers carried a mean sentiment of 
0.51-0.61, suggesting an overall neutral-positive sentiment. This was to be expected, considering the comparatively
large number of paragraphs in each paper, and also for intuitive reasons: scientific literature is generally devoid of
opinions and language that would carry extreme polarity. The focus of a scientific paper is to outline an introduction,
discuss the methods used in the experiment, results, and discussion of results.

The last category of texts examined for sentiment analysis was children's short stories, all retrieved from 
https://freechildrenstories.com. Each story was written by the same author, Daniel Errico. Each of the four stories 
began with a positive sentiment, and most of them ended on a positive sentiment as well. Much like the other categories
of text, however, the sentiment also fluctuated a considerable amount. Other than the strong sentiments at the beginning
and end of the stories, there do not seem to be any discernible differences between the sentiment curves of the short 
stories and scientific papers. 

For future experimentation, a larger sample size for each category of text should be considered. This could help us spot
more patterns in the data that cannot as easily be spotted with a handful of samples. Each article should
also be similar in length to better illustrate the sentiment history from start to finish and do cross-comparisons. 
Larger articles should also be analyzed to reduce variance and converge the data to the true mean.

## References

[1] https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

[2] https://www.tensorflow.org/api_docs/python/tf/data/Dataset

[3] https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625

[4] https://www.tensorflow.org/guide/data_performance#prefetching

[5] https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work

[6] https://www.tensorflow.org/tutorials/text/text_classification_rnn#create_the_model

[7] https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

[8] https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping