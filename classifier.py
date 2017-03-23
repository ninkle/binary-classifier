import os
import time
import glob
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


'''Employs sklearn's Multinomial and Bernoulli Naive Bayes classifiers to conduct binary sentiment analysis on
a dataset of 1000 positive and 1000 negative Rotten Tomatoes movie reviews
(found at http://www.cs.cornell.edu/People/pabo/movie-review-data/). Each trial trains and tests on a randomly selected
80/20 split. Reviews are passed through nltk's SnowballStemmer to account for grammatical discrepancies, then converted
into a bag-of-ngrams model (range (1, 6)). Passed through sklearn's TidfVectorizer to normalize frequencies and
account for stopwords. Of the two classifiers, Multinomal Naive Bayes was found to consistently produce higher
precision, recall, and f-1 scores.'''

def get_stems(text):
    # initializes snowball stemmer from nltk
    snowball = SnowballStemmer("english")

    # initializes count vectorizer from sklearn, creating bag of ngrams (ngram_range could use further testing)
    ngram_vectorizer = CountVectorizer('char_wb', ngram_range=(1, 6))

    # delineates words by empty character " " and converts to list format
    tokenized = (ngram_vectorizer.build_tokenizer()(text))

    # applies snowball stemmer so we can account for grammatical discrepancies
    for i in range(len(tokenized)):
        tokenized[i] = snowball.stem(tokenized[i])

    # convert list back to string and return
    text = (" ".join(tokenized))
    return text


def extract_text(label):
    reviews = []
    for filename in glob.iglob(os.path.join('txt_sentoken', label, '*.txt')):
        f = open(filename, "r")
        text = f.read()
        reviews.append(text)
    return reviews


if __name__ == '__main__':
    # extract negative reviews and shuffle to randomize input
    neg_data = extract_text('neg')
    random.shuffle(neg_data)
    # extract positive reviews
    pos_data = extract_text('pos')
    random.shuffle(pos_data)

    # create lists for negative and positive labels
    neg_labels = ['neg'] * len(neg_data)
    pos_labels = ['pos'] * len(pos_data)

    # split into training and testing sets (80/20)
    train_data = neg_data[:800] + pos_data[:800]
    train_labels = neg_labels[:800] + pos_labels[:800]

    test_data = neg_data[801:] + pos_data[801:]
    test_labels = neg_labels[801:] + pos_labels[801:]

    # convert words in each review to their stems
    for i in range(len(train_data)):
        train_data[i] = get_stems(train_data[i])

    for i in range(len(test_data)):
        test_data[i] = get_stems(test_data[i])

    # initialize TfidfVectorizer from sklearn to vectorize text and normalize word frequencies
    # played around with min_df and max_df a bit to determine optimal values
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 decode_error='ignore')

    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # classification with Multinomial Naive Bayes
    mnb = MultinomialNB()
    t0 = time.time()
    mnb.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_mnb = mnb.predict(test_vectors)
    t2 = time.time()
    time_mnb_train = t1 - t0
    time_mnb_predict = t2 - t1

    # classification with Bernoulli Naive Bayes
    bnb = BernoulliNB()
    t0 = time.time()
    bnb.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_bnb = bnb.predict(test_vectors)
    t2 = time.time()
    time_bnb_train = t1 - t0
    time_bnb_predict = t2 - t1

    # print results
    print("Results for MultinomialNB()")
    print("Training time: %fs; Prediction time: %fs" % (time_mnb_train, time_mnb_predict))
    print(classification_report(test_labels, prediction_mnb))

    print("Results for BernoulliNB()")
    print("Training time: %fs; Prediction time: %fs" % (time_bnb_train, time_bnb_predict))
    print(classification_report(test_labels, prediction_bnb))


 
