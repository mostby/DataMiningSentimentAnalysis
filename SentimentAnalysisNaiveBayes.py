# Project - Sentiment Analysis
# CSCD 429 Winter 2018

from __future__ import print_function
import random
import csv
import nltk
import re
from nltk.corpus import stopwords

all_words = []

phraseIndex = None
sentimentIndex = None

word_features = None


def main():
    global all_words
    global word_features

    print("Reading in training data...")
    train_reader = data_reader()

    print("Preprocessing data...")
    train_data = data_preprocessing(train_reader)

    print("Finished reading in data...")

    random.shuffle(train_data)

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:int(len(all_words) * 0.50)]

    # convert all reviews to feature sets
    print("Generating feature sets...")
    feature_sets = [(find_features(rev), rate) for (rev, rate) in train_data]

    # split data for testing, use 75% for training, 25% for testing
    train_set = feature_sets[:int(len(feature_sets) * 0.75)]
    test_set = feature_sets[int(len(feature_sets) * 0.75):]

    # run naive bayes
    print("Running naive bayes...")
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print((nltk.classify.accuracy(classifier, test_set)) * 100, "%", sep="")

    classifier.show_most_informative_features(15)
    print("Finished")


def find_features(doc):
    global word_features

    words = set(doc)
    features = {}
    for word in word_features:
        features[word] = (word in words)  # bool of if the word is in the review

    return features


def data_reader():
    global phraseIndex
    global sentimentIndex

    train_path = "train.csv"
    train_file = open(train_path, "r")  # docs use "rb", fails on Python 3.5+
    train_reader = csv.reader(train_file, delimiter=',')
    headers = next(train_reader)
    phraseIndex = headers.index("Phrase")
    sentimentIndex = headers.index("Sentiment")
    return train_reader


def data_preprocessing(train_reader):
    train_data = []
    global all_words
    global phraseIndex
    global sentimentIndex

    word_net = nltk.WordNetLemmatizer()

    for row in train_reader:
        tokens = nltk.word_tokenize(row[phraseIndex])
        tokens = [tk.lower() for tk in tokens]
        tokens = [re.sub(r'[^a-zA-Z]', " ", x) for x in tokens]
        tokens = [w for w in tokens if not w in set(stopwords.words("english"))]
        tokens = [word_net.lemmatize(x) for x in tokens]
        tokens = [word_net.lemmatize(x, 'v') for x in tokens]
        tokens[:] = [item for item in tokens if item != ' ']
        for w in tokens:
            all_words.append(w.lower())
        train_data.append((tokens, row[sentimentIndex]))

    return train_data


main()
