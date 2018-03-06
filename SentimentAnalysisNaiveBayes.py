# Project - Sentiment Analysis
# CSCD 429 Winter 2018

from __future__ import print_function
import random
import csv
import nltk
import nltk
import re
from nltk.corpus import stopwords
wnlemmatizer = nltk.WordNetLemmatizer()

trainData = []
all_words = []

phraseIndex = None
sentimentIndex = None

word_features = None

def main():
    global trainData
    global all_words
    global word_features

    print("Reading in training data...")

    trainReader = dataReader()
    dataPreprocessing(trainReader)

    print("Finished reading in data...")

    random.shuffle(trainData)

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:int(len(all_words) * 0.50)]

    print("Generating feature sets...")

    # convert all reviews to feature sets
    feature_sets = [(find_features(rev), rating) for (rev, rating) in trainData]

    # split data for testing, use 75% for training, 25% for testing
    train_set = feature_sets[:int(len(feature_sets) * 0.75)]
    test_set = feature_sets[int(len(feature_sets) * 0.75):]

    print("Running naive bayes...")

    # naive bayes
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print((nltk.classify.accuracy(classifier, test_set)) * 100, "%", sep="")

    #classifier.show_most_informative_features(15)

    print("Finished...")

def find_features(doc):
    global word_features

    words = set(doc)
    features = {}
    for word in word_features:
        features[word] = (word in words)  # bool of if the word is in the review

    return features

def dataReader():
    global phraseIndex
    global sentimentIndex

    trainPath = "train.csv"
    trainFile = open(trainPath, "r")  # docs say to use "rb", fails on Python 3.5+
    trainReader = csv.reader(trainFile, delimiter=',')
    headers = next(trainReader)
    phraseIndex = headers.index("Phrase")
    sentimentIndex = headers.index("Sentiment")
    return trainReader

def dataPreprocessing(trainReader):
    global trainData
    global all_words
    global phraseIndex
    global sentimentIndex

    for row in trainReader:
        tokens = nltk.word_tokenize(row[phraseIndex])
        tokens = [tk.lower() for tk in tokens]
        tokens = [re.sub(r'[^a-zA-Z]', " ", x) for x in tokens]
        tokens = [w for w in tokens if not w in set(stopwords.words("english"))]
        tokens = [wnlemmatizer.lemmatize(x) for x in tokens]
        tokens = [wnlemmatizer.lemmatize(x, 'v') for x in tokens]
        tokens[:] = [item for item in tokens if item != ' ']
        for w in tokens:
            all_words.append(w.lower())
        trainData.append((tokens, row[sentimentIndex]))

main()