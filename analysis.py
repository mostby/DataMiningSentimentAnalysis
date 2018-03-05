# Project - Sentiment Analysis
# CSCD 429 Winter 2018

# Usage: python analysis.py trainPath.tsv testPath.tsv

# data columns: PhraseId, SentenceId, Phrase, Sentiment

import sys
import random
import csv
import nltk

# get the file paths
if len(sys.argv) == 1:
    print("No training file or testing file specified")
    sys.exit(-1)
elif len(sys.argv) == 2:
    print("No testing file specified")
    sys.exit(-1)

trainPath = sys.argv[1]
testPath = sys.argv[2]


print("Reading in training data...")


trainFile = open(trainPath, "r")  # docs say to use "rb", fails on Python 3.5+
trainReader = csv.reader(trainFile, delimiter='\t')

# save the headers (the first row)
headers = next(trainReader)
phraseIndex = headers.index("Phrase")
sentimentIndex = headers.index("Sentiment")


# the csv reader is only for streaming in, need a place to actually store data
trainData = []
all_words = []

# read in the data, set it up for NLTK
for row in trainReader:
    tokens = nltk.word_tokenize(row[phraseIndex])
    tokens = [tk.lower() for tk in tokens]
    for w in tokens:
        all_words.append(w.lower())
    trainData.append((tokens, row[sentimentIndex]))


print("Finished reading in data...")


random.shuffle(trainData)

all_words = nltk.FreqDist(all_words)

# strip off the least common 50%
# keeping too much can cause problems with memory
word_features = list(all_words.keys())[:int(len(all_words) * 0.50)]


def find_features(doc):
    words = set(doc)
    features = {}
    for word in word_features:
        features[word] = (word in words)  # bool of if the word is in the review

    return features


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

# classifier.show_most_informative_features(15)
