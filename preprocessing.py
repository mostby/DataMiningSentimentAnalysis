import nltk
import random
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
pstemmer = nltk.PorterStemmer()
wnlemmatizer = nltk.WordNetLemmatizer()
import random
import csv
import nltk

def main():
    print("Start")
    traindf = getCSVDataFrame("train.csv")

    more_final_words = preprocessing(traindf)


    printcount = 1
    for word in more_final_words:
        #print(more_final_words[printcount])
        for x in more_final_words[printcount]:
            print(x)
        printcount += 1

    print("End")

def getCSVDataFrame(csvname):
    df = pd.read_csv(csvname)
    return df


def preprocessing(data):
    alphhabet_only = {}
    words_lower = {}
    useful_words = {}
    final_words = {}
    more_final_words = {}
    counter = 1

    for phrases in data['Phrase']:
        alphhabet_only[counter] = re.sub(r'[^a-zA-Z]', " ", phrases)
        words_lower[counter] = alphhabet_only[counter].lower().split()
        useful_words[counter] = [w for w in words_lower[counter] if not w in set(stopwords.words("english"))]
        final_words[counter] = [wnlemmatizer.lemmatize(x) for x in useful_words[counter]]
        more_final_words[counter] = [wnlemmatizer.lemmatize(x, 'v') for x in useful_words[counter]]
        counter += 1

    return more_final_words


main()