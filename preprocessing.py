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

def main():
    print("Start")
    traindf = getCSVDataFrame("train.csv")

    alphhabet_only = {}
    words_lower = {}
    useful_words = {}
    counter = 1

    for phrases in traindf['Phrase']:
        alphhabet_only[counter] = re.sub(r'[^a-zA-Z]', " ", phrases)
        words_lower[counter] = alphhabet_only[counter].lower().split()
        useful_words[counter] = [w for w in words_lower[counter] if not w in set(stopwords.words("english"))]
        counter += 1

    final_words = {}
    counter2 = 1
    for phrases in useful_words:
        final_words[counter2] = [pstemmer.stem(x) for x in phrases]
        counter2 += 1

    print(final_words)

    processed_train = traindf['Phrase'].apply(word_tokenize)

    print("End")

def getCSVDataFrame(csvname):
    df = pd.read_csv(csvname)
    return df

main()