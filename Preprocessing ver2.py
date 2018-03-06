import nltk
import re
from nltk.corpus import stopwords
pstemmer = nltk.PorterStemmer()
wnlemmatizer = nltk.WordNetLemmatizer()
import csv


def main():
    print("Start")
    #traindf = getCSVDataFrame("train.csv")
    #more_final_words = preprocessing(traindf)

    trainPath = "train.csv"
    trainFile = open(trainPath, "r")  # docs say to use "rb", fails on Python 3.5+
    trainReader = csv.reader(trainFile, delimiter=',')
    headers = next(trainReader)
    phraseIndex = headers.index("Phrase")
    sentimentIndex = headers.index("Sentiment")

    trainData = []
    all_words = []

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

    for x in trainData:
        print x[0], x[1]

    print("End")

main()