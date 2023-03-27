import nltk
import string
import re
import pandas as pd
import glob
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

data = []
file_names = []

directory = 'E:\CSE508_Winter2023_Dataset'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        file_names.append(filename)
        OpenHtmlFile = open(f, "r")
        contents = OpenHtmlFile.read()
        OpenHtmlFile.close()
        data.append(contents)


def preprocessing(d):
    no_punc = []
    d_lower = d.lower()

    token_in_nltk = nltk.word_tokenize(d_lower)

    stop_words_removed = []
    for w in token_in_nltk:
        if w not in stop_words:
            stop_words_removed.append(w)

    wordings = []
    for x in stop_words_removed:
        if (x.isalnum() and x != " "):
            wordings.append(x)

    return wordings

preprocessed_data = []
for d in data:
    fresh_data = preprocessing(d)
    preprocessed_data.append(fresh_data)

query = input("Enter desired query! ")
preprocessed_query = preprocessing(query)

JC_score = []
for file_data in preprocessed_data:
    intersection = list(set(file_data) & set(preprocessed_query))
    union = list(set().union(file_data, preprocessed_query))
    JC = len(intersection)/len(union)
    JC_score.append(JC)

maximum = max(JC_score)
JC_score.index(maximum)

M = [x for _,x in sorted(zip(JC_score,file_names))]
M.reverse()
i=0
for i in range(10):
    print(M[i])
