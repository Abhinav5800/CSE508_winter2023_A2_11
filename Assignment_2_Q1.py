import nltk
import string
import re
import pandas as pd
import glob
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from nltk import word_tokenize
from scipy.special import logsumexp
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import math
import statistics
from statistics import mode
stop_words = set(stopwords.words('english'))
data = []
files = []

directory = 'E:\CSE508_Winter2023_Dataset'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        files.append(filename)
        OpenHtmlFile = open(f, "r")
        contents = OpenHtmlFile.read()
        OpenHtmlFile.close()
        data.append(contents)

#print(data)

def preprocessing(d):
    d_lower = d.lower()
    nltk_tokens = nltk.word_tokenize(d_lower)
    stop_words_removed = []
    for w in nltk_tokens:
        if w not in stop_words:
            stop_words_removed.append(w)
    wordings = []
    for x in stop_words_removed:
        if (x.isalnum() and x != " "):
            wordings.append(x)
    return wordings

counting_words=0
preprocessed_data = []
for d in data:
    fresh_data = preprocessing(d)
    counting_words = counting_words+len(fresh_data)
    preprocessed_data.append(fresh_data)

#print(counting_words)
#print(preprocessed_data)

list_of_vocab = []
count = 0
for p in preprocessed_data:
    i=0
    for i in range(len(p)):
        if not p[i] in list_of_vocab:
            count=count+1
            list_of_vocab.append(p[i])
#print(count)
#print(list_of_vocab)

with open('list_of_vocab', 'wb') as f:
    pickle.dump(list_of_vocab, f)
with open('list_of_vocab', 'rb') as f:
    v_list = pickle.load(f)

frequency_list = []
for v in list_of_vocab:
    counter = 0
    for p in preprocessed_data:
        if v in p:
            counter = counter + 1
    frequency_list.append(counter)

with open('Document_Frequency_List', 'wb') as f:
    pickle.dump(frequency_list, f)

with open('Document_Frequency_List', 'rb') as f:
    new_frequency_list = pickle.load(f)

dimensions = (len(files) ,len(new_frequency_list) )
tf_idf_matrix = np.zeros(dimensions)

total_document = len(files)
for i in range(len(v_list)):
    idf = math.log10(total_document/(new_frequency_list[i]+1))
    for j in range(len(files)):
        tf_idf_matrix[j][i] = idf

#print(tf_idf_matrix)

tf_idf_matrix1 = tf_idf_matrix
tf_idf_matrix2 = tf_idf_matrix
tf_idf_matrix3 = tf_idf_matrix
tf_idf_matrix4 = tf_idf_matrix
tf_idf_matrix5 = tf_idf_matrix

#tfid matrix with term weighing scheme (Binary)
for i in range(len(v_list)):
    word = v_list[i]
    for j in range(len(files)):
        if word not in preprocessed_data[j]:
            tf_idf_matrix1[j][i] = 0

with open('tf_idf_bin', 'wb') as f:
    pickle.dump(tf_idf_matrix1, f)

with open('tf_idf_bin', 'rb') as f:
    tf_idf_1 = pickle.load(f)

# tfid matrix with term weighing scheme (Raw count)
for i in range(len(v_list)):
    word = v_list[i]
    for j in range(len(files)):
        if word not in preprocessed_data[j]:
            tf_idf_matrix2[j][i] = 0
        else:
            tf_idf_matrix2[j][i] = tf_idf_matrix2[j][i] * preprocessed_data[j].count(word)

with open('tf_idf_raw', 'wb') as f:
    pickle.dump(tf_idf_matrix2, f)

with open('tf_idf_raw', 'rb') as f:
    tf_idf_2 = pickle.load(f)
"""
# tfid matrix with term weighing scheme (Term frequency)
tf_idf_3 = tf_idf_2
for i in range(len(files)):
    doc_length = len(preprocessed_data[i])
    for j in range(len(v_list)):
        tf_idf_3[i][j] = tf_idf_3[i][j]/doc_length


with open('tf_idf_tf', 'wb') as f:
    pickle.dump(tf_idf_3, f)

with open('tf_idf_tf', 'rb') as f:
    tf_idf_3 = pickle.load(f)
"""
# tfid matrix with term weighing scheme (Log Normalization)
for i in range(len(v_list)):
    word = v_list[i]
    for j in range(len(files)):
        if word not in preprocessed_data[j]:
            tf_idf_matrix4[j][i] = 0
        else:
            tf_idf_matrix4[j][i] = tf_idf_matrix4[j][i] * math.log10(1+preprocessed_data[j].count(word))

with open('tf_idf_log', 'wb') as f:
    pickle.dump(tf_idf_matrix4, f)

with open('tf_idf_log', 'rb') as f:
    tf_idf_4 = pickle.load(f)

# tfid matrix with term weighing scheme (Double Normalization)
for i in range(len(v_list)):
    word = v_list[i]
    for j in range(len(files)):
        if word not in preprocessed_data[j]:
            tf_idf_matrix5[j][i] = tf_idf_matrix5[j][i] * 0.5
        else:
            frequent_word = mode(preprocessed_data[j])
            frequent_word_count = preprocessed_data[j].count(frequent_word)
            tf_idf_matrix5[j][i] = tf_idf_matrix5[j][i] * (0.5 + 0.5*(preprocessed_data[j].count(word)/frequent_word_count))

with open('tf_idf_doub', 'wb') as f:
    pickle.dump(tf_idf_matrix5, f)
with open('tf_idf_doub', 'rb') as f:
    tf_idf_5 = pickle.load(f)


def method1(query_mod, tf_idf):
    tf_idf_value = []
    for i in range(len(files)):
        s = 0
        for q in query_mod:
            j = v_list.index(q)
            s = s + tf_idf[i][j]
        tf_idf_value.append(s)
    Z = [x for _, x in sorted(zip(tf_idf_value, files))]
    Z.reverse()

    return Z, tf_idf_value

query = input("Enter a query")
query_mod = preprocessing(query)
Z,tf_idf_value = method1(query_mod,tf_idf_1)
for i in range(5):
    print(Z[i])
Z,tf_idf_value = method1(query_mod,tf_idf_2)
for i in range(5):
    print(Z[i])
"""
Z,tf_idf_value = method1(query_mod,tf_idf_3)
for i in range(5):
    print(Z[i])
"""
Z,tf_idf_value = method1(query_mod,tf_idf_4)
for i in range(5):
    print(Z[i])
Z,tf_idf_value = method1(query_mod,tf_idf_5)
for i in range(5):
    print(Z[i])