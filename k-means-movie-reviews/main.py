# !/usr/bin/env python3
__author__ = 'agoss'

from collections import OrderedDict
import glob
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

stemming = PorterStemmer()
stops = set(stopwords.words("english"))


def preprocess_movie_reviews(polarity, movie_review_data):
    for file in glob.glob(os.path.join('./review_polarity/' + polarity + '/', "*.txt")):
        print(file)
        file_gen = (line for line in open(file))
        rows = list(file_gen)

        for idx, row in enumerate(rows):
            # apply data cleaning function to row text
            rows[idx] = clean_text(row)

        movie_review_data.extend(list(OrderedDict.fromkeys(rows)))
    return movie_review_data


def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) tokenizes (breaks down into words)
        2) removes punctuation and non-word text
        3) finds word stems
        4) rejoins meaningful stem words"""

    # tokenize
    tokens = nltk.word_tokenize(raw_text)

    # keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]

    # stemming
    stemmed_words = [stemming.stem(w) for w in token_words]

    # # remove stop words and duplicates
    # meaningful_words = [w for w in stemmed_words if w not in stops]
    # meaningful_words = list(OrderedDict.fromkeys(meaningful_words))

    # rejoin meaningful stemmed words
    joined_words = (' '.join(stemmed_words))

    # return cleaned data
    return joined_words


def main():
    # initialize lists
    movie_review_data = []
    flattened_movie_review_data = []
    polarities = ['pos', 'neg']

    # preprocess positive and negative movie review data
    for polarity in polarities:
        movie_review_data.extend(preprocess_movie_reviews(polarity, movie_review_data))

    # flatten all movie review text into list of words
    for line in movie_review_data:
        flattened_movie_review_data.extend(line.split(' '))

    # create dataframe
    df = pd.DataFrame(flattened_movie_review_data, columns=['movie_review_words'])
    # df = df.drop_duplicates('movie_review_words')
    desc = df['movie_review_words'].values

    # TODO: identify number of times unique token appears
    # TODO: limit to top 5% that occur in no more than 10%, reduce dimensionality
    #  - we care about less things and what is left indicates key categories (trimming step needed)

    # remove stop words
    vectorizer = TfidfVectorizer(stop_words=stops)

    x = vectorizer.fit_transform(desc)
    wcss = []

    for i in range(1, 2):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 2), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow.png')
    plt.show()

    # distortions = []
    # k_range = range(1, 10)
    # for k in k_range:
    #     kmean_model = KMeans(n_clusters=k)
    #     kmean_model.fit(df)
    #     distortions.append(kmean_model.inertia_)
    #
    # plt.figure(figsize=(16, 8))
    # plt.plot(k_range, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    #
    # # determine number of clusters using elbow method
    # vectorizer = CountVectorizer()
    # x = vectorizer.fit_transform(movie_review_words)
    # print(vectorizer.get_feature_names())
    # print(x.toarray())
    #
    # # visualize where 'elbow' forms
    # sum_of_squared_distances = []
    # k = range(2, 10)
    # for k in k:
    #     km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    #     km = km.fit(x)
    #     sum_of_squared_distances.append(km.inertia_)
    # plt.plot(k, sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()


    # df = pd.DataFrame(data, columns=['line'])
    # print(df.head())
    # print(df)

    # df = pd.read_csv('./review_polarity/txt_sentoken/neg/cv000_29416.txt', header=None)
    # df.head()

    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('./review_polarity/txt_sentoken/neg/', "*.txt"))))
    # df.head()

try:
    main()
except Exception as err:
    print(err)
    sys.exit(1)
