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
from sklearn.decomposition import PCA
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

    # remove stop words
    meaningful_words = [w for w in stemmed_words if w not in stops]

    # rejoin meaningful stemmed words
    joined_words = (' '.join(meaningful_words))

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

    # create initial dataframe
    df = pd.DataFrame(flattened_movie_review_data, columns=['movie_review_words'])

    # get frequency count of words based on grouped column values
    df = df.apply(pd.Series.value_counts).fillna(0)

    # get top 5 percent of words that appear in 10 percent of docs
    # TODO: check for terms across all documents and only keep the ones that appear
    #  in <=10 percent of docs (noise reduction technique)
    df = df.head(int(len(df) * (5 / 100)))

    # convert index (word) values to new dataframe
    df = pd.DataFrame(list(df.index.values), columns=['movie_review_words'])

    desc = df['movie_review_words'].values
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(desc)

    # principal component analysis (pca) - reduce dimensionality
    pca = PCA()

    # TODO: error - pca does not support sparse input. see truncatedsvd for a possible alternative
    # TODO: if PCA still does not work, truncatedsvd is another possible method
    pca.fit_transform(x)
    pca_variance = pca.explained_variance_

    plt.figure(figsize=(8, 6))
    plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
    plt.legend()
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()

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


try:
    main()
except Exception as err:
    print(err)
    sys.exit(1)
