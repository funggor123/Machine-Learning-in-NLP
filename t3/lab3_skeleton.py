# author: ‘CHENG CHI FUNG"
# student_id: "12219691"
import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict


def calculate_bow(corpus):
    """
    Calculate bag of words representations of corpus
    Parameters
    ----------
    corpus: list
        Documents represented as a list of string

    Returns
    ----------
    corpus_bow: list
        List of tuple, each tuple contains raw text and vectorized text
    vocab: list
    """
    # YOUR CODE HERE
    corpus_bow = []
    vocab = set()
    for doc in corpus:
        for word in doc.split():
            vocab.add(word)
    vocab = list(vocab)

    for doc in corpus:
        word_vector = np.zeros((len(vocab)))
        for word_in_vocab in vocab:
            word_count = 0
            for word in doc.split():
                if word == word_in_vocab:
                    word_count = word_count + 1
            word_vector[vocab.index(word_in_vocab)] = word_count
        corpus_bow.append((doc, word_vector))
    return corpus_bow, vocab


def calculate_tfidf(corpus, vocab):
    """
    Parameters
    ----------
    corpus: list of tuple
        Output of calculate_bow()
    vocab: list
        List of words, output of calculate_bow()

    Returns
    corpus_tfidf: list
        List of tuple, each tuple contains raw text and vectorized text
    ----------

    """

    def termfreq(matrix, doc, term):
        try:
            if doc[term] < 1:
                return 0
            else:
                return doc[term] / np.sum(doc)
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            word_in_text_count = 0
            for text in matrix:
                vectorized_text = text[1]
                if vectorized_text[term] > 0:
                    word_in_text_count = word_in_text_count + 1
            return len(matrix) / word_in_text_count
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE
    corpus_tfidf = []
    for text in corpus:
        text_vector = np.zeros((len(vocab)))
        for word_in_bag in vocab:
            index = vocab.index(word_in_bag)
            text_vector[index] = inversedocfreq(corpus, index) * termfreq(corpus, text[1], index)
        corpus_tfidf.append((text[0], text_vector))

    return corpus_tfidf


def cosine_sim(u, v):
    """
    Parameters
    ----------
    u: list of number
    v: list of number

    Returns
    ----------
    cosine_score: float
        cosine similarity between u and v
    """
    # YOUR CODE HERE
    return np.dot(u, v) / (sqrt(np.dot(v, v)) * sqrt(np.dot(u, u)))


def print_similarity(corpus):
    """
    Print pairwise similarities
    """
    for sentx in corpus:
        for senty in corpus:
            print("{:.4f}".format(cosine_sim(sentx[1], senty[1])), end=" ")
        print()
    print()


def q1():
    all_sents = ["this is a foo bar",
                 "foo bar bar black sheep",
                 "this is a sentence"]
    corpus_bow, vocab = calculate_bow(all_sents)
    print(corpus_bow)
    print(vocab)

    print("Test BOW cosine similarity")
    print_similarity(corpus_bow)

    print("Test tfidf cosine similarity")
    # corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf = calculate_tfidf(corpus_bow, vocab)
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)


if __name__ == "__main__":
    q1()

# End of line comment
# C1H2E2N1G 9C6H9I 1F9U9N9G
# 1S2T2U1D9E6N9LT1