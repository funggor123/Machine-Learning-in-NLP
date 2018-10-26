# author: ‘CHENG CHI FUNG'
# student_id: '12219691'
import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import wordnet as wn

nltk.download('gutenberg')
nltk.download('brown')
nltk.download('wordnet')

# load the words from corpus gutenberg
words = gutenberg.words('austen-sense.txt')

# load the sentences from corpus gutenberg
sents = gutenberg.sents('austen-sense.txt')

# sentences is a list of words
# raw = each character

# load the words from corpus romance
romance_words = brown.words(categories='romance')
# import the data from corpus hobbies and romance
hobbies_words = brown.words(categories='hobbies')


def q1():
    print('q1: {:}'.format(''))
    # 1. Print the number of word tokens
    # YOUR CODE
    print(len(words))

    # 2. Print the number of word types
    # YOUR CODE
    #  Use set function to eliminate the duplicate words
    print(len(set([w.lower() for w in words])))

    # 3. Print all tokens in the first sentence
    # YOUR CODE
    print(sents[0])


def q2():
    print('q2: {:}'.format(''))
    # 1. Print the top 10 most common words in the 'romance' category.
    # Your Code
    # Get the frequency distribution of a list of words
    fdist = nltk.FreqDist(w.lower() for w in romance_words)
    print(fdist.most_common(10))

    # 2.  Print the word frequency of the following words:['ring','activities','love','sports','church'] 
    # in the 'romance'  and 'hobbies' categories respectively. 
    # Your Code
    fdist = nltk.FreqDist(w.lower() for w in hobbies_words )
    models = ['ring', 'activities', 'love', 'sports', 'church']
    print("Hobbies : ")
    for m in models:
        print(m + ':', fdist[m], end=' ')
    print()
    print("Romance : ")
    fdist = nltk.FreqDist(w.lower() for w in romance_words)
    for m in models:
        print(m + ':', fdist[m], end=' ')
    print()


def q3():
    print('q3: {:}'.format(''))
    # 1. Print all synonymous words (lemmas) of the word 'dictionary'
    # Your Code
    #  Synset: a set of synonyms that share a common meaning.
    for w in wn.synsets('dictionary')[0].lemma_names():
        print(w, end=" ")
    print()


    # 2. Print all hyponyms of the word 'dictionary'
    # Your Code

    for w in wn.synsets('dictionary')[0].hyponyms():
        for o in w.lemma_names():
            print(o, end=' ')
            print(wn.synsets(o))
            # Just a testing

    print(wn.synset('gazetteer.n.02').path_similarity(wn.synset('bilingual_dictionary.n.01')))
    print()
    # 3. Use one of the predefined similarity measures to score the similarity of
    # the following pairs of synsets and rank the pairs in order of decreasing similarity.
    # (right_whale.n.01,novel.n.01)
    # (right_whale.n.01,minke_whale.n.01)
    # (right_whale.n.01,tortoise.n.01)
    # synset_names = ['right_whale.n.01', 'novel.n.01', 'minke_whale.n.01', 'tortoise.n.01']
    # Your Code
    right_whale = wn.synset('right_whale.n.01')
    synset_names = ['novel.n.01', 'minke_whale.n.01', 'tortoise.n.01']
    dict = {}
    for w in synset_names:
        dict[w] = right_whale.path_similarity(wn.synset(w))
    for key in sorted(dict, key=dict.get, reverse=True):
        print(key + " : ", dict[key])


if __name__ == '__main__':
    q1()

    print()

    q2()

    print()
    q3()

# End of line comment
# C1H2E2N1G 9C6H9I 1F9U9N9G
# 1S2T2U1D9E6N9LT
