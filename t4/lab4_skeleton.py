# author: ‘CHENG CHI FUNG"
# student_id: "12219691"
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedKFold

stop_words = set(stopwords.words('english') + list(string.punctuation))


contractions_dict = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}

def tokenize(text):
    '''
       :param text: a doc with multiple sentences, type: str
       return a word list, type: list
       https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
       e.g.
       Input: 'It is a nice day. I am happy.'
       Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
       '''

    wnl = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))



    def expand_contractions(s, contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, s)

    text = expand_contractions(text)

    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    regex = r"(?<!\d)[.,;:](?!\d)"

    text = re.sub(regex, "", text, 0)

    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)

    stop = nltk.corpus.stopwords.words('english')
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in tokens if w not in stop)
    mostCommon = nltk.FreqDist(dict(allWordExceptStopDist.most_common()[-500:]))

    # Filter Common
    non_common = [word for word in tokens if word not in mostCommon]

    # Stemming
    stem_words = [porter_stemmer.stem(word) for word in non_common]


    #  Remove stop words
    filtered_words = [word for word in stem_words if word not in stop_words]

    # re-create document from filtered tokensf word not in stop_words]

    return filtered_words


def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))
    # YOUR CODE HERE
    index = 0
    for doc in data:
        for word in doc:
            if vocab_dict.get(word) is not None:
                data_matrix[index, vocab_dict[word]] += 1
        index += 1
    return data_matrix


def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)

    return df['id'], df['label'], data_matrix, vocab


def normalize(P):
    """
    normalize P to make sure the sum of the first dimension equals to 1
    e.g.
    Input: [1,2,1,2,4]
    Output: [0.1,0.2,0.1,0.2,0.4] (without laplace smoothing) or [0.1333,0.2,0.1333,0.2,0.3333] (with laplace smoothing)
    """
    # YOUR CODE HERE
    # With Laplace Smoothing
    # (n+lambda)/(N+(lambda*N))
    # 1 >= lambda >= 0
    lam = 0.97
    return (P + lam) / (np.sum(P) + len(P) * lam)


def train_NB(data_label, data_matrix):
    '''
    	:param data_label: [N], type: list
    	:param data_matrix: [N(document_number) * V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    	return the P(y) (an K array), P(x|y) (a V*K matrix)
    	'''
    N = data_matrix.shape[0]
    K = max(data_label)  # labels begin with 1
    # YOUR CODE HERE
    data_delta = np.zeros((N, K))  # N * K
    for i, l in enumerate(data_label):
        data_delta[i, l - 1] = 1
    P_y = normalize(np.sum(data_delta, axis=0, keepdims=False))
    P_xy = normalize(data_matrix.transpose().dot(data_delta))
    return P_y, P_xy


def predict_NB(data_matrix, P_y, P_xy):
    '''
    	:param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    	:param P_y: [K(class number)], type: np.ndarray
    	:param P_xy: [V, K], type: np.ndarray
    	return data_pre (a N array)
    	'''
    # compute the label probabilities using the P(y) and P(x|y) according to the naive Bayes algorithm
    # YOUR CODE HERE
    log_P_y = np.expand_dims(np.log(P_y), axis=0)  # N * K
    log_P_xy = np.log(P_xy)
    log_P_dy = data_matrix.dot(log_P_xy)
    log_P = log_P_y + log_P_dy

    # get labels for every document by choosing the maximum probability
    # YOUR CODE HERE
    return np.argmax(log_P, axis=1) + 1


def evaluate(y_true, y_pre):
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1


if __name__ == '__main__':
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv")
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", vocab)
    print("Test Set Size:", len(test_id_list))

    P_y, P_xy = train_NB(train_data_label, train_data_matrix)
    train_data_pre = predict_NB(train_data_matrix, P_y, P_xy)
    acc, precision, recall, f1 = evaluate(train_data_label, train_data_pre)
    print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    test_data_pre = predict_NB(test_data_matrix, P_y, P_xy)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = test_data_pre
    sub_df.to_csv("submission.csv", index=False)

# End of line comment
# C1H2E2N1G 9C6H9I 1F9U9N9G
# 1S2T2U1D9E6N9LT1
