# author: ‘CHENG CHI FUNG"
# student_id: "12219691"

from keras.models import Sequential
from keras import layers
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection._split import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

nltk.download('averaged_perceptron_tagger')

contractions_dict = {
    "ai n't": "am not / are not / is not / has not / have not",
    "are n't": "are not / am not",
    "ca n't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "did n't": "did not",
    "doesn't": "does not",
    "do n't": "do not",
    "had n't": "had not",
    "had n't've": "had not have",
    "has n't": "has not",
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
    "you've": "you have",
    "n't": "not"
}


def usekeras(X_train, y_train, X_test, y_test):
    y_train = y_train - 1
    y_test = y_test - 1
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    input_dim = X_train.shape[1]  # Number of features
    model = Sequential()

    model.add(layers.Dense(12, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

def embb(val):
    def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # Main settings
    epochs = 20
    embedding_dim = 50
    maxlen = 100

    df = pd.read_csv("data/train.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=1000)

    y_train = y_train - 1
    y_test = y_test - 1

    tokenizer = Tokenizer(num_words=15000)
    tokenizer.fit_on_texts(x_train)

    X_train = tokenizer.texts_to_sequences(x_train)
    X_test = tokenizer.texts_to_sequences(x_test)
    #  val = tokenizer.texts_to_sequences(val)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 200

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # Parameter grid for grid search
    param_grid = dict(num_filters=[32],
                      kernel_size=[3, 5],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=True)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)
    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)
    print(grid_result)
    print(test_accuracy)
    # prediction = model.predict(val)
    return -1


stop_words = set(stopwords.words('english'))


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


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

    text = text.lower()

    text = expand_contractions(text)

    regex = r"[^a-zA-Z ]"

    text = re.sub(regex, " ", text, 0)

    # text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore')

    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)

    # Lemma
    lemma = [wnl.lemmatize(word) for word in tokens]


    # Stemming
    stem_words = [porter_stemmer.stem(word) for word in lemma]


    return stem_words

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
    vectorizer = CountVectorizer(tokenizer=lambda text: tokenize(text), vocabulary=vocab, min_df=0.01, max_df=0.99,
                                 )
    X = vectorizer.fit_transform(df['text'])
    print(len(vectorizer.vocabulary_))
    return df['id'], df['label'], X.toarray(), vectorizer.vocabulary_

# return the best model after training
def train_NB(data_label, data_matrix):
    hyper_params = {}
    other_params = {}

    # Parameters Tuning
    hyper_params[0] = {0.001}  # learning rate
    hyper_params[1] = {5000}  # epochs
    hyper_params[2] = {2048}  # batch size

    other_params['dimension'] = data_matrix.shape[1]
    other_params['num_class'] = np.max(data_label)
    print("Dimension:", other_params['dimension'], "Num Class:", other_params['num_class'])

    return train_model_with_different_params(data_matrix, data_label, hyper_params, other_params)


def nn(x, nn_params, isTraining=True):
    num_class = nn_params['num_class']

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

    # Each layer contain 500 neurons
    # Layer one
    fc1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.elu, kernel_regularizer=regularizer)
    # BN Layer
    bn1 = tf.layers.batch_normalization(fc1, training=isTraining)
    # Drop out
    dp1 = tf.layers.dropout(bn1, training=isTraining)
    # Layer two
    fc2 = tf.layers.dense(inputs=dp1, units=128, activation=tf.nn.elu, kernel_regularizer=regularizer)
    # BN Layer
    bn2 = tf.layers.batch_normalization(fc2, training=isTraining)
    # Drop out
    dp2 = tf.layers.dropout(bn2, training=isTraining)
    # Layer three
    fc3 = tf.layers.dense(inputs=dp2, units=num_class, activation=None)
    return fc3


def train_model_with_different_params(input_data, label, hyper_params, other_params):
    print("Start training")

    # Train the model with different hyper parameters
    if hyper_params is None:
        # TODO: // Log space hyperparameter searching
        # Current default = Grid Search
        hyper_params[0] = {1e-3}  # learning rate
        hyper_params[1] = {3000}  # epochs
        hyper_params[2] = {8000}  # batch size

    dimension = other_params["dimension"]
    num_class = other_params["num_class"]

    best_model_acc = -1
    best_model = None
    label = label - 1

    for lr in hyper_params[0]:
        for epch in hyper_params[1]:
            for bs in hyper_params[2]:

                # train params
                train_params = {"learning_rate": lr, "epochs": epch, "batch_size": bs}
                # Model parameters
                model_params = {"dimension": dimension, "num_class": num_class, "learning_rate": lr}

                # train the model and return the model acc
                model_acc, session = train_model(input_data, label, model(model_params), train_params)

                # Create saver
                saver = tf.train.Saver(max_to_keep=1)

                # Store the best model parameters
                if best_model_acc == -1:
                    best_model_acc = model_acc
                    best_model = model_params
                    save_path = saver.save(session, "/tmp/model/model.ckpt")
                    print("best model parameters save to", save_path)
                elif model_acc > best_model_acc:
                    best_model_acc = model_acc
                    best_model = model_params
                    save_path = saver.save(session, "/tmp/model/model.ckpt")
                    print("best models parameters save to", save_path)

                print("current training: ", "learning_rate ", lr, "epochs ", epch, "batch_size ", bs,
                      "accuracy ", model_acc)
    print("The best model with: ", best_model, " and acc: ", best_model_acc)
    print("End Training")
    return best_model


def train_model(input_data, label, trained_model, train_params):
    '''
    :param input_data: input data
    :param label: label
    :param trained_model: model
    :param train_params: train parameters
    :return:
    '''
    # Start a tf session and init a global initializer
    session = tf.Session()

    # Setting up a model
    x, nn_out, y, loss, optimizer = trained_model

    # Since the input data is a sparse matrix, so we need to use special method to convert it to tensor first
    # sparse = convert_sparse_matrix_to_sparse_tensor(input_data)
    # dense = tf.data.Dataset.from_tensors(sparse)

    X_init = tf.placeholder(tf.float32, shape=(input_data.shape[0], input_data.shape[1]))
    Y_init = tf.placeholder(tf.float32, shape=(label.shape[0]))

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((X_init, Y_init))

    seg_dataSet = dataset.batch(batch_size=train_params['batch_size']).shuffle(1000).repeat(train_params['epochs'])
    #
    iterator = seg_dataSet.make_initializable_iterator()
    next_element = iterator.get_next()

    # Setting
    print_out_rate = 100
    best_accuracy = 0

    # One gradient decent
    optimizer = optimizer.minimize(loss)

    session.run(tf.global_variables_initializer())
    session.run(iterator.initializer, feed_dict={X_init: input_data, Y_init: label})
    accuracy_t = 0
    total = 0
    # train
    for epch in range(0, train_params['epochs']):
        # Shuffle the data
        input_d, label_d = session.run(next_element)
        # Set up the data feed
        feed_dict = {x: input_d, y: label_d}
        # decent one step
        _, accuracy, los = train_one_step(optimizer, train_accuracy(y, nn_out), feed_dict, session, loss)
        accuracy_t += accuracy
        total += 1
        print(accuracy_t / total)
        if epch % print_out_rate:
            print("epch : ", epch, " accuracy :", accuracy, "loss", los)
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return best_accuracy, session


def train_accuracy(y, nn_out):
    """
    :param y: label y tensor
    :param nn_out: y_out tensor
    :return:
    """
    accuracy_matrix = tf.cast(tf.equal(y, tf.cast(tf.argmax(nn_out, 1), tf.int32)), tf.int32)
    return tf.reduce_sum(accuracy_matrix) / tf.shape(accuracy_matrix)


def train_one_step(optimizer, accuracy, feed_dict, session, loss):
    """
    :param optimizer: optimizer
    :param loss: loss tensor
    :param accuracy: accuracy tensor
    :param feed_dict: data feed
    :param session: tf session
    :return:
    """

    return session.run([optimizer, accuracy, loss], feed_dict)


def model(model_params=None, isTraining=True):
    if model_params is None:
        model_params = {}

    # default learning rate is 1e-3
    lr = model_params['learning_rate']
    dimension = model_params['dimension']
    num_class = model_params['num_class']

    # Placeholder for NN
    x = tf.placeholder(tf.float32, (None, dimension))
    y = tf.placeholder(tf.int32, None)

    # Set NN params
    nn_params = {'num_class': num_class}

    # Set NN
    y_out = nn(x, nn_params, isTraining)

    l2_loss = tf.losses.get_regularization_loss()

    # Loss Function and weight updated rules
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=y))) + l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    return x, y_out, y, loss, optimizer


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def predict_NB(data_matrix, best_params):
    print("Start Prediction")

    tf.reset_default_graph()

    # Setting up a model
    x, nn_out, y, loss, optimizer = model(best_params, isTraining=False)

    y_out = tf.argmax(nn_out, axis=1) + 1

    # Set up the model
    saver = tf.train.Saver()

    session = tf.Session()

    # Load the best parameters
    saver.restore(session, "/tmp/model/model.ckpt")

    # Feed the data
    feed_dict = {x: data_matrix}

    # Predict
    y_o = session.run(y_out, feed_dict)

    print("End Prediction")

    return y_o


def evaluate(y_true, y_pre):
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1


if __name__ == '__main__':
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
    sess = tf.Session(config=config)

    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv", None)
    print("Training Set Size:", len(train_id_list))
    print(vocab)
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", vocab)
    #print("Test Set Size:", len(test_id_list))

    #best_model_params = train_NB(train_data_label, train_data_matrix)

    #train_data_pre = predict_NB(train_data_matrix, best_model_params)

    #acc, precision, recall, f1 = evaluate(train_data_label, train_data_pre)
    #print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    #test_data_pre = predict_NB(test_data_matrix,best_model_params)
    clf = GridSearchCV(cv=None,
                 estimator=LogisticRegression(C=1.0, intercept_scaling=1,
                                              dual=False, fit_intercept=True, penalty='l2', tol=0.0001, max_iter=500, solver='saga', multi_class='multinomial' ,verbose=True),
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

                             })
    clf.fit(train_data_matrix[0:8000], train_data_label[0:8000])

    Z = clf.predict_proba(train_data_matrix[8000:16000])
    train_data_pre = np.argmax(Z, axis=1)+1

    acc, precision, recall, f1 = evaluate(train_data_label[8000:16000], train_data_pre)
    print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    #Z = logreg.predict_proba(train_data_matrix[8000:16000])
    #valid_data_pre = np.argmax(Z, axis=1) + 1

    #ub_df = pd.DataFrame()
    #sub_df["id"] = test_id_list
    #sub_df["pred"] = valid_data_pre
    #sub_df.to_csv("submission.csv", index=False)

# End of line comment
# C1H2E2N1G 9C6H9I 1F9U9N9G
# 1S2T2U1D9E6N9LT1
