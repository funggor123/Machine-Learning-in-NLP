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
import tensorflow as tf

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + list(string.punctuation))


def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g. 
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    # YOUR CODE HERE

    # Split word.word to word. word
    text = re.sub(r'\.(?=[^ \W\d])', '. ', text)

    # Sentence and word tokenize
    sent_tokens = nltk.tokenize.sent_tokenize(text)
    for sent_token in sent_tokens:
        tokens = tokens + nltk.tokenize.word_tokenize(sent_token)
    #  Remove stop words
    filtered_words = [word for word in tokens if word not in stop_words]

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
    lam = 1
    return (P + lam) / (np.sum(P) + len(P) * lam)


# return the best model after training
def train_NB(data_label, data_matrix):
    '''
    :param data_label: [N], type: list
    :param data_matrix: [N(document_number) * V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    return the P(y) (an K array), P(x|y) (a V*K matrix)
    '''
    hyper_params = {}
    other_params = {}

    hyper_params[0] = {1e-3, 1e-4, 1e-5, 1e-6}  # learning rate
    hyper_params[1] = {4}  # epochs
    hyper_params[2] = { data_matrix.shape[0]}  # batch size

    other_params['dimension'] = data_matrix.shape[1]
    other_params['num_class'] = np.max(data_label)

    return train_model_with_different_params(data_matrix, data_label, hyper_params, other_params)


def nn(x, nn_params):
    '''
    :param x: input tensor
    :param nn_params: parameter for nn
    :return:
    '''
    num_class = nn_params['num_class']
    batch_size = nn_params['batch_size']

    # Mask placeholder for dropout layer
    keep_prob = tf.placeholder(tf.float32)

    # Each layer contain 1000 neurons

    # Layer one
    fc1 = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.relu)

    """
     # BN Layer
    bn1 = tf.layers.batch_normalization(fc1, training=True)
    # Drop out
    dp1 = tf.nn.dropout(inputs=bn1,keep_prob=keep_prob)
    # Layer two
    fc2 = tf.layers.dense(inputs=dp1, units=1000, activation=tf.nn.relu)
    # BN Layer
    bn2 = tf.layers.batch_normalization(fc2, training=True)
    # Drop out
    dp1 = tf.nn.dropout(inputs=bn2,keep_prob=keep_prob)
    # Layer three
    """
    fc3 = tf.layers.dense(inputs=fc1, units=num_class, activation=None)

    return fc3


def train_model_with_different_params(input_data, label, hyper_params , other_params):
    '''
    :param input_data: input data
    :param label: label data
    :param hyper_params: hyper parameters
    :param other_params: other parameters
    :return:
    '''
    # Train the model with different hyper parameters
    if hyper_params is None:
        # TODO: // Log space hyperparameter searching
        # Current default = Grid Search
        hyper_params[0] = {1e-3, 1e-4, 1e-5, 1e-6}  # learning rate
        hyper_params[1] = {4}  # epochs
        hyper_params[2] = {}  # batch size

    dimension = other_params["dimension"]
    num_class = other_params["num_class"]

    best_model_acc = -1
    best_model = None

    for lr in hyper_params[0]:
        for epch in hyper_params[1]:
            for bs in hyper_params[2]:

                # train params
                train_params = {"learning_rate": lr, "epochs": epch, "batch_size": bs}
                # Model parameters
                model_params = {"input_size": bs, "dimension": dimension, "num_class": num_class, "learning_rate": lr}

                # train the model and return the model acc
                model_acc = train_model(input_data, label, model(model_params), train_params)

                # Store the best model parameters
                if best_model_acc is -1:
                    best_model_acc = model_acc
                    best_model = train_params
                elif model_acc > best_model_acc:
                    best_model = train_params

                print("current training: ", "learning_rate ", lr, "epochs ", epch, "batch_size ", bs,
                      "accuracy ", model_acc)
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
    session.run(tf.global_variables_initializer())

    # Set up the threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session)

    # Setting up a model
    x, nn_out, y, loss, optimizer = trained_model

    # Since the input data is a sparse matrix, so we need to use special method to convert it to tensor first
    input_data = convert_sparse_matrix_to_sparse_tensor(input_data)

    # Shuffle the data
    shuffle_input, shuffle_labels = tf.train.shuffle_batch([
        input_data,
        label
    ],
        batch_size=train_params['batch_size'],
        num_threads=4,
        capacity=300,
        min_after_dequeue=200
    )

    # Setting
    print_out_rate = 100
    best_accuracy = 0

    # train
    for epch in range(0, train_params['epochs']):
        # Shuffle the data
        input_d, label_d = session.run([shuffle_input, shuffle_labels])
        # Set up the data feed
        feed_dict = {x: input_d, y: label_d}
        # decent one step
        loss, accuracy = train_one_step(optimizer, loss, train_accuracy(y, nn_out), feed_dict, session)
        if epch % print_out_rate:
            print("epch : ", epch, " accuracy :", accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    # stop the threads
    coord.request_stop()
    coord.join(threads)
    return best_accuracy


def train_accuracy(y, nn_out):
    """
    :param y: label y tensor
    :param nn_out: y_out tensor
    :return:
    """
    accuracy_matrix = tf.cast(tf.equal(y, tf.cast(tf.arg_max(nn_out, dimension=1), tf.int32)), tf.int32, )
    return tf.reduce_sum(accuracy_matrix) / tf.shape(accuracy_matrix)


def train_one_step(optimizer, loss, accuracy, feed_dict, session):
    """
    :param optimizer: optimizer
    :param loss: loss tensor
    :param accuracy: accuracy tensor
    :param feed_dict: data feed
    :param session: tf session
    :return:
    """
    # One gradient decent
    optimizer = optimizer.minimize(loss)
    return session.run([optimizer, accuracy], feed_dict)


def model(model_params=None):
    """
    :param model_params: model params
    :return:
    """
    if model_params is None:
        model_params = {}

    # default learning rate is 1e-3
    lr = model_params.get('learning_rate')
    dimension = model_params.get('dimension')
    num_class = model_params.get('num_class')
    batch_size = model_params.get('batch_size')

    # Placeholder for NN
    x = tf.placeholder(tf.float32, (batch_size, dimension))
    y = tf.placeholder(tf.int32, batch_size)

    # Set NN params
    nn_params = {'num_class': num_class, "batch_size": batch_size}

    # Set NN
    y_out = nn(x, nn_params)

    # Loss Function and weight updated rules
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    return x, y_out, y, loss, optimizer


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def predict_NB(data_matrix, P_y, P_xy, best_model):
    """
    :param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    :param P_y: [K(class number)], type: np.ndarray
    :param P_xy: [V, K], type: np.ndarray
    return data_pre (a N array)
    """
    # TODO: prediction and save the weight
    return


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
    #train_data_pre = predict_NB(train_data_matrix, P_y, P_xy)

    ''''
    acc, precision, recall, f1 = evaluate(train_data_label, train_data_pre)
    print("Evalution: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))
    )
    test_data_pre = predict_NB(test_data_matrix, P_y, P_xy)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = test_data_pre
    sub_df.to_csv("submission.csv", index=False
    '''''

# End of line comment
# C1H2E2N1G 9C6H9I 1F9U9N9G
# 1S2T2U1D9E6N9LT1
