from sklearn import preprocessing
from keras.preprocessing import text, sequence
import numpy, collections


def nnPreprocessing(test_DF, train_DF):
    # split the dataset into training and validation datasets
    train_DF.sample(frac=1)
    test_DF.sample(frac=1)
    train_x = train_DF['text']
    test_x = test_DF['text']
    train_y = train_DF['label']
    test_y = test_DF['label']

    # load the pre-trained word-embedding vectors
    embeddings_index = {}
    for i, line in enumerate(open('../data/all.review.vec.txt')):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

    # create a tokenizer
    token = text.Tokenizer(num_words=10000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    token.fit_on_texts(train_DF['text'])
    word_index = token.word_index
    word_count = token.word_counts
    word_count = collections.OrderedDict(sorted(word_count.items(), key=lambda t : t[1], reverse=True))
    temp_word_count_key = list(word_count.keys())
    word_count_key=[]
    for i in range(100,10100):
        word_count_key.append(temp_word_count_key[i])

    print("the total number of unique words in T: ", len(word_index))
    final_word_index=dict()
    for i in range(0,10000):
        final_word_index[word_count_key[i]]=i

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    train_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=100)
    test_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=100)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)

    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(final_word_index), 100))
    for word, i in final_word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return train_x, train_y, test_x, test_y,  final_word_index, embedding_matrix

