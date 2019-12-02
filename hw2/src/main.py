import TotalDatasetToList as TDTL
import preprocessing as pp
import rnn_lstm, cnn
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_model(classifier, train_x,label, test_x, test_y):
    # fit the training dataset on the classifier
    history = classifier.fit(train_x, label, epochs= 5, batch_size=64 ,validation_data=(test_x, test_y), shuffle=True)

    score, acc = classifier.evaluate(test_x, test_y, batch_size=64)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return score, acc

if __name__ == "__main__":
    test_DF, train_DF = TDTL.TotalDatasetToList()
    train_x, train_y, test_x, test_y, word_index, embedding_matrix = pp.nnPreprocessing(test_DF, train_DF)

    classifier_cnn = cnn.create_cnn_pretrained(embedding_matrix)
    score_cnn_trained, accuracy_cnn_trained = train_model(classifier_cnn, train_x, train_y, test_x,test_y)
    print ("CNN_pretrained, Word Embeddings score: ", score_cnn_trained, "Word Embeddings accuracy: ",accuracy_cnn_trained)


    classifier_rnn = rnn_lstm.create_rnn_lstm_pretrained(embedding_matrix)
    score_rnn_trained, accuracy_rnn_trained  = train_model(classifier_rnn, train_x, train_y, test_x, test_y)
    print ("RNN-LSTM_pretrained, Word Embeddings  score:", score_rnn_trained, "Word Embeddings accuracy: ",accuracy_rnn_trained)

    classifier_cnn = cnn.create_cnn()
    score_cnn, accuracy_cnn = train_model(classifier_cnn, train_x, train_y, test_x,test_y)
    print ("CNN, Word Embeddings score: ", score_cnn, "Word Embeddings accuracy: ",accuracy_cnn)


    classifier_rnn = rnn_lstm.create_rnn_lstm()
    score_rnn, accuracy_rnn  = train_model(classifier_rnn, train_x, train_y, test_x, test_y)
    print ("RNN-LSTM, Word Embeddings  score:", score_rnn, "Word Embeddings accuracy: ",accuracy_rnn)

    import keras
    from matplotlib import pyplot as plt


