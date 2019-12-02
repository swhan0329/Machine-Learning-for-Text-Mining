from keras import layers, models, optimizers

def create_rnn_lstm_pretrained(embedding_matrix):
    # Add an Input Layer
    input_layer = layers.Input((100,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(10000, 100,weights=[embedding_matrix], trainable=True)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100,dropout=0.2,recurrent_dropout=0.2, activation="tanh")(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(1, activation="sigmoid")(lstm_layer)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer1)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((100,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(10000, 100, trainable=True)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100,dropout=0.2,recurrent_dropout=0.2, activation="tanh")(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(1, activation="sigmoid")(lstm_layer)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer1)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


