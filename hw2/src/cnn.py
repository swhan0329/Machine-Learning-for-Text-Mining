from keras import layers, models, optimizers

def create_cnn_pretrained(embedding_matrix):
    # Add an Input Layer
    input_layer = layers.Input((100,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(10000, 100, weights=[embedding_matrix], trainable=True)(
            input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Conv1D(100, 3, padding='valid', activation="relu",strides=1)(embedding_layer)


    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(1, activation="sigmoid")(pooling_layer)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer1)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((100,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(10000, 100, trainable=True)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Conv1D(100, 3, padding='valid', activation="relu",strides=1)(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(1, activation="sigmoid")(pooling_layer)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer1)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
