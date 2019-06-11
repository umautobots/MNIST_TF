from tensorflow.keras import Model, layers


def build_model():
    _in = layers.Input(shape=(28, 28))
    x = layers.Reshape((28, 28, 1))(_in)
    for k in range(3):
        x = layers.Conv2D(2**k * 64, kernel_size=3, strides=2, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    _out = layers.Dense(10, activation='softmax')(x)

    return Model(_in, _out)
