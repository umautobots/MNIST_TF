import tensorflow as tf
from tensorflow.keras import Model, layers


def build_model(fft=False):
    _in = layers.Input(shape=(28, 28))
    x = 0.5 * (_in - 127.5)

    if fft:
        x = tf.signal.rfft(x)
        x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x = tf.reshape(x, [-1, 28, 30, 1]) / 28
    else:
        x = tf.reshape(x, [-1, 28, 28, 1])

    for k in range(3):
        x = layers.Conv2D(2**k * 64, kernel_size=3, strides=2, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.Activation('relu')(x)
        x = layers.SpatialDropout2D(0.5)(x)

    x = layers.Flatten()(x)
    _out = layers.Dense(10, activation='softmax')(x)

    return Model(_in, _out)


def allow_gpu_memory_growth():
    # allow memory to grow when needed
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f'{len(gpus):d} Physical GPU(s), {len(logical_gpus):d} Logical GPU(s)')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
