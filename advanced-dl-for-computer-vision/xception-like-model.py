from tensorflow import keras
from tensorflow.keras import layers

def build_xception_model(input_shape, num_classes):
    """Instantiates a Xception-like architecture.
    Arguments:
        input_shape: tuple, shape of input image tensor.
        num_classes: integer, number of classes.
    Returns:
        A Keras model instance.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, kernel_size=5, use_bias=False)(x)

    for size in [32, 64, 128, 256]:
        residual = x

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, kernel_size=3, padding='same', use_bias=False)(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, kernel_size=3, padding='same', use_bias=False)(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(size, kernel_size=1, strides=2, padding='same', use_bias=False)(residual)
        x = layers.add([x, residual])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax' if num_classes > 0 else 'sigmoid')(x)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    model = build_xception_model((32, 32, 3), 1)
    model.summary()