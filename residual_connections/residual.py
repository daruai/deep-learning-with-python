from tensorflow import keras
from tensorflow.keras import layers

def residual_block(x, filters, conv_num=3, activation='relu', pooling=False):
    """A residual block.
    Arguments:
        x: input tensor.
        filters: integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        conv_num: integer, the number of conv layer in the block.
        activation: activation function.
        pooling: boolean, whether to use max pooling at the end of the block.
    Returns:
        Output tensor for the residual block.
    """
    residual=x
    for _ in range(conv_num):
        x = layers.Conv2D(filters, 3, padding='same', activation=activation)(x)

    if pooling:
        x = layers.MaxPooling2D(2, padding='same')(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual) # 1x1 conv to project the residual to the expected shape
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual) # 1x1 conv to project the residual to the expected shape

    x = layers.add([x, residual])
    return x

def build_residual_model(input_shape, num_classes):
    """Instantiates a ResNet-like architecture.
    Arguments:
        input_shape: tuple, shape of input image tensor.
        num_classes: integer, number of classes.
    Returns:
        A Keras model instance.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = residual_block(x, 32, 2, pooling=True)
    x = residual_block(x, 64, 2, pooling=True)
    x = residual_block(x, 128, 2, pooling=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax' if num_classes > 0 else 'sigmoid')(x)
    return keras.Model(inputs, x)

if __name__ == '__main__':
    model = build_residual_model((32, 32, 3), 1)
    model.summary()