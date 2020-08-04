import numpy as np 
import keras
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

def _bn_relu(inputs):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=2)(inputs)
    return Activation("relu")(norm)
    
def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", 1)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "causal")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    def f(inputs):
        activation = _bn_relu(inputs)
        return Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                        kernel_initializer=kernel_initializer, 
                        kernel_regularizer=kernel_regularizer)(activation)
    return f
    
def _shortcut(inputs, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    Expand channels of shortcut to match residual.
    Stride appropriately to match residual (n_sequence_size)
    Should be int if network architecture is correcty configured
    """
    inputs_shape = K.int_shape(inputs)
    residual_shape = K.int_shape(residual)
    equal_channels = inputs_shape[2] == residual_shape[2]
    shortcut = inputs
    # 1 x 1 conv if shape is different. Else identity.
    if not equal_channels:
        shortcut = Conv1D(filters=residual_shape[2],
                        kernel_size=1, strides=1, padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(1.e-4))(inputs)
    return add([shortcut, residual])
    
def bottleneck(filters, init_strides=1):
    """Bottleneck architecture
    
    Returns:
        A final conv layer of filters * 4
    """
    def f(inputs):
        conv_1 = _bn_relu_conv(filters=filters, kernel_size=1, strides=init_strides)(inputs)
        conv_2 = _bn_relu_conv(filters=filters, kernel_size=2)(conv_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=1)(conv_2)
        return _shortcut(inputs, residual)
    return f
    
def _residual_block(block_function, filters, repetitions):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(inputs):
        for i in range(repetitions):
            inputs = block_function(filters=filters, init_strides=1)(inputs)
        return inputs
    return f  
    
class Resnet1DBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, repetitions):
        """Builds a custom ResNet

        Args:
            input_shape: The input shape in the form (nb_sequence_size, nb_channels)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use.
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        
        Returns:
            The keras `Model`
        """
        if len(input_shape) != 2:
            raise Exception("Input shape should be a tuple (nb_sequence_size, nb_channels)")
        inputs = Input(shape=input_shape)
        conv1 = Conv1D(filters=64, kernel_size=2, strides=1, padding='causal',
                        kernel_initializer="he_normal", kernel_regularizer=l2(1.e-4))(inputs)    
        block = conv1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(bottleneck, filters=filters, repetitions=r)(block)
            filters *= 2
        
        # last activation
        block = _bn_relu(block)
        
        # Classifier block
        block_shape = K.int_shape(block)
        pool = AveragePooling1D(pool_size=block_shape[1], strides=1)(block)
        flatten = Flatten()(pool)
        if num_outputs == 2:
            dense = Dense(units=1, kernel_initializer="he_normal",
                        activation="sigmoid")(flatten)
        else:
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                        activation="softmax")(flatten)
        model = Model(inputs=inputs, outputs=dense)
        return model
        
def test():
    X = np.random.random((1000, 100, 4))
    y = np.random.randint(0, 2, size=1000)
    resnet = Resnet1DBuilder.build((100, 4), 2, [2, 2, 2, 2])
    resnet.compile(loss='binary_crossentropy', optimizer='rmsprop')
    resnet.fit(X, y)
    