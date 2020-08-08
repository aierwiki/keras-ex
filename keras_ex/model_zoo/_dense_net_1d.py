import numpy as np
import keras 
from keras.models import Model 
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D
)
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.regularizers import l2
from keras import backend as K


def _conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu, 2 * 2 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution1d added (optional bottleneck)
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    if bottleneck:
        inner_channel = nb_filter * 4 
        x = Conv1D(inner_channel, 1, kernel_initializer='he_normal', padding='causal',
            use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
    x = Conv1D(nb_filter, 2, kernel_initializer='he_normal', padding='causal', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x



def _dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, 
                weight_decay=1e-4, grow_nb_filters=True, return_concat_list=False):
    """Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]
    for i in range(nb_layers):
        cb = _conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)
        if grow_nb_filters:
            nb_filter += growth_rate
    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

def _transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    """Apply BatchNorm, Relu 1x1, Conv1D, optinal compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculate as 1 - reduction. Reduces the number of feature maps in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    """
    concat_axis = 1 if K.image_data_format() == "channels_first" else -1
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv1D(int(nb_filter * compression), 1, kernel_initializer='he_normal',
                padding='causal', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    return x


class DenseNet1DBuilder(object):
    @staticmethod
    def build_dense_net(inputs_shape, num_outputs, nb_layers_per_block=-1, growth_rate=12, nb_filter=-1,
                bottleneck=False, reduction=0.0,
                dropout_rate=None, weight_decay=1e-4, activation="softmax"):
        """Builds a custom DenseNet

        Args:
            inputs_shape: The inputs shape in the form (nb_sequence_size, nb_channels)
            num_outputs: The number of outputs at final softmax layer
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_blok from the depth of the network. 
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must be (nb_dense_block + 1)
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks. Note : reduction value is inversed to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'
        """
        concat_axis = -1
        if reduction != 0.0:
            assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'
        nb_dense_block= len(nb_layers_per_block) - 1
        final_nb_layer = nb_layers_per_block[-1]
        nb_layers = nb_layers_per_block[:-1]
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate 
        compression = 1.0 - reduction 
        initial_kernel = 2
        initial_strides = 1
        inputs = Input(shape=inputs_shape)
        x = Conv1D(nb_filter, initial_kernel, kernel_initializer="he_normal", padding="causal", 
                    strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(inputs)
        
        # Add dense blocks
        for block_idx in range(nb_dense_block):
            x, nb_filter = _dense_block(x, nb_layers[block_idx], nb_filter, growth_rate,
                            bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay)
            x = _transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
            nb_filter = int(nb_filter * compression)

        # The last dense block does not have a transition_block
        x, nb_filter = _dense_block(x, final_nb_layer, nb_filter, growth_rate,
                        bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        block_shape = K.int_shape(x)
        x = AveragePooling1D(pool_size=block_shape[1], strides=1)(x)
        flatten = Flatten()(x)
        x = Dense(num_outputs, activation=activation)(flatten)
        model = Model(inputs=inputs, outputs=x)
        return model


def test_1():
    dense_net = DenseNet1DBuilder.build_dense_net((100, 4), 1, [2, 2, 2, 2], growth_rate=8, reduction=0.5, activation="sigmoid")
    X = np.random.random((1000, 100, 4))
    y = np.random.randint(0, 2, size=1000)
    dense_net.compile(loss="binary_crossentropy", optimizer="adam")
    dense_net.fit(X, y, batch_size=64, epochs=2)


if __name__ == "__main__":
    test_1()

