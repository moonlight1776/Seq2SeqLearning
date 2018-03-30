from __future__ import absolute_import

import keras
from keras import backend as K
from keras.engine import Layer


class Attention(Layer):
    def __init__(self, embedding_size, activation, kernel_regularizer,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

    def build(self, inputs):
        self.W_a = self.add_weight(
            name='W_aw',
            shape=(
                1,
                self.embedding_size
            ),
            initializer=keras.initializers.RandomUniform(
                minval=-0.1, maxval=0.1, seed=None),
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        self.W_q = self.add_weight(
            name='W_qw',
            shape=(
                1,
                self.embedding_size
            ),
            initializer=keras.initializers.RandomUniform(
                minval=-0.1, maxval=0.1, seed=None),
            trainable=True,
            regularizer=self.kernel_regularizer
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0][1], input_shape[1][1]

    def call(self, inputs, **kwargs):
        # Question attention
        # (1, seq_length)
        part1 = K.dot(self.W_q, K.tf.transpose(inputs[0], [0, 2, 1]))
        # (1, seq_length)
        part2 = K.dot(self.W_a, K.tf.transpose(inputs[1], [0, 2, 1]))
