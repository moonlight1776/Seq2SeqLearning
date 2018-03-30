from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer
from keras.layers import multiply


class Gated(Layer):
    def __init__(self, **kwargs):
        super(Gated, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return multiply([inputs[0], K.sigmoid(inputs[1])])
