import keras
from keras.engine.topology import Layer


class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name='bias',
                                 shape=(1, input_shape[2]),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        return inputs + self.b

    def compute_output_shape(self, input_shape):
        return input_shape
