from keras import backend as K
from keras.engine.topology import Layer


class DynamicMaxPooling1D(Layer):
    def __init__(self, top_k, shape_length, **kwargs):
        # self.layer_index = layer_index
        # self.num_of_layer = num_of_layer
        self.top_k = top_k
        self.shape_length = shape_length
        super(DynamicMaxPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    # Selects most active feature
    def call(self, x):
        # self.ratio = float(self.num_of_layer-self.layer_index)/self.num_of_layer

        # kk =int(self.ratio * x.shape[1].value+1)
        x = K.tf.transpose(
            K.tf.nn.top_k(
                K.tf.transpose(x, [0, 2, 1]),
                # k=K.tf.maximum(
                #     self.top_k,kk
                # )
                k=self.shape_length
            )[0], [0, 2, 1])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.shape_length, input_shape[2]
