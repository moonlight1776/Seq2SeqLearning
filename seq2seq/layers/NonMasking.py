from keras.engine.topology import Layer


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        return input_shape
