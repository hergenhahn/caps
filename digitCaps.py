import tensorflow as tf
from tensorflow.keras.layers import Layer
from capsules.auxfunc import sprint

#
# squashed length function as given by equation (1)
#
def squashed_len(s, axis = -1):
    squared_norm = tf.reduce_sum(tf.square(s), axis = axis, keepdims = True)
    norm = tf.sqrt(squared_norm)
    unit_vector = s / (norm  + 1e-9) # avoid division by zero
    return unit_vector * squared_norm / (1.0 + squared_norm)

class DigitCaps(Layer):
    def __init__(self,
                 n_inputs,
                 input_vector_len,
                 n_capsules,
                 output_vector_len,
                 routing_iterations=3,
                 kernel_initializer='glorot_uniform',
                 b_initializer='zeros',
                 **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.n_inputs = n_inputs                      # 32x6x6
        self.input_vector_len = input_vector_len      # 8
        self.n_capsules = n_capsules                  # 10
        self.output_vector_len = output_vector_len    # 16
        self.routing_iterations = routing_iterations  # 3

    def build(self, input_shape):
        sprint('input_shape',input_shape)

        # Weight matrix W
        self.W = self.add_weight(shape=[1,
                                 self.n_inputs,
                                 self.n_capsules,
                                 self.output_vector_len,
                                 self.input_vector_len],
                                 name='W')
        sprint('W',self.W)

        # Coupling coefficients for routing.
        self.b = self.add_weight(shape=[1, self.n_inputs, self.n_capsules, 1, 1],
                                  name='b',
                                  trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # Expand dims to (None, n_inputs, 1, 1, input_vector_len)
        bs = inputs.shape[0]
        sprint('inputs',inputs)
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 4)

        sprint('inputs_expand',inputs_expand)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.n_capsules, 1, 1])
        sprint('inputs_tiled',inputs_tiled)

        sprint('W',self.W)
        W_tiled = tf.tile(self.W, [bs, 1, 1, 1, 1])

        sprint('1.W_tiled',W_tiled)
        sprint('2.inputs_tiled',inputs_tiled)
        # matrix muliplication û = (Wij * ui)
        u_hat = tf.matmul(W_tiled, inputs_tiled)
        sprint('1x2 = u_hat',u_hat)

        # Routing algorithm
        self.b = self.b*0.0  # set b o zeros
        sprint('b',self.b)
        for i in range(self.routing_iterations):
            c = tf.nn.softmax(self.b, axis=2)       # dim=2 is the n_capsules dimension
            outputs = squashed_len(tf.keras.backend.sum(c * u_hat, 1, keepdims=True))
            if i != self.routing_iterations - 1:
                self.b = self.b + tf.keras.backend.sum(u_hat * outputs, -1, keepdims=True)
        sprint('b',self.b)
        sprint('output',outputs)
        outr = tf.reshape(outputs, [-1, self.n_capsules, self.output_vector_len])
        sprint('outr',outr)
        return outr
