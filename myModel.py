import tensorflow as tf
from capsules.digitCaps import DigitCaps
from capsules.auxfunc import *

# several constants for sizes of
# capsules:
caps1_w = 6                 # width of primary capsules
caps1_h = 6                 # height of primary capsules
caps1_count = 32            # number of primary capsules
caps1_ov_len = 8            # primary capsules output vector length
caps2_iv_len = 8            # secondary capsules input vector length
caps2_ov_len = 16           # number of dimensions from capsules
# decoder:
n_decoder_l1 = 512          # size of first decoder layer
n_decoder_l2 = 1024         # size of first decoder layer
n_output = 28 * 28          # size of decoder output
n_categories = 10

# margins and lambda for digit existence ( equation (4)):
margin_plus = 0.9
margin_minus = 0.1
llambda = 0.5
alpha = 0.0005

def norm(s, axis = -1, keepdims = False):
  squared_norm = tf.reduce_sum(tf.square(s), axis = axis, keepdims = keepdims)
  return tf.sqrt(squared_norm)

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(input_shape = (28, 28, 1), filters = 256, kernel_size =(9,9),
            strides =1, padding = "valid", activation = tf.nn.relu)

    self.conv2 = tf.keras.layers.Conv2D(input_shape = (caps1_w, caps1_h, 256), filters = caps1_count * caps1_ov_len,
            kernel_size =(9,9), strides =2, padding = "valid", activation = tf.nn.relu)
    self.dcaps = DigitCaps(caps1_count*caps1_h*caps1_w, caps2_iv_len, 10, caps2_ov_len)
    self.decoder1 = tf.keras.layers.Dense(n_decoder_l1, activation = tf.nn.relu)
    self.decoder2 = tf.keras.layers.Dense(n_decoder_l2,input_shape=(n_decoder_l1,), activation = tf.nn.relu)
    self.decoder3 = tf.keras.layers.Dense(n_output,input_shape=(n_decoder_l2,), activation = tf.nn.sigmoid)

  def call(self, inputs, training=False):
    if not inputs['bypass']:    # execute capsule layers
      original_image = inputs['x']
      x = self.conv1(original_image)
      x = self.conv2(x, training=training)
      x = tf.keras.layers.Reshape(target_shape=[-1, caps2_iv_len])(x)
      v_j = self.dcaps(x)
      sprint('v_j ',v_j )
      y_proba = norm(v_j, axis = -1)
    else:                       # bypass capsule layer, feed input to decoder
      y_proba = inputs['v']
      original_image = None
    sprint('y_proba',y_proba)
    y_proba_argmax = tf.argmax(y_proba, axis = 1) #new
    sprint('y_proba_argmax',y_proba_argmax)
    y_pred = y_proba_argmax
    y_pred=tf.cast(
        y_pred, dtype='uint8'
    )
    #self.y_pred=v_j
    out_caps =y_pred
    sprint('y_pred',y_pred)
    y = inputs['y']
    self.y_true=y
    sprint('y',y)
    T = tf.one_hot(y, depth = n_categories)
    sprint('T',T)

    if not inputs['bypass']:

      # Equation (4):

      v_j_norm = norm(v_j, axis = -1, keepdims = True)
      sprint('v_j_norm t:',v_j_norm)

      v_k_norm = tf.reshape(v_j_norm, shape = (-1, 10))

      sprint('v_k_norm:',v_k_norm)
      left_part = tf.square(tf.maximum(0., margin_plus - v_k_norm))
      right_part = tf.square(tf.maximum(0., v_k_norm - margin_minus))
      L_k = tf.add(T * left_part, llambda * (1.0 - T) * right_part)

      margin_loss1 = tf.reduce_mean(tf.reduce_sum(L_k, axis = 1))

      T_reshaped = tf.reshape(T, [-1, n_categories, 1]) #new
      sprint('T_reshaped',T_reshaped)

      sprint('v_j',v_j)
      v_j_masked = tf.multiply(v_j, T_reshaped)
    else:
      v_j = inputs['v']    
      v_j_masked = inputs['v']          # bypass capsule layer, feed input to decoder
    sprint('v_j_masked',v_j_masked)

    self.decoder_input = tf.reshape(v_j_masked, shape = [-1, 160])
    sprint('decoder_input',self.decoder_input)

    d1 = self.decoder1 (self.decoder_input)
    d2 = self.decoder2 (d1)
    decoder_out = self.decoder3 (d2)
    sprint('decoder_out',decoder_out)

    reconstructed_image = tf.reshape(decoder_out, (-1,28, 28,1))
    sprint('reconstructed image',reconstructed_image)

    if not inputs['bypass']:
      diff = original_image - reconstructed_image
      sprint('diff',diff)
      sq_diff = tf.square(diff)
      sprint('sq_diff',sq_diff)
      reconstruction_loss = tf.reduce_mean(sq_diff)

    # Final loss

      loss = tf.add(margin_loss1, alpha * reconstruction_loss)
      sprint('loss',loss)

    # Accuracy
      is_equal = tf.equal(y, y_pred)

      self.accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))
      accuracy = self.accuracy
    else:
      loss = None
      self.accuracy = None
      reconstruction_loss = None
    dec_verbosity()
    return loss, self.accuracy, reconstruction_loss, reconstructed_image, y, v_j, original_image
