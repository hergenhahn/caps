#
# Auxiliary functions
#
import numpy as np
import tensorflow as tf
#
# print shape, dtype and number of nonzero elements.
# while most tensors are way to large to print them, the number of
# nonzero elements may provide valuable hints for debugging.
#
verbose=0 #controls printing

def sprint(vn,v):
  if verbose>0:
    try:
      print(vn,':', v.shape,v.dtype,' nz:',tf.math.count_nonzero(v).numpy())
    except:
      try:
        print(vn,':', v.shape,v.dtype)
      except:
        print(vn,': !!!!!!!')

def set_verbosity(n):
    global verbose
    verbose=n

def dec_verbosity():
    global verbose
    verbose=verbose-1

def loadMNIST(batch_size):
#
# Load the MNIST dataset, convert to float
#
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape, x_test.shape, x_test.min(),"..",x_test.max(),y_train.shape,y_test.shape)

    x_train = np.array(x_train, dtype=np.float32)
    x_train *= (1.0/255.0)
    y_train = tf.convert_to_tensor(y_train)
    x_test = np.array(x_test, dtype=np.float32)
    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))
    x_test *= (1.0/255.0)


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# make batches:
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    return train_dataset, test_dataset, x_train.shape[0], x_test.shape[0]
