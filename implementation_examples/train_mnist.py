import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization, ReLU, LeakyReLU
import utils
from coord_conv import AddCoords, CoordConv

#Parameters
BATCH_SIZE = 256
MAX_STEP = 5000
lr = 1e-4
f_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = f_mnist.load_data()

def norm_cast(inp):
    '''
    Normalizing and casting data to float32
    '''
    return tf.expand_dims(tf.cast((inp/255.0), dtype=tf.float32), -1)

x_train, x_test = norm_cast(x_train), norm_cast(x_test)

print('Shape of data: {}, Data type: {}'.format(x_train.shape, x_train.dtype))

f_mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).repeat(10).batch(BATCH_SIZE)
f_mnist_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#Model
class model_v01(Model):
    """
    Model with normal Conv
    """
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = Conv2D(128, 3, padding='same', activation='relu')
        self.maxpooling = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, 3, padding='same', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, inp):
        x = self.conv1(inp)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x

class model_v02(Model):
    """
    Model with normal Conv
    """
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = CoordConv(x_dim = 28, y_dim = 28, with_r = False, filters = 128, kernel_size = 3, padding='same', activation='relu')
        self.maxpooling = MaxPooling2D((2, 2))
        self.conv2 = CoordConv(x_dim = 14, y_dim = 14, with_r = False, filters = 64, kernel_size = 3, padding='same', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, inp):
        x = self.conv1(inp)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x

#model = model_v01()
model = model_v02()

opt = tf.keras.optimizers.Adam(lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_fn(imgs, lbls):
    with tf.GradientTape() as tape:
        output = model(imgs)
        loss = loss_fn(lbls, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(lbls, output)


for steps, (img_batch, lbl_batch) in enumerate(f_mnist_train):
    if steps > MAX_STEP:
        break

    train_fn(img_batch, lbl_batch)

    Template = 'Step: {}, Loss: {}, Accuracy: {}'

    if not steps % 100:
        print(Template.format(steps, train_loss.result(), train_acc.result()))