import tensorflow as tf
from coord_conv import CoordConv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import InputLayer, Conv2DTranspose, Activation, BatchNormalization
from tensorflow.keras.regularizers import l1

def conv_block(x_in, filters, kernel_size, strides, padding,
               activation, kernel_regularizer=False,
               activity_regularizer = False,
               batch_norm=False, max_pool=False, l1_coeff=None):
    '''
    Build convolutional block with batch normalization
    '''
    if kernel_regularizer:
        print('L1 kernel regularizer is activate!')
        x = Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=l1(l1_coeff))(x_in)
    else:
        x = Conv2D(filters, kernel_size, strides, padding)(x_in)

    if activity_regularizer:
        print('L1 activity regularizer is activate!')
        x = Conv2D(filters, kernel_size, strides, padding, activity_regularizer=l1(l1_coeff))(x_in)
    else:
        x = Conv2D(filters, kernel_size, strides, padding)(x_in)

    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    if max_pool:
        assert strides < 2 or strides < (2, 2), "Downsampling too fast for strides greater than 2"

        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    return x


def coordconv_block(x_in, x_dim, y_dim, filters, kernel_size,
                    strides, padding, activation, kernel_regularizer=False,
                    activity_regularizer=False,
                    batch_norm=False, max_pool=False,
                    with_r=False, l1_coeff=None):
    '''
    Build coordconv block with batch normalization
    '''
    if kernel_regularizer:
        print('L1 kernel regularizer is activate!')
        x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size,
                      strides, padding, kernel_regularizer=l1(l1_coeff))(x_in)
    else:
        x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size, strides, padding)(x_in)

    if activity_regularizer:
        print('L1 kernel regularizer is activate!')
        x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size,
                      strides, padding, activity_regularizer=l1(l1_coeff))(x_in)
    else:
        x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size, strides, padding)(x_in)

    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    if max_pool:
        assert strides < 2 or strides < (2, 2), "Downsampling too fast for strides greater than 2"

        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    return x

def up_block(x_in, up_size, filters, kernel_size, strides, padding, activation,
             batch_norm = False):
    '''
    Build upsampling block with upsamping + convolutional operation
    '''
    u = UpSampling2D(up_size)(x_in)
    #by default during upsampling Conv2D does not need maxpooling!
    conv_u = conv_block(u, filters, kernel_size, strides, padding, activation, batch_norm)
    return conv_u

def up_coord_block(x_in, up_size, xdim, ydim, filters, kernel_size, strides, padding, activation,
             batch_norm = False):
    '''
    Build upsampling block with upsamping + coordconv operation
    '''
    u = UpSampling2D(up_size)(x_in)
    #by default during upsampling Conv2D does not need maxpooling!
    coordconv_u = coordconv_block(u, xdim, ydim, filters, kernel_size, strides, padding, activation, batch_norm)
    return coordconv_u

def data_aug(x_train, y_train, batch_size):
    '''
    Generate data augmentation with shifting and rotation
    '''
    data_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(x_train, x_train, batch_size, seed=42)
    mask_generator = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10).flow(y_train, y_train, batch_size, seed=42)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch

def img_mean(imgs, img_size):
    '''
    Modified mean images for tensorflow
    '''
    sums = tf.zeros((img_size, img_size))
    total_index = 0
    for i in range(imgs.shape[0]):
        sums+=tf.squeeze(imgs[i])
        total_index+=1
    #print(total_index)
    mean_img = sums/total_index
    return tf.expand_dims(mean_img, axis = -1)

def min_max_norm(images):
    """
    Modified Min max normalization of images in Tensorflow
    Parameters:
        images: Input stacked image list
    Return:
        Image list after min max normalization
    """
    m = tf.math.reduce_max(images)
    mi = tf.math.reduce_min(images)
    images = (images - mi)/ (m - mi)
    return tf.expand_dims(images, axis = 0)

def dice_coef(y_true, y_pred):
    '''
    Dice coefficient for tensorflow
    '''
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / \
(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())

def dice_coef_loss(y_true, y_pred):
    '''
    Dice coefficient loss for IOU
    '''
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    """
    intersection = tf.reduce_sum(tf.math.abs(y_true * y_pred), axis=-1)
    sum_ = tf.reduce_sum(tf.math.abs(y_true) + tf.math.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return tf.reduce_sum((1 - jac) * smooth)


def max_intensity_projection(input_layer, BATCH_SIZE):
    '''
    MIP for tensorflow implementation with MaxPooling3D
    :param input_layer: Input layer with tensorflow.keras.layers.Input
    :param BATCH_SIZE: batch size
    :return:
        Maximum intensity projection of the batch with dim 5
    '''
    x = tf.expand_dims(input_layer, -1)
    x = tf.transpose(x, [3, 1, 2, 0, 4])  # swap (batch, dim1, dim2, dim3, channel) => (d3, d1, d2, batch, channel)
    x = MaxPooling3D(pool_size=(1, 1, BATCH_SIZE), strides=1)(x)

    return x
# input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
# mip = Model(input_layer, max_intensity_projection(input_layer, BATCH_SIZE))

def dicesq(y_true, y_pred):
    '''
    Modified dice coefficient as refer to: https://arxiv.org/abs/1606.04797
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: Modified dice coefficient
    '''
    nmr = 2*tf.reduce_sum(y_true*y_pred)
    dnmr = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2) + tf.keras.backend.epsilon()
    return (nmr / dnmr)

def dicesq_loss(y_true, y_pred):
    '''
    Modified dice coefficient loss
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    '''
    return 1- dicesq(y_true, y_pred)


def getConfusionMatrix(mask_truth, mask_predicted):

    #Create masks

    mask_truth = mask_truth > 0
    mask_truth = np.multiply(mask_truth ,1)
    mask_predicted = mask_predicted > 0 
    mask_predicted = np.multiply(mask_predicted ,1)

    #True Positives: Predicted correctly as Neuron 
    overlap = np.multiply(mask_truth ,mask_predicted)
    unique, counts = np.unique(overlap, return_counts=True)
    TP = counts[1] / overlap.size

    # False Positives: Predicted as neuron, but is no neuron
    FPmask = np.subtract(mask_truth, mask_predicted)
    FP = np.count_nonzero(FPmask == -1)
    FP = FP / FPmask.size

    # False Negatives: Is Neuron but was not predicted
    FNmask = np.subtract(mask_predicted, mask_truth)
    FN = np.count_nonzero(FNmask == -1)
    FN = FN / FNmask.size
    
    #True Nagatives: Is correctly recognized as no neuron
    TN = 1 - TP - FP - FN

    return TP, FP, FN, TN


    def f1score(TP, FP, FN): 
    f1 = 2*TP / (2*TP + FP + FN)
    return f1