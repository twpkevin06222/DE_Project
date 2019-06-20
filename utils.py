import os
import sys
import time
import glob
from PIL import Image
import numpy as np
#import timesynth as ts
import matplotlib.pyplot as plt
from skimage.external import tifffile as sktiff

def time_series_generator(stop_time, num_points, frequency, noise = None, std = 0.1):
    '''
    Generate a time series based on parameters
    :param stop_time:
    :param num_points:
    :param frequency:
    :param noise:
    :param std:

    :return: samples, signals, errors
    '''
    # Initializing TimeSampler
    time_sampler = ts.TimeSampler(stop_time=stop_time)
    # Sampling irregular time samples
    irregular_time_samples = time_sampler.sample_irregular_time(num_points= num_points, keep_percentage=10)
    # Initializing Sinusoidal signal
    #sinusoid = ts.signals.Sinusoidal(frequency=0.5)
    PseudoPeriodic = ts.signals.PseudoPeriodic(frequency= frequency)
    if noise == None:
        white_noise = None
    else:
    # Initializing Gaussian noise
        white_noise = ts.noise.GaussianNoise(std=std)
    # Initializing TimeSeries class with the signal and noise objects
    timeseries = ts.TimeSeries(PseudoPeriodic, noise_generator=white_noise)
    # Sampling using the irregular time samples
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    return samples, signals, errors

#Example:
# _, signals,_ = time_series_generator(stop_time = 20, num_points = 500, frequency = 0.1, noise = True, std = 0.3)
# plt.plot(signals)
# plt.show()

def min_max_norm(images):
    """
    Min max normalization of images
    Parameters:
        images: Input stacked image list
    Return:
        Image list after min max normalization
    """
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/ (m - mi)
    return images


def resize(img_list, NEW_SIZE):
    """
    Resize image
    Parameter:
        image volume height list, new size for image
    Return:
        resize image list
    """
    new_img_list = []

    for img in img_list:
        new_img = []

        for i in range(np.size(img, 2)):
            new_img.append(np.asarray(Image.fromarray(img[:,:,i],mode='F').resize((NEW_SIZE,NEW_SIZE), Image.LANCZOS)))

        new_img_list.append(np.swapaxes(new_img, 0, 2))

    return new_img_list

def tiff(dir_path):
    im = sktiff.imread(dir_path)
    return im.shape, im

def append_tiff(path, verbose = True):
    '''
    Append tiff image from path
    :param path: data directory
    :param verbose: output directory info
    :return:
        list of tiff images, list of directories of tiff images
    '''
    tiff_list = []
    dir_list = []
    for main_dir in sorted(os.listdir(path)):
        if verbose:
            print('Directory of mice index:', main_dir)
            print('Directory of .tif files stored:')

        merge_dir = os.path.join(path + main_dir)

        for file in sorted(os.listdir(merge_dir)):
            tif = glob.glob('{}/*.tif'.format(os.path.join(merge_dir + '/' + file)))

            if verbose:
                print(tif)

            shape, im = tiff(tif)
            tiff_list.append(im)
            dir_list.append(main_dir + '/' + file)

    return(tiff_list, dir_list)
