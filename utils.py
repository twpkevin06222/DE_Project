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
    '''
    Read .tif extension

    :param dir_path: directory path where data is stored
    :return:
        shape of the particular tif file, arrays of the tif file
    '''
    im = sktiff.imread(dir_path)
    return im.shape, im

def append_tiff(path, verbose = True, timer = False):
    '''
    Append tiff image from path

    :param path: data directory
    :param verbose: output directory info
    :param timer: time measurement
    :return:
        list of tiff images, list of directories of tiff images
    '''
    start = time.time()

    dir_list = []
    image_stack = []
    for main_dir in sorted(os.listdir(path)):
        if verbose:
            print('Directory of mice index:', main_dir)
            print('Directory of .tif files stored:')

        merge_dir = os.path.join(path + main_dir)

        for file in sorted(os.listdir(merge_dir)):
            tif = glob.glob('{}/*.tif'.format(os.path.join(merge_dir + '/' + file)))

            shape, im = tiff(tif)
            dir_list.append(main_dir + '/' + file)
            image_stack.append(im)

            if verbose:
                print('{}, {}'.format(tif, shape))

    images = np.asarray(image_stack)
    end = time.time()

    if timer == True:
        print('Total time elapsed: ', end - start)

    return images, dir_list


def mat_2_npy(input_path, save_path):
    '''
    convert arrays in .mat to numpy array .npy

    input_path: path where data files of LIN is store, no need on specific path of .mat!
    save_path: where .npy is save
    '''
    for main_dir in sorted(os.listdir(input_path)):
        print('Directory of mice index:', main_dir)
        merge_dir = os.path.join(input_path + main_dir)

        print('Directory of .mat files stored:')
        print()
        for file in sorted(os.listdir(merge_dir)):
            mat_list = glob.glob('{}/*.mat'.format(os.path.join(merge_dir + '/' + file)))
            for mat in mat_list:

                print(mat)
                # obtain file name .mat for new file name during the conversion
                mat_dir_split = mat.split(os.sep)
                mat_name = mat_dir_split[-1]
                # print(mat_name)

                # returns dict
                data = scipy.io.loadmat(mat)
                for i in data:
                    if '__' not in i and 'readme' not in i:
                        print(data[i].shape)

                        # save matlab arrays into .npy file
                        np.save(save_path + "{}_{}.npy".format(mat_name, i), data[i])

    print()
