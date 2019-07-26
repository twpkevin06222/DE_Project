import os
import sys
import time
import glob
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.external import tifffile as sktiff
import skimage.color

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


def vid_2_frames(vid_path, output_path, extension='.jpg'):
    '''
    Converting video to image sequences with specified extension

    Params:
    vid_path: Path where video is stored
    output_path: Path where the converted image should be stored
    extension: Desired image extension, by DEFAULT .jpg

    Example:
        vid_path = '7-12-17-preprocessed.avi'
        output_path = retrieve_filename(vid_path)

        vid_2_frames(vid_path, output_path, extension = '.jpg')

    Return:
        >> For:  7-12-17-preprocessed.avi

        >> Creating..../7-12-17-preprocessed/frame_0000.jpg
        >> Creating..../7-12-17-preprocessed/frame_0001.jpg
                ...
    '''
    # Read the video from specified path
    cam = cv2.VideoCapture(vid_path)

    try:

        # creating a folder named output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of output path')

        # frame
    currentframe = 0

    print('For: ', vid_path)
    print()

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            # name = ('./'+ output_path +'/frame_' + str(currentframe) + extension)
            name = ('./{}/frame_{:04d}{}').format(output_path, currentframe, extension)
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def retrieve_filename(file_path):
    '''
    Retrieve file name from path and remove file extension

    Example:
        file_path = 'home/user/Desktop/test.txt'
        retrieve_filename(file_path)
    Return:
        >> test
    '''
    base_name = os.path.basename(file_path)

    # extract base name without extension
    base_name = os.path.splitext(base_name)[0]

    # print(base_name)

    return base_name


def img_to_array(inp_img, RGB=True):
    '''
    Convert single image from RGB or from Grayscale to array

    Params:
    inp_img: Desire image to convert to array
    RGB: Convert RGB image to grayscale if FALSE
    '''
    if RGB:
        return skimage.io.imread(inp_img)
    else:
        img = skimage.io.imread(inp_img)
        grayscale = skimage.color.rgb2gray(img)

        return grayscale


def imgs_to_arrays(inp_imgs, extension='.jpg', RGB=True, save_as_npy=False, img_resize = None, save_path=None):

    '''
    Convert image stacks from RGB or from Grayscale to array

    Params:
    inp_imgs: Desire image stacks to convert to array
    extension: input images extension, by DEFAULT '.jpg'
    RGB: Convert RGB image to grayscale if FALSE
    save_as_npy: Save as .npy extension
    save_path: Specify save path
    '''
    if img_resize != None:
        IMG_SIZE = img_resize

    imgs_list = []
    for imgs in sorted(glob.glob('{}/*{}'.format(inp_imgs, extension))):
        img_array = img_to_array(imgs, RGB)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        imgs_list.append(img_array)

    imgs_list = np.asarray(imgs_list)

    if save_as_npy:
        assert save_path != None, "Save path not specified!"
        # by default
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_name = retrieve_filename(inp_imgs)
        np.save(save_path + '/{}.npy'.format(save_name), imgs_list)

    return imgs_list

