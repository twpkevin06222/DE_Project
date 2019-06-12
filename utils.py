import os
import sys
import time
import numpy as np
import timesynth as ts
import matplotlib.pyplot as plt

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