import timesynth as ts
import matplotlib.pyplot as plt

# Initializing TimeSampler
time_sampler = ts.TimeSampler(stop_time=20)
# Sampling irregular time samples
irregular_time_samples = time_sampler.sample_irregular_time(num_points=500, keep_percentage=10)
# Initializing Sinusoidal signal
#sinusoid = ts.signals.Sinusoidal(frequency=0.5)
PseudoPeriodic = ts.signals.PseudoPeriodic(frequency=0.1)
# Initializing Gaussian noise
white_noise = ts.noise.GaussianNoise(std=0.3)
# Initializing TimeSeries class with the signal and noise objects
timeseries = ts.TimeSeries(PseudoPeriodic, noise_generator=white_noise)
# Sampling using the irregular time samples
samples, signals, errors = timeseries.sample(irregular_time_samples)


plt.plot(signals)
plt.show()
