import scipy.io.wavfile
import scipy.fftpack
from scipy.signal import butter, lfilter, decimate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import math

def plot_amplitude(sample_rate, data):
    """
    calculates the carrier frequency
    """
    n = len(data) # Sample length
    k = np.arange(n)
    T = n/sample_rate
    freq = k/T
    freq = freq[range(n/2)]

    Y = scipy.fftpack.fft(data)/n
    Y = Y[range(n/2)]

    plt.plot(freq, abs(Y))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_carrier_candidates(c_freqs, data, sample_rate):
    fig = plt.figure()
    plot_num = 311
    for f in c_freqs:
        f = f * 1000
        lowcut = f - 10000
        highcut = f + 10000
        
        y = butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=10)
        sub = fig.add_subplot(plot_num)
        sub.plot(y)
        plot_num += 1
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def IQ_demodulation(carrier_freq, signal_time, num_samples, data):
    #lowcut = fc - 10000
    #highcut = fc + 10000
    #y = butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=10)
    t = np.linspace(0.0, signal_time, num_samples)
    yI = data * 2 *np.cos(2 * math.pi *fc *t)
    yQ = data * -2 *np.cos(2 * math.pi *fc *t)
    return yI, yQ

def write_
        

if __name__ == "__main__":
    mpl.rcParams['agg.path.chunksize'] = 10000
    sample_rate, data = scipy.io.wavfile.read("signal-freis685.wav")
    num_samples = len(data)
    signal_time = num_samples / sample_rate
    bandwidth = sample_rate / 2
    if len(sys.argv) == 1:
        plot_amplitude(sample_rate, data)
        c_freqs = [56, 94, 151] # kHz
        plot_carrier_candidates(c_freqs, data, sample_rate)
    else:
        fc = 56000
        I, Q = IQ_demodulation(fc, signal_time, num_samples, data)



