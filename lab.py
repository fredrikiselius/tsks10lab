import scipy.io.wavfile
import scipy.fftpack
from scipy.signal import butter, lfilter, decimate, fftconvolve
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

def butter_lowpass_filter(cutoff, data):
    b, a = butter(10, cutoff, btype='low')
    return lfilter(b, a, data)

def IQ_demodulation(signal_time, num_samples, y, lp_cutoff):
    t = np.linspace(0.0, signal_time, num_samples)
    yI = y * 2 *np.cos(2 * math.pi *fc *t)
    yQ = y * -2 *np.sin(2 * math.pi *fc *t)
    return butter_lowpass_filter(lp_cutoff, yI), butter_lowpass_filter(lp_cutoff, yQ)

        

if __name__ == "__main__":
    mpl.rcParams['agg.path.chunksize'] = 10000
    Fs, data = scipy.io.wavfile.read("signal-freis685.wav") # Sample rate and samples


    #c_cand = [56, 94, 151] # 94
    #plot_amplitude(Fs, data)
    #plot_carrier_candidates(c_cand, data, Fs)

    fc = 94000
    num_samples = len(data)
    signal_time = num_samples / Fs
    bandwidth = 20000
    nq = 0.5 * Fs
    lp_cutoff = bandwidth / (2*nq)
    low = (fc - bandwidth / 2)
    high = (fc + bandwidth / 2)
    y = butter_bandpass_filter(data, low, high, Fs, 10)

    f1 = 141500
    f2 = f1 + 1
    t = np.linspace(0.0, signal_time, num_samples)
    w = 0.001  * ( np.cos(2 * math.pi * f1 * t) * np.cos(2 * math.pi * f2 * t))


    
    corr = fftconvolve(y, y[::-1], 'full')
    #plt.plot(corr[len(corr)/2:])
    #plt.show()

    tau = 0.43
    tau_idx = 172000



    for i in range(len(y)):
        if i > tau_idx: 
            y[i] = y[i] - 0.9*y[i-tau_idx]


    yI, yQ = IQ_demodulation(signal_time, num_samples, y, lp_cutoff)

    print yI == yQ
    yI = decimate(yI, 10)
    yQ = decimate(yQ, 10)

    yI = yI / np.max(np.abs(yI))
    yQ = yQ / np.max(np.abs(yQ))

    scipy.io.wavfile.write('yi.wav', 40000, yI)
    scipy.io.wavfile.write('yq.wav', 40000, yQ)
    




