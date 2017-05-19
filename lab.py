#!/usr/bin/env python
import scipy.io.wavfile
import scipy.fftpack
from scipy.signal import butter, lfilter, decimate, fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import sys
import math

def plot_amplitude(sample_rate, data, save=False):
    n = len(data) # Sample length
    k = np.arange(n)
    T = n/sample_rate
    freq = k/T
    freq = freq[range(n/2)]

    Y = scipy.fftpack.fft(data)#/n
    Y = Y[range(n/2)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(freq, abs(Y))

    # Formatter for x ticks. All values will be in kHz
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('%.0f')%(y*1e-3)))
    ax.set_xlabel('Frekvens $f$  [kHz]')
    ax.set_ylabel('amplitud')

    if save:
        plt.savefig('amplitude.pdf')
    plt.show()



def plot_carrier_candidates(c_freqs, data, sample_rate, save=False):
    y = data
    fig = plt.figure()
    plot_num = 311
    t = np.linspace(0.0, len(y)/sample_rate, len(y))
    for i in range(len(c_freqs)):
        f = c_freqs[i] * 1000
        lowcut = f - 10000
        highcut = f + 10000
        
        y = butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=10)
        sub = fig.add_subplot(plot_num)
        sub.set_title('$f_{c' + str(i+1) + '} = ' + str(f) + '$ Hz')
        sub.set_xlabel('tid [s]')

        sub.plot(t, y)
        plot_num += 1
    fig.tight_layout()
    if save:
        plt.savefig('candidates.pdf')
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

def autocorrelate(y, sample_rate, marker=None, save=False):
    # plot autocorrelation of signal to find tau where the echo begins
    cor = fftconvolve(y, y[::-1], 'full')

    # Fix values for t axis
    t = len(y)/sample_rate
    t_axis = np.linspace(-t, t, len(cor))

    # Plot resuts
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('tid [s]')
    ax.set_ylabel('korrelation')
    ax.plot(t_axis, cor)
    
    if marker is not None:
        x, y = marker
        ax.annotate('$\\tau = $$' + str(x) + '$', xy=(x, y), xytext=(x*1.4, y*1.4), arrowprops=dict(facecolor='black', shrink=0.005))


    if save:
        plt.savefig('autocorrelation.pdf')
    plt.show()


def remove_echo(tau_idx, y):
    # Remove echo by subtracting the inital part of the signal, from t=0 to t=tau,
    # with the rest of the signal t=tau+1 to t =signal length
    for i in range(len(y)):
        if i > tau_idx: 
            y[i] = y[i] - 0.9*y[i-tau_idx]
    return y


if __name__ == "__main__":
    mpl.rcParams['agg.path.chunksize'] = 10000
    Fs, data = scipy.io.wavfile.read("signal-freis685.wav") # Sample rate and samples


    c_cand = [56, 94, 151] # fc candidates
    fc = 94000 # Carrier frequency

    plot_amplitude(Fs, data)
    plot_carrier_candidates(c_cand, data, Fs)
    
    num_samples = len(data) # Number of samples
    signal_time = num_samples / Fs # Signal length in time
    bandwidth = 20000 # Upper limit for audible frequencies
    nq = 0.5 * Fs
    lp_cutoff = bandwidth / (2*nq) # Low pass filter
    low = (fc - bandwidth / 2)
    high = (fc + bandwidth / 2)
    y = butter_bandpass_filter(data, low, high, Fs, 10)

    f1 = 141500
    f2 = f1 + 1
    t = np.linspace(0.0, signal_time, num_samples)
    w = 0.001  * ( np.cos(2 * math.pi * f1 * t) * np.cos(2 * math.pi * f2 * t))

    autocorrelate(y, Fs)


    tau = 0.43
    tau_idx = 172000
    y = remove_echo(tau_idx, y)

    # IQ demodulate the signal
    yI, yQ = IQ_demodulation(signal_time, num_samples, y, lp_cutoff)

    # Decimate the signal to lower frequency
    yI = decimate(yI, 10)
    yQ = decimate(yQ, 10)

    # Adjust volume
    yI = yI / np.max(np.abs(yI))
    yQ = yQ / np.max(np.abs(yQ))

    # Write to file
    scipy.io.wavfile.write('yi.wav', 40000, yI)
    scipy.io.wavfile.write('yq.wav', 40000, yQ)
    




