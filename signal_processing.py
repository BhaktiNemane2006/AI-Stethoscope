import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(signal, fs=1000):
    low = 20/(fs/2)
    high = 150/(fs/2)
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)