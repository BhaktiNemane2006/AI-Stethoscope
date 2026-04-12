import librosa
import numpy as np

def get_spectrogram(signal, fs=1000):
    S = librosa.feature.melspectrogram(y=signal.astype(float), sr=fs)
    return librosa.power_to_db(S, ref=np.max)