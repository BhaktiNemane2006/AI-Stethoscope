import librosa
import numpy as np

def extract_spectrogram(file):
    y, sr = librosa.load(file, sr=1000)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    return S_db