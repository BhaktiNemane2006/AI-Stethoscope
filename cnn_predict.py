import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("models/cnn_model.h5")

def preprocess_signal(signal):
    y = librosa.util.fix_length(signal, size=4000)

    S = librosa.feature.melspectrogram(
        y=y, sr=1000, n_fft=512, hop_length=128
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    return S_db[..., np.newaxis]


def predict_cnn(signal):
    spec = preprocess_signal(signal)
    spec = np.expand_dims(spec, axis=0)

    pred = model.predict(spec)[0][0]

    return pred