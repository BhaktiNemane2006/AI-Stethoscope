import os
import numpy as np
import librosa
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_spectrogram(file):
    y, sr = librosa.load(file, sr=1000)

    # Fix length
    y = librosa.util.fix_length(y, size=4000)

    # Spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=128
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    return S_db


# -----------------------------
# Load Dataset
# -----------------------------
X = []
y = []

for label, folder in enumerate(["normal", "abnormal"]):
    path = os.path.join("dataset", folder)

    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
        continue

    for file in os.listdir(path):
        if file.endswith(".wav"):
            file_path = os.path.join(path, file)

            try:
                spec = extract_spectrogram(file_path)
                X.append(spec)
                y.append(label)
            except Exception as e:
                print("Error:", file, e)

# -----------------------------
# Convert to NumPy
# -----------------------------
X = np.array(X)
X = X[..., np.newaxis]   # Add channel dimension
y = np.array(y)

print("Dataset shape:", X.shape)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# -----------------------------
# CNN Model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=X.shape[1:]),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
X = X / np.max(X)
# -----------------------------
# Train
# -----------------------------
model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# -----------------------------
# Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.h5")

print("✅ CNN model saved!")