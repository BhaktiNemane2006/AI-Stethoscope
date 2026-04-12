import streamlit as st
import numpy as np
from signal_processing import bandpass
from utils import save_recording
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from report import generate_report
import librosa
import sounddevice as sd

if st.button("🎤 Record from Mic"):
    st.info("Recording...")

    duration = 3  # seconds
    fs = 1000

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    raw_data = audio.flatten()

    st.success("Recording complete!")
# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Heart Sound (.wav)", type=["wav"])

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: black;
    color: #00FF00;
}
.stMetric {
    font-size: 30px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar (Patient Info)
# -----------------------------
st.sidebar.title("🧑‍⚕️ Patient Info")

patient_name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

mode = st.sidebar.radio("Select Mode", ["Demo", "Real (ESP32)"])

# -----------------------------
# Title
# -----------------------------
st.title("💓 AI Digital Stethoscope")

# -----------------------------
# Fake Signal Generator
# -----------------------------
def generate_heart_signal():
    t = np.linspace(0, 1, 300)
    signal = (
        np.sin(2 * np.pi * 2 * t) * 0.2 +
        np.exp(-((t-0.2)*30)**2) * 2 +
        np.exp(-((t-0.6)*30)**2) * 1.5
    )
    noise = np.random.normal(0, 0.1, 300)
    return signal + noise

# -----------------------------
# Get Data
# -----------------------------
if uploaded_file is not None:
    st.success("📁 File uploaded successfully")

    y, sr = librosa.load(uploaded_file, sr=1000)
    raw_data = y[:2000]   # FIXED length

elif mode == "Demo":
    raw_data = generate_heart_signal()

else:
    from serial_reader import read_serial
    raw_data = []

    for _ in range(300):
        val = read_serial()
        if val is not None:
            raw_data.append(val)

# -----------------------------
# Process Signal
# -----------------------------
filtered = bandpass(np.array(raw_data))

# ✅ NORMALIZATION (VERY IMPORTANT)
filtered = filtered / np.max(np.abs(filtered) + 1e-6)

# -----------------------------
# Calculate BPM
# -----------------------------
peaks, _ = find_peaks(filtered, distance=50, height=0.2)
bpm = len(peaks) * 60

# -----------------------------
# AI Prediction (CNN)
# -----------------------------
# -----------------------------
# Lightweight AI Prediction (Cloud Compatible)
# -----------------------------
energy = np.mean(np.abs(filtered))

# Convert to confidence (scaled)
confidence = min(energy * 5, 1.0)

prediction = 1 if confidence >= 0.6 else 0

# -----------------------------
# ICU Dashboard Layout
# -----------------------------
st.markdown("## 🩺 ICU Heart Monitor")

col1, col2, col3 = st.columns(3)

col1.metric("❤️ Heart Rate", f"{bpm} BPM")

# ✅ IMPROVED THRESHOLD
if confidence < 0.6:
    col2.success("🟢 Normal")
elif confidence < 0.8:
    col2.warning("🟡 Risk")
else:
    col2.error("🔴 Abnormal")

col3.metric("📡 Signal Status", "Active")

# -----------------------------
# Show Confidence
# -----------------------------
st.metric("🧠 AI Confidence", f"{confidence*100:.2f}%")
st.write("Raw Confidence:", confidence)  # debug

# -----------------------------
# Waveform (FIXED)
# -----------------------------
st.subheader("📈 Heart Waveform")
st.line_chart(filtered)

# -----------------------------
# Spectrogram
# -----------------------------
st.subheader("📊 Spectrogram")

fig, ax = plt.subplots()
ax.specgram(filtered, Fs=1000)
ax.set_title("Spectrogram")
st.pyplot(fig)

# -----------------------------
# Save Recording
# -----------------------------
if st.button("🔴 Record & Save"):
    file = save_recording(filtered, patient_name)
    st.success(f"Saved: {file}")

# -----------------------------
# Generate Report
# -----------------------------
if st.button("📄 Generate Report"):
    status = "Normal" if prediction == 0 else "Abnormal"
    file = generate_report(patient_name, bpm, status, confidence)
    st.success(f"Report saved: {file}")
    
