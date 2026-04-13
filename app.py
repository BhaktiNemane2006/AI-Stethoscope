import streamlit as st
import numpy as np
from signal_processing import bandpass
from utils import save_recording
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import librosa
import os
import time

# -----------------------------
# Login System
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")
    st.stop()

# -----------------------------
# UI STYLE (ICU)
# -----------------------------
st.markdown("""
<style>
body {background-color:black; color:#00FF00;}
.stApp {background-color:black;}
</style>
""", unsafe_allow_html=True)

st.title("💓 ICU Heart Monitor")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload Heart Sound (.wav)", type=["wav"])

# -----------------------------
# Patient Info
# -----------------------------
st.sidebar.title("🧑‍⚕️ Patient Info")
patient_name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)

# -----------------------------
# Signal
# -----------------------------
def generate_heart_signal():
    t = np.linspace(0, 1, 300)
    return np.sin(2*np.pi*2*t) + np.random.normal(0,0.1,300)

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=1000)
    raw_data = y[:2000]
else:
    raw_data = generate_heart_signal()

filtered = bandpass(np.array(raw_data))
filtered = filtered / (np.max(np.abs(filtered)) + 1e-6)

# -----------------------------
# BPM
# -----------------------------
peaks, _ = find_peaks(filtered, distance=50, height=0.2)
bpm = len(peaks) * 60

# -----------------------------
# Improved AI (better logic)
# -----------------------------
variance = np.var(filtered)
energy = np.mean(np.abs(filtered))

confidence = min((energy + variance) * 3, 1.0)

prediction = "Normal" if confidence < 0.6 else "Abnormal"

# -----------------------------
# Dashboard
# -----------------------------
col1, col2 = st.columns(2)

col1.metric("❤️ BPM", bpm)

if prediction == "Normal":
    col2.success("🟢 Normal")
else:
    col2.error("🔴 Abnormal")

st.metric("🧠 Confidence", f"{confidence*100:.2f}%")

# -----------------------------
# Probability Graph
# -----------------------------
st.subheader("📊 AI Probability")
st.bar_chart({"Normal":1-confidence, "Abnormal":confidence})

# -----------------------------
# Doctor Recommendation
# -----------------------------
st.subheader("🧑‍⚕️ Recommendation")

if confidence < 0.6:
    st.success("Healthy heart. Maintain lifestyle.")
else:
    st.error("Consult cardiologist immediately.")

# -----------------------------
# Waveform Animation (ICU)
# -----------------------------
st.subheader("📈 Live Waveform")
placeholder = st.empty()

for _ in range(10):
    placeholder.line_chart(generate_heart_signal())
    time.sleep(0.1)

# -----------------------------
# Spectrogram
# -----------------------------
st.subheader("📊 Spectrogram")
fig, ax = plt.subplots()
ax.specgram(filtered, Fs=1000)
st.pyplot(fig)

# -----------------------------
# MULTIPLE PATIENT RECORDS
# -----------------------------
if "records" not in st.session_state:
    st.session_state.records = []

if st.button("💾 Save Patient Record"):
    st.session_state.records.append({
        "name": patient_name,
        "bpm": bpm,
        "status": prediction
    })

st.subheader("📁 Patient Records")

for r in st.session_state.records:
    st.write(r)

# -----------------------------
# PDF REPORT WITH GRAPH
# -----------------------------
def generate_pdf(name, bpm, status):
    filename = f"{name}_report.pdf"

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Patient: {name}", styles["Normal"]))
    content.append(Paragraph(f"BPM: {bpm}", styles["Normal"]))
    content.append(Paragraph(f"Condition: {status}", styles["Normal"]))

    # Save graph image
    img_path = "temp_plot.png"
    plt.figure()
    plt.plot(filtered)
    plt.savefig(img_path)
    plt.close()

    content.append(Image(img_path, width=300, height=150))

    doc.build(content)
    return filename

if st.button("📄 Generate Report"):
    file = generate_pdf(patient_name, bpm, prediction)
    with open(file, "rb") as f:
        st.download_button("⬇ Download Report", f, file_name=file)
    
