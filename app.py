import streamlit as st
import numpy as np
import sqlite3
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from signal_processing import bandpass
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import time
from streamlit_webrtc import webrtc_streamer

# -----------------------------
# AUDIO BUFFER (FIXED)
# -----------------------------
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

def audio_callback(frame):
    audio = frame.to_ndarray().flatten()
    st.session_state.audio_buffer.extend(audio.tolist())

    # keep only latest data (avoid overflow)
    st.session_state.audio_buffer = st.session_state.audio_buffer[-2000:]
    return frame

st.subheader("🎤 Live Mic (Browser)")
webrtc_streamer(
    key="mic",
    audio_frame_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
)

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT,
    role TEXT
)
""")

c.execute("INSERT OR IGNORE INTO users VALUES ('admin','1234','admin')")
c.execute("INSERT OR IGNORE INTO users VALUES ('doctor','1234','doctor')")
conn.commit()

# -----------------------------
# LOGIN
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = ""

if not st.session_state.logged_in:
    st.title("🔐 Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()

        if user:
            st.session_state.logged_in = True
            st.session_state.role = user[2]
        else:
            st.error("Invalid credentials")

    st.stop()

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
body {background-color:#000000; color:#00FFAA;}
.stApp {background-color:#000000;}
h1,h2,h3 {color:#00FFAA; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("💓 AI ICU Heart Monitoring System")

# -----------------------------
# INPUT SOURCE
# -----------------------------
uploaded_file = st.file_uploader("Upload Heart Sound (.wav)", type=["wav"])

# 🔥 PRIORITY: MIC > FILE > DEMO
if len(st.session_state.audio_buffer) > 500:
    raw_data = np.array(st.session_state.audio_buffer)

elif uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=1000)
    raw_data = y[:2000]

else:
    t = np.linspace(0,1,300)
    raw_data = np.sin(2*np.pi*2*t)

# -----------------------------
# PROCESS SIGNAL
# -----------------------------
filtered = bandpass(np.array(raw_data))
filtered = filtered / (np.max(np.abs(filtered)) + 1e-6)

# -----------------------------
# BPM (REAL-TIME)
# -----------------------------
peaks, _ = find_peaks(filtered, distance=50, height=0.2)
bpm = len(peaks) * 60

# -----------------------------
# AI (REAL-TIME)
# -----------------------------
energy = np.mean(np.abs(filtered))
variance = np.var(filtered)

confidence = min((energy + variance) * 3, 1.0)

if bpm < 60 or bpm > 120:
    status = "Abnormal"
elif confidence > 0.7:
    status = "Abnormal"
else:
    status = "Normal"

# -----------------------------
# DASHBOARD
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("❤️ BPM", bpm)
col2.metric("🧠 Confidence", f"{confidence*100:.1f}%")
col3.metric("📡 Status", status)

if bpm > 150:
    st.error("🚨 CRITICAL ALERT!")

# -----------------------------
# WAVEFORM (REAL MIC)
# -----------------------------
st.subheader("📈 Live Waveform")
st.line_chart(filtered)

# -----------------------------
# HISTORY
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append(bpm)
st.session_state.history = st.session_state.history[-10:]

st.subheader("📊 BPM History")
st.line_chart(st.session_state.history)

# -----------------------------
# AI GRAPH
# -----------------------------
st.subheader("📊 AI Probability")
st.bar_chart({"Normal":1-confidence, "Abnormal":confidence})

# -----------------------------
# RECOMMENDATION
# -----------------------------
st.subheader("🧑‍⚕️ Recommendation")

if status == "Normal":
    st.success("Healthy heart")
else:
    st.error("Consult doctor")

# -----------------------------
# SPECTROGRAM
# -----------------------------
fig, ax = plt.subplots()
ax.specgram(filtered, Fs=1000)
st.pyplot(fig)
    
