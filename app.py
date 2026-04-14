import streamlit as st
import numpy as np
import sqlite3
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import csv
import os
import pandas as pd
import time
import soundfile as sf
st.set_page_config(layout="wide")

# -----------------------------
# AUDIO BUFFER
# -----------------------------
st.subheader("🎤 Live Mic (Browser)")

if "audio_data" not in st.session_state:
    st.session_state.audio_data = np.array([])

def audio_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray().flatten().astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    st.session_state.audio_data = np.concatenate(
        (st.session_state.audio_data, audio)
    )
    st.session_state.audio_data = st.session_state.audio_data[-2000:]
    return frame

webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
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
conn.commit()

# -----------------------------
# LOGIN
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        if c.fetchone():
            st.session_state.logged_in = True
        else:
            st.error("Invalid")

    st.stop()

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
body {background:#000; color:#00FFAA;}
h1,h2,h3 {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.title("💓 AI Heart Monitoring System")

# -----------------------------
# PATIENT INFO
# -----------------------------
st.sidebar.header("🧑‍⚕️ Patient Info")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120)
gender = st.sidebar.selectbox("Gender", ["Male","Female","Other"])

# -----------------------------
# INPUT
# -----------------------------
uploaded_file = st.file_uploader("Upload (.wav)", type=["wav"])
# -----------------------------
# PLAY UPLOADED AUDIO
# -----------------------------
if uploaded_file is not None:
    st.subheader("🎧 Uploaded Audio Playback")
    st.audio(uploaded_file, format="audio/wav")
if len(st.session_state.audio_data) > 500:
    raw = st.session_state.audio_data
elif uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=1000)
    raw = y[:2000]
else:
    t = np.linspace(0,1,300)
    raw = np.sin(2*np.pi*2*t)

# -----------------------------
# FILTER
# -----------------------------
def heart_filter(sig):
    b,a = butter(3,[20/500,150/500],btype='band')
    return filtfilt(b,a,sig)

filtered = heart_filter(raw)
filtered = filtered / (np.max(np.abs(filtered))+1e-6)

# -----------------------------
# BPM
# -----------------------------
peaks,_ = find_peaks(filtered, distance=80, prominence=0.3)

if len(peaks)>1:
    bpm = int(60*1000/np.mean(np.diff(peaks)))
else:
    bpm = 0

# -----------------------------
# HEART BEAT MARKERS (FIXED)
# -----------------------------
st.subheader("🫀 Heart Beat Markers")

fig, ax = plt.subplots()
ax.plot(filtered)

for p in peaks[:10]:
    ax.axvline(p, color='red')

st.pyplot(fig)

# -----------------------------
# ARRHYTHMIA
# -----------------------------
arrhythmia = False

if len(peaks) > 2:
    rr_intervals = np.diff(peaks)
    variability = np.std(rr_intervals)

    if variability > 50:
        arrhythmia = True

if arrhythmia:
    st.warning("⚠ Irregular Heart Rhythm Detected")

# -----------------------------
# FEATURES
# -----------------------------
variance = np.var(filtered)
noise = np.std(filtered)

# -----------------------------
# SIGNAL QUALITY
# -----------------------------
st.subheader("📡 Signal Quality")

if noise < 0.1:
    st.success("Excellent signal quality")
elif noise < 0.3:
    st.warning("Moderate noise")
else:
    st.error("Poor signal")

# -----------------------------
# CLASSIFICATION
# -----------------------------
if bpm < 50 or bpm > 130:
    label = "Extrastole"
elif variance > 0.5:
    label = "Murmur"
else:
    label = "Normal"

# -----------------------------
# STATUS
# -----------------------------
if label=="Normal":
    status="🟢 Normal"
elif label=="Murmur":
    status="🟡 Murmur"
else:
    status="🔴 Abnormal"

# -----------------------------
# DOCTOR RECOMMENDATION
# -----------------------------
st.subheader("🧑‍⚕️ Doctor Recommendation")

if label == "Normal":
    st.success("Healthy heart. Maintain lifestyle.")
elif label == "Murmur":
    st.warning("Possible murmur. Get echo test.")
else:
    st.error("Consult cardiologist immediately.")

if arrhythmia:
    st.error("Possible Arrhythmia detected.")

# -----------------------------
# DASHBOARD
# -----------------------------
c1,c2,c3 = st.columns(3)

c1.metric("❤️ BPM", bpm)
c2.metric("Noise", f"{noise:.2f}")
c3.metric("Condition", label)

# -----------------------------
# SAVE HISTORY
# -----------------------------
def save_history():
    file = "history.csv"
    header = ["Name","Age","Gender","BPM","Condition"]
    data = [name, age, gender, bpm, label]

    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

if st.button("💾 Save Record"):
    save_history()
    st.success("Saved!")

# -----------------------------
# HISTORY
# -----------------------------
st.subheader("📂 Patient History")

if os.path.exists("history.csv"):
    df = pd.read_csv("history.csv")
    st.dataframe(df)

# -----------------------------
# WAVEFORM
# -----------------------------
st.subheader("📈 Live Waveform")

placeholder = st.empty()
for _ in range(100):
    
    if len(st.session_state.audio_data) > 500:
        raw = st.session_state.audio_data
    else:
        raw = np.zeros(300)

    filtered = heart_filter(raw)
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-6)

    peaks, _ = find_peaks(filtered, distance=80, prominence=0.3)

    if len(peaks) > 1:
        bpm = int(60 * 1000 / np.mean(np.diff(peaks)))
    else:
        bpm = 0

    with placeholder.container():
        st.line_chart(filtered[-300:])

    time.sleep(0.2)
# -----------------------------
# PLAY RECORDED AUDIO
# -----------------------------
st.subheader("🔊 Playback")

if st.button("Play Recorded Sound"):
    if len(st.session_state.audio_data) > 500:
        audio_file = save_temp_audio(st.session_state.audio_data)
        st.audio(audio_file)
    else:
        st.warning("No audio recorded yet")
# -----------------------------
# ADVANCED GRAPHS
# -----------------------------
st.subheader("📊 Advanced Analysis")

colA, colB = st.columns(2)

fft = np.abs(np.fft.fft(filtered))
with colA:
    st.line_chart(fft[:200])

rolling = np.convolve(np.abs(filtered), np.ones(50)/50, mode='same')
with colB:
    st.line_chart(rolling)

# -----------------------------
# SPECTROGRAM
# -----------------------------
fig,ax = plt.subplots()
ax.specgram(filtered,Fs=1000)
st.pyplot(fig)

# -----------------------------
# SUMMARY
# -----------------------------
st.subheader("📋 Patient Summary")

st.info(f"""
Name: {name}  
Age: {age}  
Condition: {label}  
BPM: {bpm}  
Arrhythmia: {"Yes" if arrhythmia else "No"}
""")

# -----------------------------
# FINAL DECISION
# -----------------------------
st.subheader("🩺 Final Clinical Decision")

if label == "Normal" and not arrhythmia:
    st.success("✔ No immediate concern")
elif label == "Murmur":
    st.warning("⚠ Further investigation required")
else:
    st.error("🚨 Immediate medical attention recommended")

st.markdown(
    "<h3 style='color:red;'>● LIVE MONITORING</h3>",
    unsafe_allow_html=True
)

# -----------------------------
# SAVE TEMP AUDIO (MIC)
# -----------------------------
def save_temp_audio(data):
    file = "temp_audio.wav"
    sf.write(file, data, 1000)
    return file
# -----------------------------
# REPORT
# -----------------------------
def generate_pdf():
    file=f"{name}_report.pdf"
    doc=SimpleDocTemplate(file)
    styles=getSampleStyleSheet()

    content=[]
    content.append(Paragraph(f"Name: {name}",styles["Normal"]))
    content.append(Paragraph(f"Age: {age}",styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}",styles["Normal"]))
    content.append(Paragraph(f"BPM: {bpm}",styles["Normal"]))
    content.append(Paragraph(f"Condition: {label}",styles["Normal"]))
    content.append(Paragraph(f"Recommendation: {status}",styles["Normal"]))

    img="plot.png"
    plt.figure()
    plt.plot(filtered)
    plt.savefig(img)
    plt.close()

    content.append(Image(img, width=400, height=200))
    doc.build(content)

    return file

st.subheader("📄 Report")

if st.button("Generate Report"):
    f=generate_pdf()
    with open(f,"rb") as file:
        st.download_button("⬇ Download", file, file_name=f)
