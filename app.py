import streamlit as st
import numpy as np
import sqlite3
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from streamlit_webrtc import webrtc_streamer

# -----------------------------
# AUDIO BUFFER
# -----------------------------
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

def audio_callback(frame):
    audio = frame.to_ndarray().flatten()
    st.session_state.audio_buffer.extend(audio.tolist())
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
# UI STYLE
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

if len(st.session_state.audio_buffer) > 500:
    raw = np.array(st.session_state.audio_buffer)
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
# FEATURES
# -----------------------------
energy = np.mean(np.abs(filtered))
variance = np.var(filtered)
noise = np.std(filtered)

# -----------------------------
# PCG CLASSIFICATION
# -----------------------------
if bpm<50 or bpm>130:
    label = "Extrastole"
elif variance>0.5:
    label = "Murmur"
else:
    label = "Normal"

# -----------------------------
# STATUS
# -----------------------------
if label=="Normal":
    status="🟢 Normal"
elif label=="Murmur":
    status="🟡 Murmur Detected"
else:
    status="🔴 Abnormal Rhythm"

# -----------------------------
# DASHBOARD
# -----------------------------
c1,c2,c3 = st.columns(3)

c1.metric("❤️ BPM", bpm)
c2.metric("Noise", f"{noise:.2f}")
c3.metric("Condition", label)

st.success(status if label=="Normal" else status)

# -----------------------------
# WAVEFORM
# -----------------------------
st.subheader("📈 Waveform")
st.line_chart(filtered)

# -----------------------------
# SPECTROGRAM
# -----------------------------
st.subheader("📊 Spectrogram")

fig,ax = plt.subplots()
ax.specgram(filtered,Fs=1000)
st.pyplot(fig)

# -----------------------------
# MURMUR ALERT
# -----------------------------
if label=="Murmur":
    st.warning("⚠ Possible murmur detected")

# -----------------------------
# PDF REPORT
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

    img="plot.png"
    plt.figure()
    plt.plot(filtered)
    plt.savefig(img)
    plt.close()

    content.append(Image(img, width=400, height=200))
    doc.build(content)

    return file

# -----------------------------
# DOWNLOAD REPORT
# -----------------------------
st.subheader("📄 Report")

if st.button("Generate Report"):
    f=generate_pdf()
    with open(f,"rb") as file:
        st.download_button("⬇ Download", file, file_name=f)
    
