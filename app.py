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
import time

# -----------------------------
# AUDIO BUFFER (REAL-TIME)
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
# FILTER FUNCTION
# -----------------------------
def heart_filter(sig):
    b,a = butter(3,[20/500,150/500],btype='band')
    return filtfilt(b,a,sig)

# -----------------------------
# REAL-TIME DISPLAY LOOP 🔥
# -----------------------------
placeholder = st.empty()

while True:

    # INPUT SOURCE
    if len(st.session_state.audio_data) > 500:
        raw = st.session_state.audio_data
    else:
        raw = np.zeros(300)

    # FILTER
    filtered = heart_filter(raw)
    filtered = filtered / (np.max(np.abs(filtered))+1e-6)

    # BPM
    peaks,_ = find_peaks(filtered, distance=80, prominence=0.3)

    if len(peaks)>1:
        bpm = int(60*1000/np.mean(np.diff(peaks)))
    else:
        bpm = 0

    # FEATURES
    variance = np.var(filtered)
    noise = np.std(filtered)

    # CLASSIFICATION
    if bpm<50 or bpm>130:
        label = "Extrastole"
    elif variance>0.5:
        label = "Murmur"
    else:
        label = "Normal"

    # DISPLAY
    with placeholder.container():

        c1,c2,c3 = st.columns(3)
        c1.metric("❤️ BPM", bpm)
        c2.metric("Noise", f"{noise:.2f}")
        c3.metric("Condition", label)

        if label=="Normal":
            st.success("🟢 Normal")
        elif label=="Murmur":
            st.warning("🟡 Murmur Detected")
        else:
            st.error("🔴 Abnormal Rhythm")

        st.line_chart(filtered)

        fig,ax = plt.subplots()
        ax.specgram(filtered,Fs=1000)
        st.pyplot(fig)

    time.sleep(0.3)

# -----------------------------
# PDF REPORT (STATIC)
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
    
