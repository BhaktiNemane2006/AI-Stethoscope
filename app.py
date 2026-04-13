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
import os

# -----------------------------
# DATABASE (SQLite)
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

# Default users
c.execute("INSERT OR IGNORE INTO users VALUES ('admin','1234','admin')")
c.execute("INSERT OR IGNORE INTO users VALUES ('doctor','1234','doctor')")
conn.commit()

# -----------------------------
# LOGIN SYSTEM
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
            st.success(f"Welcome {user[2]} 👨‍⚕️")
        else:
            st.error("Invalid credentials")

    st.stop()

# -----------------------------
# ICU STYLE UI
# -----------------------------
st.markdown("""
<style>
body {background-color:#000000; color:#00FFAA;}
.stApp {background-color:#000000;}
h1,h2,h3 {color:#00FFAA; text-align:center;}
.stMetric {font-size:28px !important; color:#00FFAA !important;}
</style>
""", unsafe_allow_html=True)

st.title("💓 AI ICU Heart Monitoring System")
st.markdown("---")

# -----------------------------
# FILE UPLOAD + AUDIO
# -----------------------------
uploaded_file = st.file_uploader("Upload Heart Sound (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

# -----------------------------
# PATIENT INFO
# -----------------------------
patient_name = st.sidebar.text_input("Patient Name")

# -----------------------------
# SIGNAL
# -----------------------------
def generate_signal():
    t = np.linspace(0,1,300)
    return np.sin(2*np.pi*2*t) + np.random.normal(0,0.1,300)

if uploaded_file:
    y, sr = librosa.load(uploaded_file, sr=1000)
    raw_data = y[:2000]
else:
    raw_data = generate_signal()

filtered = bandpass(np.array(raw_data))
filtered = filtered / (np.max(np.abs(filtered)) + 1e-6)

# -----------------------------
# BPM
# -----------------------------
peaks, _ = find_peaks(filtered, distance=50, height=0.2)
bpm = len(peaks) * 60

# -----------------------------
# AI LOGIC
# -----------------------------
energy = np.mean(np.abs(filtered))
variance = np.var(filtered)

confidence = min((energy + variance) * 3, 1.0)
status = "Normal" if confidence < 0.6 else "Abnormal"

# -----------------------------
# DASHBOARD
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("❤️ Heart Rate", f"{bpm} BPM")
col2.metric("🧠 Confidence", f"{confidence*100:.1f}%")
col3.metric("📡 Status", status)

if status == "Normal":
    st.success("🟢 Patient Stable")
else:
    st.error("🔴 Immediate Attention Required")

st.markdown("---")

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
# PROBABILITY GRAPH
# -----------------------------
st.subheader("📊 AI Probability")
st.bar_chart({"Normal":1-confidence, "Abnormal":confidence})

# -----------------------------
# DOCTOR RECOMMENDATION
# -----------------------------
st.subheader("🧑‍⚕️ Recommendation")

if confidence < 0.6:
    st.success("Healthy heart. Maintain lifestyle.")
else:
    st.error("Consult cardiologist immediately.")

st.markdown("---")

# -----------------------------
# LIVE WAVEFORM
# -----------------------------
st.subheader("📈 Live Waveform")
placeholder = st.empty()

for _ in range(8):
    placeholder.line_chart(generate_signal())
    time.sleep(0.1)

# -----------------------------
# SPECTROGRAM
# -----------------------------
st.subheader("📊 Spectrogram")

fig, ax = plt.subplots()
ax.specgram(filtered, Fs=1000)
st.pyplot(fig)

# -----------------------------
# PDF REPORT FUNCTION
# -----------------------------
def generate_pdf(name, bpm, status, confidence, signal):
    filename = f"{name}_report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph(f"Patient: {name}", styles["Normal"]))
    content.append(Paragraph(f"BPM: {bpm}", styles["Normal"]))
    content.append(Paragraph(f"Condition: {status}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))

    # Save waveform image
    img_path = "waveform.png"
    plt.figure()
    plt.plot(signal)
    plt.title("Heart Waveform")
    plt.savefig(img_path)
    plt.close()

    content.append(Image(img_path, width=400, height=200))

    doc.build(content)

    return filename

# -----------------------------
# DOWNLOAD REPORT
# -----------------------------
st.subheader("📄 Medical Report")

if st.button("Generate & Download Report"):
    file = generate_pdf(patient_name, bpm, status, confidence, filtered)

    with open(file, "rb") as f:
        st.download_button(
            label="⬇ Download Report",
            data=f,
            file_name=file,
            mime="application/pdf"
        )

# -----------------------------
# ADMIN PANEL
# -----------------------------
if st.session_state.role == "admin":
    st.subheader("⚙ Admin Panel")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password")
    new_role = st.selectbox("Role", ["doctor","admin"])

    if st.button("Add User"):
        c.execute("INSERT INTO users VALUES (?,?,?)", (new_user, new_pass, new_role))
        conn.commit()
        st.success("User Added")

# -----------------------------
# LOGOUT
# -----------------------------
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
    
