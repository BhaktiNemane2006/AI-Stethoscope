from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(name, bpm, status, confidence):
    doc = SimpleDocTemplate(f"{name}_report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Heart Rate: {bpm} BPM", styles["Normal"]))
    content.append(Paragraph(f"Condition: {status}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))

    doc.build(content)

    return f"{name}_report.pdf"