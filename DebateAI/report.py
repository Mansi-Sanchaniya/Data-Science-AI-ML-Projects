from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def export_report(result, filename="report.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)
    story = []

    story.append(Paragraph(f"<b>Question:</b> {result['question']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Final Decision:</b> {result['final_decision']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b> {result['final_confidence']}", styles["Normal"]))

    for i, j in enumerate(result["judges"], 1):
        story.append(Paragraph(f"<b>Judge {i}</b>", styles["Heading3"]))
        story.append(Paragraph(str(j), styles["Normal"]))

    doc.build(story)
