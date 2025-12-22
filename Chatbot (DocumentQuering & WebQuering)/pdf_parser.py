import fitz 
def extract_elements_unstructured(pdf_path):
    print(f"Skipping unstructured layout-aware extraction for: {pdf_path}")
    return []  # Placeholder until fixed
print("2")

def parse_pdf_with_fitz(path):
    print(f"Parsing PDF with fitz: {path}")

    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
