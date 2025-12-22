import os
import uuid
import io
from pdf2docx import Converter
from docx2pdf import convert as docx2pdf_convert
from PIL import Image, ImageOps
from fpdf import FPDF
import fitz  # PyMuPDF
from fastapi import HTTPException
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\ms\\AppData\\Local\\Programs\\Tesseract-OCR"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # your current script directory
TMP_DIR = os.path.join(BASE_DIR, "tmp")  # your created tmp folder inside project

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

def pdf_to_word(pdf_path: str) -> str:
    output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.docx")
    cv = Converter(pdf_path)
    cv.convert(output_path)
    cv.close()
    return output_path

def word_to_pdf(word_path: str) -> str:
    output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.pdf")
    docx2pdf_convert(word_path, output_path)
    return output_path

def image_to_pdf_with_ocr(image_path: str) -> str:
    # Preprocess image for better OCR
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    
    text = pytesseract.image_to_string(image, config='--oem 3 --psm 6', lang='eng')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=text)

    # Add original image below text - optional
    pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=pdf.w - 20)
    
    output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.pdf")
    pdf.output(output_path)
    return output_path

def extract_images_from_pdf(pdf_path, output_folder):
    pdf_doc = fitz.open(pdf_path)
    images = []
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"image_p{page_index+1}_{image_index}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            images.append(image_path)
    return images  # List of paths to saved images


def ocr_pdf_to_text_pdf(pdf_path: str, output_pdf_path: str, dpi=300):
    # Set the Poppler bin path according to your installed location
    poppler_path = r"C:\\Users\\ms\\poppler-25.07.0-0\\poppler-25.07.0\\Library\\bin"

    # Convert PDF pages to images with specified dpi and poppler path
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found in PDF.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for i, page_image in enumerate(pages):
        # Preprocess image: grayscale and autocontrast for better OCR
        image = ImageOps.grayscale(page_image)
        image = ImageOps.autocontrast(image)

        # OCR the processed image
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6', lang='eng')

        # Add a new page in output PDF and write OCR text
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Page {i + 1} OCR Text:\n\n{text.strip()}")

    # Save the new PDF with OCR text
    pdf.output(output_pdf_path)
    return output_pdf_path



def extract_images_from_pdf(pdf_path, output_folder):

    pdf_doc = fitz.open(pdf_path)
    images = []
    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"image_p{page_index+1}_{image_index}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            images.append(image_path)
    return images

def extract_ocr_text_from_pdf_images(pdf_path, output_pdf_path, tmpdir):

    pdf_doc = fitz.open(pdf_path)
    texts = []
    imgs = []

    for page_index in range(len(pdf_doc)):
        page = pdf_doc[page_index]
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"image_p{page_index+1}_{image_index}.{image_ext}"
            image_path = os.path.join(tmpdir, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
            except Exception as e:
                text = f"[Could not extract image {image_filename}: {e}]"
            texts.append(text)
            imgs.append(image_path)

    if not texts:
        raise HTTPException(status_code=400, detail="No images found in PDF to extract text from.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for i, text in enumerate(texts):
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Image {i+1} Extracted Text:\n\n{text.strip()}")
    pdf.output(output_pdf_path)

    for im in imgs:
        try:
            os.remove(im)
        except Exception:
            pass

    return output_pdf_path



def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
