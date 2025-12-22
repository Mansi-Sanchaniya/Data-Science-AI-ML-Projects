import os
import fitz  # PyMuPDF
import camelot
import spacy
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image


import torch

# Load spaCy NLP model for NER and keyphrase extraction
try:
    SPACY_NLP = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except ImportError:
    print("spaCy not installed - NER will be skipped")
    HAS_SPACY = False


def extract_tables_with_camelot(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
    tables_list = []
    for table in tables:
        tables_list.append(table.df)
    print('extrcated tables')
    return tables_list


def extract_figures(pdf_path, output_folder="figures"):
    with fitz.open(pdf_path) as doc:
        os.makedirs(output_folder, exist_ok=True)
        figure_paths = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = os.path.join(output_folder, f"page{page_num+1}_img{img_index+1}.{image_ext}")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                figure_paths.append(image_path)
                print('figure appended')
    return figure_paths


def extract_ner_entities(text: str):
    if not HAS_SPACY or not text.strip():
        return []
    doc = SPACY_NLP(text)
    allowed_labels = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "LAW", "NORP"}
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in allowed_labels]
    print('extracted entities')
    return entities


def extract_keyphrases(text: str, max_phrases: int = 5):
    if not HAS_SPACY or not text.strip():
        return []
    doc = SPACY_NLP(text)
    phrases = list({chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip().split()) <= 5})
    phrases_sorted = sorted(phrases, key=lambda x: -len(x.split()))
    return phrases_sorted[:max_phrases]

def normalize_bbox(bbox, page_width, page_height):
    return [
        max(0, min(1000, int(1000 * bbox[0] / page_width))),
        max(0, min(1000, int(1000 * bbox[1] / page_height))),
        max(0, min(1000, int(1000 * bbox[2] / page_width))),
        max(0, min(1000, int(1000 * bbox[3] / page_height))),
    ]

def extract_text_and_boxes_layoutlmv3(page):
    words = page.get_text("words")  # (x0, y0, x1, y1, word, ...)
    words = sorted(words, key=lambda w: (w[1], w[0]))  # top-to-bottom, left-to-right

    tokens = []
    boxes = []
    page_width, page_height = page.rect.width, page.rect.height


    for w in words:
        x0, y0, x1, y1, word = w[:5]
        tokens.append(word)
        boxes.append(normalize_bbox([x0, y0, x1, y1], page_width, page_height))

    # Render page image at 2x scale, RGB
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    raw_text = " ".join(tokens)
    print('extracted text and box layout')
    return tokens, boxes, img, raw_text


def layoutlmv3_predict(ocr_tokens, ocr_boxes, image, processor, model):

    encoding = processor(image,
                         ocr_tokens,
                         boxes=ocr_boxes,
                         return_tensors="pt",
                         truncation=True,
                         padding="max_length")

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()

    labels = [model.config.id2label.get(idx, "O") for idx in predictions]
    tokens = encoding.tokens()

    # Filter out special tokens ([CLS], [SEP], [PAD])
    valid_indexes = [i for i, t in enumerate(tokens) if t not in ["[CLS]", "[SEP]", "[PAD]"]]
    filtered = [(tokens[i], labels[i]) for i in valid_indexes]

    return filtered


def parse_pdf_with_layoutlmv3(pdf_path, processor, model):

    with fitz.open(pdf_path) as doc:

        output = []

        for i, page in enumerate(doc):
            tokens, boxes, image, raw_text = extract_text_and_boxes_layoutlmv3(page)
            token_labels = layoutlmv3_predict(tokens, boxes, image, processor, model)
            output.append({
                "page_num": i + 1,
                "tokens_with_labels": token_labels,
                "raw_text": raw_text
            })

    return output


def parse_pdf_enhanced(pdf_path):
    # Initialize LayoutLMv3Processor with apply_ocr=False since we pass tokens/boxes manually
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

    processor = LayoutLMv3Processor.from_pretrained("nielsr/layoutlmv3-finetuned-funsd", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("nielsr/layoutlmv3-finetuned-funsd")

    
    print(f"Parsing PDF with LayoutLMv3 semantic extraction: {pdf_path}")

    tables = extract_tables_with_camelot(pdf_path)
    figures = extract_figures(pdf_path)
    layoutlm_results = parse_pdf_with_layoutlmv3(pdf_path, processor, model)

    full_text = " ".join(page_data["raw_text"] for page_data in layoutlm_results)
    entities = extract_ner_entities(full_text)
    keyphrases = extract_keyphrases(full_text)

    return {
        "tables": tables,                # List of pandas DataFrames
        "figures": figures,              # List of figure image paths
        "layoutlm_output": layoutlm_results,  # token-label pairs per page
        "full_text": full_text,
        "entities": entities,
        "keyphrases": keyphrases,
    }
