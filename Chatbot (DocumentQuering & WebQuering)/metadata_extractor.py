import re

def extract_metadata(text, elements):
    metadata = {}

    # From raw text
    match = re.search(r'circular\s*no[.:]?\s*(\w+)', text, re.IGNORECASE)
    if match:
        metadata['circular_no'] = match.group(1)

    # Try title
    for el in elements:
        if isinstance(el, dict) and el.get("category") == "Title":
            metadata['title'] = el.get("text", "").strip()
            break

    # Try date
    match = re.search(r'(\d{1,2}[-/ ]\w{3,9}[-/ ]\d{2,4})', text)
    if match:
        metadata['date'] = match.group(1)

    # All links
    metadata['links'] = re.findall(r'https?://\S+', text)

    return metadata
