import sys
import os

pdf_path = "Dynamical.ai System Redesign Specification.pdf"

try:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print("--- START PDF CONTENT ---")
    print(text)
    print("--- END PDF CONTENT ---")
    sys.exit(0)
except ImportError:
    pass

try:
    import PyPDF2
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print("--- START PDF CONTENT ---")
    print(text)
    print("--- END PDF CONTENT ---")
    sys.exit(0)
except ImportError:
    pass

print("Could not import pypdf or PyPDF2. Trying raw read (unlikely to work well)...")
try:
    with open(pdf_path, 'rb') as f:
        content = f.read()
        # Try to find text strings
        import re
        text = re.sub(b'[^\x20-\x7E\n\r\t]', b'', content).decode('utf-8', errors='ignore')
        # Filter for long consecutive strings
        lines = text.split('\n')
        clean_lines = [l for l in lines if len(l) > 20]
        print("\n".join(clean_lines[:50])) # Print first 50 lines
except Exception as e:
    print(f"Failed raw read: {e}")
