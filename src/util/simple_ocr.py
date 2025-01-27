import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

# Set the path to the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def pdf_to_images(pdf_path):
    """Convert PDF to a list of images."""
    images = []
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    return images

def ocr_images(images):
    """Perform OCR on a list of images and return the text."""
    full_text = ""
    
    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    
    return full_text

def pdf_to_text(pdf_path, output_txt_path):
    """Convert a PDF to text using OCR."""
    print(f"Processing {pdf_path}...")
    
    # Step 1: Convert PDF to images
    images = pdf_to_images(pdf_path)
    
    # Step 2: Perform OCR on the images
    text = ocr_images(images)
    
    # Step 3: Save the extracted text to a file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Text saved to {output_txt_path}")

# Example usage
#pdf_path = "example.pdf"  # Path to your PDF file
#output_txt_path = "output.txt"  # Path to save the extracted text
#pdf_to_text(pdf_path, output_txt_path)

import os

input_folder = "path/to/pdf/folder"
output_folder = "path/to/output/folder"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for pdf_file in os.listdir(input_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, pdf_file)
        output_txt_path = os.path.join(output_folder, pdf_file.replace(".pdf", ".txt"))
        pdf_to_text(pdf_path, output_txt_path)
