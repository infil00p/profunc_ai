import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

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

def process_directory(input_dir, output_dir):
    """Recursively process all PDFs in the input directory and save results in the output directory."""
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directories in the output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # Process each PDF file
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                output_txt_path = os.path.join(output_subdir, file.replace(".pdf", ".txt"))
                pdf_to_text(pdf_path, output_txt_path)

# Example usage
input_directory = "/home/bowserj/profunc/data/backups_from_paroxysms"  # Replace with your input directory
output_directory = "/home/bowserj/profunc/data/text_output"  # Replace with your output directory

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process all PDFs in the input directory tree
process_directory(input_directory, output_directory)
