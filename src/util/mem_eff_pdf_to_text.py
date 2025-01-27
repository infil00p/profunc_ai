import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

def pdf_to_images(pdf_path, images_dir):
    """Convert PDF to a list of images and save them to the images directory."""
    images = []
    pdf_document = fitz.open(pdf_path)
    
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Only resample huge images, otherwise the quality is bad and we can't read it
        if img.width > 1500 : 
            img = img.resize((img.width // 2, img.height // 2), Image.Resampling.LANCZOS)

        # Save the image
        image_path = os.path.join(images_dir, f"page_{page_num + 1}.jpg")
        img.save(image_path, "JPEG", quality=85)
        images.append(img)
    
    pdf_document.close()  # Release the PDF document
    return images

def ocr_images(images):
    """Perform OCR on a list of images and return the text."""
    full_text = ""
    
    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    
    return full_text

def process_pdf(pdf_path, output_txt_path, images_dir):
    """Process a single PDF: convert to images, perform OCR, and save results."""
    print(f"Processing {pdf_path}...")
    
    try:
        # Step 1: Convert PDF to images and save them
        images = pdf_to_images(pdf_path, images_dir)
        
        # Step 2: Perform OCR on the images
        text = ocr_images(images)
        
        # Step 3: Save the extracted text to a file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"Text saved to {output_txt_path}")
        print(f"Images saved to {images_dir}")
        
        # Release memory
        del images
        gc.collect()
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False

def process_directory(input_dir, output_dir, images_base_dir, max_workers=4):
    """Recursively process all PDFs in the input directory using parallel processing."""
    futures = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(input_dir):
            # Create corresponding directories in the output and images directories
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            images_subdir = os.path.join(images_base_dir, relative_path)
            
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # Process each PDF file
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    output_txt_path = os.path.join(output_subdir, file.replace(".pdf", ".txt"))
                    images_dir = os.path.join(images_subdir, file.replace(".pdf", ""))
                    
                    # Submit the task to the process pool
                    future = executor.submit(process_pdf, pdf_path, output_txt_path, images_dir)
                    futures.append(future)
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            future.result()  # Raise any exceptions that occurred during processing

# Example usage
input_directory = "/home/bowserj/profunc/data/backups_from_paroxysms"  # Replace with your input directory
output_directory = "/home/bowserj/profunc/data/text_output"  # Replace with your output directory
images_directory = "/home/bowserj/profunc/data/image_output"  # Replace with your images directory

# Ensure the output and images directories exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not os.path.exists(images_directory):
    os.makedirs(images_directory)

# Process all PDFs in the input directory tree using parallel processing
process_directory(input_directory, output_directory, images_directory, max_workers=4)
