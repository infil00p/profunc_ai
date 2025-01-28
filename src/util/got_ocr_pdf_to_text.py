import os
import fitz  # PyMuPDF
#import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoModel, AutoTokenizer
import gc

# Load the GOT-OCR2.0 model and processor
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

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
        # It's crazy we're passing whole ass image data across, WTF!
        images.append(image_path)
    
    pdf_document.close()  # Release the PDF document
    return images


def extract_text_from_image(image_path):
    """Extract text from an image using GOT-OCR2.0."""
    print("Starting OCR on " + image_path)
    text = model.chat(tokenizer, image_path, ocr_type='ocr')
    print("OCR Completed, text is %d" % len(text)) 
    return text

def deep_ocr_images(images):
    full_text = ""

    for img in images:
        text = extract_text_from_image(img)
        full_text += text + "\n"

    return full_text

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
        text = deep_ocr_images(images)
        
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

def process_directory(input_dir, output_dir, images_base_dir):
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directories in the output and images directories
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        images_subdir = os.path.join(images_base_dir, relative_path)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # Process each PDF file
        for file in files:
            if file.lower().endswith(".pdf") or file.lower().endswith(".PDF"):
                suffix = ".pdf"
                if file.lower().endswith(".PDF"):
                    suffix = ".PDF"
                pdf_path = os.path.join(root, file)
                output_txt_path = os.path.join(output_subdir, file.replace(suffix, ".txt"))
                images_dir = os.path.join(images_subdir, file.replace(suffix, ""))
                
                process_pdf(pdf_path, output_txt_path, images_dir)
        

# Example usage
input_directory = "/home/bowserj/profunc/data/backups_from_paroxysms"  # Replace with your input directory
output_directory = "/home/bowserj/profunc/data/text_output"  # Replace with your output directory
images_directory = "/home/bowserj/profunc/data/image_output"  # Replace with your images directory

# Ensure the output and images directories exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
if not os.path.exists(images_directory):
    os.makedirs(images_directory)

# This is using GPU, so we have to keep the max number of workers as 1 for now
process_directory(input_directory, output_directory, images_directory)
