import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import shutil 
import subprocess 
import sys 
import atexit 
import math
import fitz # Library for PDF handling

# -----------------------------------------------------------------------------
# --- 1. PATH RESOLUTION & DIRECTORY SETUP ------------------------------------
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolve the project root by going up one level from SCRIPT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) 

# Now define all subsequent directories relative to the resolved PROJECT_ROOT
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input_files') 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output_text') 
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp_conversion') 

# Display the resolved path clearly for confirmation
print(f"--- Path Resolution Debug ---")
print(f"Project Root Resolved To: {PROJECT_ROOT}")
print(f"Input Directory Check: {INPUT_DIR}")
print("-----------------------------")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Global list to track temporary files for robust cleanup
TEMP_FILES_TO_CLEAN = []

# -----------------------------------------------------------------------------
# --- 2. CLEANUP & TESSERACT CONFIGURATION ------------------------------------
# -----------------------------------------------------------------------------

# Tesseract Configuration for high-accuracy results
OCR_CONFIG = r'--oem 3 --psm 6'

# --- TESSERACT PATH FIXES ---
TESSERACT_CMD = r"/mnt/c/Program Files/Tesseract-OCR/tesseract.exe" 
WINDOWS_TEMP_DIR = r"C:\TesseractTemp" 

print("\n--- DEBUG: Environment Setup ---")
print(f"DEBUG: TESSERACT_CMD set to: {TESSERACT_CMD}")

try:
    wsl_windows_temp_dir = os.path.join('/mnt/c/', WINDOWS_TEMP_DIR.replace('C:', ''))
    os.makedirs(wsl_windows_temp_dir, exist_ok=True)
    
    os.environ['TESS_TEMP_DIR'] = WINDOWS_TEMP_DIR
    
    print(f"DEBUG: Tesseract Temp Dir Fix applied via os.environ: {WINDOWS_TEMP_DIR}")
    
except Exception as e:
    print(f"CRITICAL ERROR: Failed to set/create Windows temp directory {WINDOWS_TEMP_DIR}. Error: {e}")
    sys.exit(1)

# Run a quick check 
try:
    test_file = os.path.join(wsl_windows_temp_dir, "test_write.txt")
    with open(test_file, 'w') as f:
        f.write("Test.")
    os.remove(test_file)
    print(f"DEBUG: Successfully wrote and removed file in {WINDOWS_TEMP_DIR}.")
except Exception as e:
    print(f"CRITICAL ERROR: Cannot write to {WINDOWS_TEMP_DIR} via WSL mount. Error: {e}")
    sys.exit(1)
    
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except pytesseract.TesseractNotFoundError:
    print(f"\n*** ERROR: Tesseract not found at: {TESSERACT_CMD}")
    sys.exit(1)
print(f"DEBUG: pytesseract.tesseract_cmd is set.")

# --- CLEANUP FUNCTION FOR EXIT ---
def cleanup_temp_files():
    """Removes all files tracked in TEMP_FILES_TO_CLEAN."""
    print("\n[CLEANUP] Initiating temporary file cleanup...")
    temp_files_cleaned = 0
    for f in TEMP_FILES_TO_CLEAN:
        if os.path.exists(f):
            try:
                os.remove(f)
                temp_files_cleaned += 1
            except Exception as e:
                print(f"Warning: Failed to clean up temp file {f}. Error: {e}")
    
    # Also clean up the consistent preprocessed temp file
    preproc_temp_path = os.path.join(TEMP_DIR, "preproc_temp.png")
    if os.path.exists(preproc_temp_path):
        try:
            os.remove(preproc_temp_path)
            temp_files_cleaned += 1
        except Exception as e:
            print(f"Warning: Failed to clean up {preproc_temp_path}. Error: {e}")
    
    if temp_files_cleaned > 0:
        print(f"[CLEANUP] Successfully removed {temp_files_cleaned} temporary file(s).")
    else:
        print("[CLEANUP] No temporary files requiring cleanup found.")

atexit.register(cleanup_temp_files)

# -----------------------------------------------------------------------------
# --- 3. PDF CONVERSION FUNCTIONS ---------------------------------------------
# -----------------------------------------------------------------------------

def get_pdf_page_count(pdf_path):
    """Returns the total number of pages in a PDF."""
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        print(f"❌ Error getting page count for {pdf_path}: {e}")
        return 0

def convert_pdf_to_image(pdf_path, temp_dir=TEMP_DIR, page_number=0):
    """Converts a single page of a PDF file to a high-resolution PNG image."""
    print(f"    - Converting PDF page {page_number + 1} to temporary image...")
    try:
        doc = fitz.open(pdf_path)
        if page_number >= doc.page_count:
            return None 

        page = doc.load_page(page_number)
        zoom = 300 / 72 # 300 DPI
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        # Create a unique temp file name for this page
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_img_path = os.path.join(temp_dir, f"{base_filename}_page_{page_number}.png")
        pix.save(temp_img_path)
        
        doc.close()
        return temp_img_path
        
    except Exception as e:
        print(f"❌ Error during PDF conversion: {e}")
        return None

# -----------------------------------------------------------------------------
# --- 4. SECURE DELETION FUNCTION ---------------------------------------------
# -----------------------------------------------------------------------------

def secure_delete(file_path, overwrite_passes=3):
    """
    Overwrites the file content multiple times with random data before deletion.
    """
    if not os.path.exists(file_path):
        return

    print(f"    - Starting {overwrite_passes} passes of secure overwrite...")
    
    try:
        file_size = os.path.getsize(file_path)
        
        for i in range(overwrite_passes):
            # Open the file in binary write mode ('r+b')
            with open(file_path, 'r+b') as f:
                f.seek(0)
                # Overwrite the entire file with cryptographically secure random bytes
                random_data = os.urandom(file_size)
                f.write(random_data)
                f.flush()
                os.fsync(f.fileno())

            time.sleep(0.1) 

        # Finally, delete the file using the OS command
        os.remove(file_path)
        print(f"    - ✅ SUCCESS: Securely deleted input file: {file_path}")
        
    except Exception as e:
        print(f"    - ⚠️ ERROR during secure deletion of {file_path}: {e}")
        
# -----------------------------------------------------------------------------
# --- 5. IMAGE PRE-PROCESSING (Final Robust Logic) ----------------------------
# -----------------------------------------------------------------------------

def deskew_image(image):
    """
    Detects and corrects only small skew/tilt of the image using moments, 
    while safeguarding against large 90 degree rotations.
    """
    
    # 1. Convert to Grayscale and Threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. Calculate Skew Angle
    coords = np.column_stack(np.where(thresh > 0))
    
    if coords.size == 0:
        print("    - Warning: No content detected for deskewing.")
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Normalize the angle into the range [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # 3. ROTATION SAFEGUARD: 
    # Only apply the rotation if the angle is small (true skew, e.g., < 30 degrees).
    if abs(angle) < 30.0:
        # Proceed with small deskew correction
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed = cv2.warpAffine(image, M, (w, h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        
        print(f"    - Deskewed by {angle:.2f} degrees.")
        return deskewed
    else:
        # Skip correction if the angle is too large (i.e., the algorithm incorrectly 
        # detected a ~90 degree rotation when the image is fine).
        print(f"    - Deskewing skipped. Detected angle ({angle:.2f}) is too large, likely a false positive.")
        return image

def preprocess_image(image_path):
    """
    Load image using PIL (for broader support), convert to CV2 format, 
    and apply processing without relying on the failing ROTATE_EXIF attribute.
    """
    temp_preproc_path = os.path.join(TEMP_DIR, "preproc_temp.png")
    try:
        # 1. Load image using PIL (no explicit EXIF rotation attempt)
        img_pil = Image.open(image_path)
        
        # Convert PIL Image to OpenCV format (NumPy array)
        img_cv = np.array(img_pil.convert('RGB')) # Ensure 3 channels for deskewing
        
        if img_cv is None:
              raise FileNotFoundError(f"Could not read image file at {image_path}")
              
        # 2. Deskew (Correct Tilt) - The safeguard prevents unwanted 90 degree rotations
        img_deskewed = deskew_image(img_cv)
        
        # 3. Final Processing (Grayscale, Denoising, Binarization)
        gray = cv2.cvtColor(img_deskewed, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3) 
        _, final_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save the final preprocessed image to a consistent temp file
        Image.fromarray(final_binary).save(temp_preproc_path)
        
        # Return the WSL path to the saved temp file for the subprocess OCR function
        return temp_preproc_path
        
    except Exception as e:
        print(f"    - ⚠️ ERROR during pre-processing of {image_path}: {e}")
        return None
# -----------------------------------------------------------------------------
# --- 6. CUSTOM OCR FUNCTION --------------------------------------------------
# -----------------------------------------------------------------------------

def run_tesseract_subprocess(image_wsl_path, config_flags, windows_temp_dir):
    """
    Executes Tesseract.exe directly via subprocess (WSL compatibility fix).
    """
    
    try:
        # Convert the WSL image path to a Windows path format for tesseract.exe
        windows_img_path = subprocess.check_output(
            ['wslpath', '-w', image_wsl_path], 
            text=True, 
            stderr=subprocess.PIPE
        ).strip()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to convert WSL path to Windows path: {e}")
        return None

    raw_command = [
        TESSERACT_CMD, 
        windows_img_path, 
        'stdout', 
        *config_flags.split() 
    ]
    
    custom_env = os.environ.copy()
    custom_env['TESS_TEMP_DIR'] = windows_temp_dir

    print(f"DEBUG: Subprocess Command: {' '.join(raw_command)}")
    print(f"DEBUG: Subprocess ENV used: TESS_TEMP_DIR={custom_env['TESS_TEMP_DIR']}")

    result = subprocess.run(
        raw_command, 
        capture_output=True, 
        text=True, 
        check=False,
        env=custom_env 
    )

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print(f"\n❌ Tesseract failed with Return Code {result.returncode}")
        print(f"Tesseract STDERR: {result.stderr.strip()}")
        return None

# -----------------------------------------------------------------------------
# --- 7. MAIN CONTROL LOGIC (Hybrid Batch/Prompt) -----------------------------
# -----------------------------------------------------------------------------

def run_secure_ocr_batch():
    """Manages the full OCR workflow: Preprocess, OCR, Save, Secure Delete Prompt."""
    
    # Updated to include PDF and common image formats
    valid_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    try:
        files_to_process = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
    except FileNotFoundError:
        print(f"❌ Error: The Input directory was not found: {INPUT_DIR}")
        return

    if not files_to_process:
        print(f"No documents found in '{INPUT_DIR}'. Please place files here and run again.")
        return

    print(f"--- Found {len(files_to_process)} sensitive document(s) to process ---")
    
    global WINDOWS_TEMP_DIR 

    for file_name in files_to_process:
        input_path = os.path.join(INPUT_DIR, file_name)
        base_name = os.path.splitext(file_name)[0]
        is_pdf = file_name.lower().endswith('.pdf')
        
        total_pages = 1
        
        if is_pdf:
            total_pages = get_pdf_page_count(input_path)
            if total_pages == 0:
                print(f"Error: Could not process PDF {file_name}. Skipping.")
                continue

        print(f"\nProcessing: **{file_name}** ({total_pages} page(s))")

        # List to hold text extracted from all pages of the current document
        document_extracted_text = []

        for page_index in range(total_pages):
            
            path_to_process = input_path
            temp_img_path = None
            
            if is_pdf:
                print(f"    - Processing Page {page_index + 1} of {total_pages}")
                temp_img_path = convert_pdf_to_image(input_path, page_number=page_index)
                if temp_img_path is None:
                    continue
                path_to_process = temp_img_path
                # Add PDF page image to cleanup list
                TEMP_FILES_TO_CLEAN.append(temp_img_path) 
            
            # 1. Image Pre-processing (OpenCV)
            preproc_wsl_path = preprocess_image(path_to_process)
            if preproc_wsl_path is None:
                continue

            # 2. CORE OCR EXECUTION (WSL Compatible Subprocess)
            try:
                extracted_text = run_tesseract_subprocess(preproc_wsl_path, OCR_CONFIG, WINDOWS_TEMP_DIR)
                
            except Exception as e:
                print(f"    - ⚠️ OCR failed for page {page_index+1}. Error: {e}")
                extracted_text = None
            
            if extracted_text is not None:
                document_extracted_text.append(extracted_text)
            else:
                document_extracted_text.append(f"[OCR FAILED ON PAGE {page_index+1}]")
        
        # --- End of Document Processing ---

        # 3. Save Output Text (All pages combined)
        output_path = os.path.join(OUTPUT_DIR, base_name + '.txt')
        final_text = "\n\n--- PAGE BREAK ---\n\n".join(document_extracted_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        
        # **********************************************
        # *** ENHANCEMENT: Report individual file path ***
        # **********************************************
        print(f"    - ✅ Text successfully saved to: **{output_path}**")

        # 4. SECURE CLEANUP PROMPT (Interactive Choice for the original file)
        if os.path.exists(input_path):
            while True:
                file_action = input(
                    "\nChoose action for the original sensitive file:\n"
                    "**S**: Securely Delete the file (overwrite 3x before deletion).\n"
                    "**K**: Keep the file in the 'input_files' directory.\n"
                    "**E**: Exit the program immediately (All remaining files will be kept).\n"
                    f"Enter choice (S/K/E): "
                ).upper()
                
                if file_action == 'S':
                    secure_delete(input_path)
                    break
                elif file_action == 'K':
                    print(f"ℹ️ Document **kept** in input directory: {input_path}")
                    break
                elif file_action == 'E':
                    print("\nInitiating clean exit...")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter S, K, or E.")
    
    print("\n=======================================================")
    print("*** All batch processing complete. ***")
    
    # **********************************************
    # *** ENHANCEMENT: Report output directory path ***
    # **********************************************
    print(f"Extracted text files are located in: **{OUTPUT_DIR}**")
    print("=======================================================")

# -----------------------------------------------------------------------------
# --- 8. MAIN EXECUTION BLOCK -------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    try:
        run_secure_ocr_batch()
            
    except KeyboardInterrupt:
        print("\n\n👋 **Keyboard Interrupt Detected (Ctrl+C).** Shutting down cleanly...")
        sys.exit(0)