import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import time
import shutil 
import fitz 
import subprocess 
import sys 
import atexit 

# --- 1. PATH RESOLUTION (Run-from-Anywhere) ---------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

INPUT_DIR = os.path.join(PROJECT_ROOT, 'input_files') 
CORRECTED_DIR = os.path.join(PROJECT_ROOT, 'training_data', 'correct')
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, 'training_data', 'ground_truth')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp_conversion') 

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(CORRECTED_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Global list to track temporary files for robust cleanup
TEMP_FILES_TO_CLEAN = []

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
## ⚙️ Tesseract Configuration and Path Fixes
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
## 📄 Script 0: PDF Conversion
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
    """Converts a single page of a PDF file to a high-resolution PIL Image."""
    print(f"    - Converting PDF page {page_number + 1} to temporary image...")
    try:
        doc = fitz.open(pdf_path)
        if page_number >= doc.page_count:
            return None 

        page = doc.load_page(page_number)
        zoom = 300 / 72 
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        temp_img_path = os.path.join(temp_dir, f"{os.path.basename(pdf_path)}_page_{page_number}.png")
        pix.save(temp_img_path)
        
        doc.close()
        return temp_img_path
        
    except Exception as e:
        print(f"❌ Error during PDF conversion: {e}")
        return None

# -----------------------------------------------------------------------------
## 🖼️ Script 1: Pre-processing Functions
# -----------------------------------------------------------------------------

def deskew_image(image):
    """Detects and corrects the skew/tilt of the image using moments."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    deskewed = cv2.warpAffine(image, M, (w, h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def preprocess_image(image_path):
    """Load image, apply pre-processing, save to temp file, and return path."""
    temp_preproc_path = os.path.join(TEMP_DIR, "preproc_temp.png")
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
              raise FileNotFoundError(f"Could not read image file at {image_path}")

        img_deskewed = deskew_image(img_cv)
        gray = cv2.cvtColor(img_deskewed, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3) 
        _, final_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save the preprocessed image to a single, consistent temp file for Tesseract to read
        Image.fromarray(final_binary).save(temp_preproc_path)
        return temp_preproc_path
        
    except Exception as e:
        print(f"    - ⚠️ ERROR during pre-processing of {image_path}: {e}")
        return None

# -----------------------------------------------------------------------------
## 🧠 Custom OCR Function
# -----------------------------------------------------------------------------

def run_tesseract_subprocess(image_wsl_path, config_flags, windows_temp_dir):
    """
    Executes Tesseract.exe directly via subprocess, ensuring the TESS_TEMP_DIR 
    environment variable is explicitly passed.
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
## 🔄 Script 2: Iterative Training Loop
# -----------------------------------------------------------------------------

def interactive_train(file_name, initial_config=r'--oem 3 --psm 6'):
    """
    Runs OCR on ALL pages/images and prompts the user for correction/feedback.
    """
    input_path = os.path.join(INPUT_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]
    is_pdf = file_name.lower().endswith('.pdf')
    
    page_index = 0
    total_pages = 1
    
    # Track if any ground truth was successfully captured
    ground_truth_captured = False 
    
    if is_pdf:
        total_pages = get_pdf_page_count(input_path)
        if total_pages == 0:
            print(f"Error: Could not process PDF {file_name}. Skipping.")
            return

    print(f"\n--- Starting Interactive Training for: {file_name} ({total_pages} page(s)) ---")
    
    global WINDOWS_TEMP_DIR 
    
    # Outer loop to handle all pages
    while page_index < total_pages:
        
        current_config = initial_config 
        page_base_name = f"{base_name}_page_{page_index+1}"
        
        path_to_process = input_path
        temp_img_path = None
        
        print(f"\n#######################################################")
        print(f"### Processing Page {page_index + 1} of {total_pages} ###")
        print(f"#######################################################")
        
        # --- Page Conversion ---
        if is_pdf:
            temp_img_path = convert_pdf_to_image(input_path, page_number=page_index)
            if temp_img_path is None:
                page_index += 1
                continue
            path_to_process = temp_img_path
            # Add to cleanup list for automatic removal by atexit
            TEMP_FILES_TO_CLEAN.append(temp_img_path) 
        
        # --- Preprocessing ---
        preproc_wsl_path = preprocess_image(path_to_process)
        if preproc_wsl_path is None:
            print(f"Failed to preprocess page {page_index+1}. Skipping.")
            page_index += 1
            continue

        # --- Inner Loop for Config Tuning ---
        final_ground_truth = None
        
        while True:
            full_config = current_config 
            
            print(f"\n[Page {page_index + 1}/{total_pages}] --- Running Tesseract with Config: {full_config} ---")
            
            extracted_text = run_tesseract_subprocess(
                preproc_wsl_path, 
                full_config, 
                WINDOWS_TEMP_DIR
            )

            if extracted_text is None:
                break 
                
            print("\n=======================================================")
            print(f"      OCR OUTPUT (Review and Correct) - Page {page_index + 1}")
            print("=======================================================")
            print(extracted_text)
            print("=======================================================")
            
            user_input = input(
                "Is the output correct? (Y/N/C/E)\n"
                "Y: Accept and save as ground truth.\n"
                "N: Type in the correct text now.\n"
                "C: Change Tesseract config (e.g., --psm 4) and re-run.\n"
                "E: Exit the program immediately (All temporary files will be cleaned).\n"
                "Enter choice (Y/N/C/E): "
            ).upper()

            if user_input == 'Y':
                final_ground_truth = extracted_text
                break
            elif user_input == 'N':
                print("\n-- Enter the COMPLETELY CORRECT text below (Ctrl+D or Ctrl+Z then Enter to finish): --")
                correct_text_lines = []
                while True:
                    try:
                        line = input()
                        correct_text_lines.append(line)
                    except EOFError:
                        break
                final_ground_truth = "\n".join(correct_text_lines)
                break
            elif user_input == 'C':
                new_config = input("Enter new Tesseract config flags (e.g., --psm 4): ")
                current_config = new_config
            elif user_input == 'E':
                # Exit cleanly from the program entirely
                print("\nInitiating clean exit...")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter Y, N, C, or E.")
        
        # --- POST-PAGE SAVE ---
        if final_ground_truth is not None:
            ground_truth_path = os.path.join(GROUND_TRUTH_DIR, page_base_name + '.txt')
            with open(ground_truth_path, 'w', encoding='utf-8') as f:
                f.write(final_ground_truth)
            print(f"\n✅ Ground Truth saved for Page {page_index + 1}: {ground_truth_path}")
            ground_truth_captured = True
        else:
            print(f"\n❌ OCR failed for Page {page_index + 1}. Skipping ground truth save.")
            
        page_index += 1
        
    # -----------------------------------------------------------
    # --- POST-DOCUMENT FINAL ACTION: EXIT/MOVE ---
    # -----------------------------------------------------------
    print("\n--- Document Processing Complete ---")
    
    # After all pages, ask the user what to do with the original input file
    if os.path.exists(input_path):
        
        # Loop for robust user input
        while True:
            file_action = input(
                "\nAll pages processed. What should be done with the original file?\n"
                "**M**: Move to 'training_data/correct' (Recommended).\n"
                "**K**: Keep the file in the 'input_files' directory.\n"
                "**D**: Delete the original file.\n"
                "**E**: Exit the program immediately (Cleanup is already done).\n"
                f"Enter choice (M/K/D/E): "
            ).upper()
            
            if file_action == 'M':
                corrected_doc_path = os.path.join(CORRECTED_DIR, file_name)
                shutil.move(input_path, corrected_doc_path)
                print(f"✅ Document moved to training set: {corrected_doc_path}")
                break
            elif file_action == 'K':
                print(f"ℹ️ Document **kept** in input directory: {input_path}")
                break
            elif file_action == 'D':
                os.remove(input_path)
                print(f"🗑️ Document **deleted**: {input_path}")
                break
            elif file_action == 'E':
                print("\nInitiating clean exit...")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter M, K, D, or E.")
    else:
        print("ℹ️ Original input file no longer exists (perhaps it was already moved/deleted).")
        
    print("--- Training loop finished. ---")


# -----------------------------------------------------------------------------
## 🏁 Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    valid_extensions = ('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    
    try:
        files_in_input = os.listdir(INPUT_DIR)
        files_to_process = [f for f in files_in_input if f.lower().endswith(valid_extensions)]
        
        if not files_to_process:
            print(f"Please place a document (image or PDF) in the '{INPUT_DIR}' folder to begin interactive training.")
        elif len(files_to_process) > 1:
            print("⚠️ Found multiple files. Please only place ONE file in the input folder for interactive training.")
            file_to_train = files_to_process[0]
            print(f"Processing only the first file: {file_to_train}")
            interactive_train(file_to_train)
        else:
            interactive_train(files_to_process[0])
            
    except FileNotFoundError:
        print(f"❌ Error: The Input directory was not found: {INPUT_DIR}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n👋 **Keyboard Interrupt Detected (Ctrl+C).** Shutting down cleanly...")
        # sys.exit(0) is called implicitly here, which triggers the atexit cleanup.
        sys.exit(0)