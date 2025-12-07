import os
import subprocess
import sys
import shutil
from glob import glob
from PIL import Image

# --- CONFIGURATION -----------------------------------------------------------

# Define the name and language code for your new trained module
# CRITICAL: This is the name you will use with the -l flag (e.g., -l myforms)
MODULE_NAME = 'myforms' 
# Base language (typically 'eng' for English-based scripts)
BASE_LANG = 'eng' 

# Root directory where ground truth data resides
TRAINING_DATA_ROOT = '../training_data/'
CORRECT_IMAGES_DIR = os.path.join(TRAINING_DATA_ROOT, 'correct')
GROUND_TRUTH_DIR = os.path.join(TRAINING_DATA_ROOT, 'ground_truth')

# Output directory for temporary training files and the final module
OUTPUT_DIR = './custom_tessdata_temp/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full path to Tesseract's tessdata directory (standard location on Linux/WSL)
TESSDATA_PATH = '/usr/share/tesseract-ocr/5/tessdata'


# --- CORE TRAINING FUNCTIONS -------------------------------------------------

def check_dependencies():
    """Checks for necessary executables (Tesseract, training tools) and data paths."""
    print("--- 1. Checking Dependencies ---")
    
    # Check for Tesseract executable
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        print("✅ Tesseract executable found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: Tesseract executable not found or not in PATH.")
        print("   Ensure you ran 'sudo apt install tesseract-ocr'.")
        sys.exit(1)
        
    # Check for Tesseract training tools (lstmtraining is the key binary)
    try:
        subprocess.run(['lstmtraining', '--version'], capture_output=True, check=True)
        print("✅ Tesseract training tools (lstmtraining) found.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: Tesseract training tools (lstmtraining) not found.")
        print("   Hint: On Ubuntu/Debian, install 'tesseract-ocr-dev'.")
        sys.exit(1)
        
    # Check for training data
    if not os.path.isdir(CORRECT_IMAGES_DIR) or not os.listdir(CORRECT_IMAGES_DIR):
        print(f"❌ Error: No images found in {CORRECT_IMAGES_DIR}. Run the interactive trainer first.")
        sys.exit(1)
        
    print("--------------------------------")

def prepare_data():
    """Step 2: Convert images to TIFF and generate box files using ground truth."""
    print("--- 2. Preparing Data (TIFF Conversion & GT File Setup) ---")
    
    # 2a. TIFF Conversion (Tesseract training requires TIFF images)
    image_files = glob(os.path.join(CORRECT_IMAGES_DIR, '*.[jpPNGtiff]*'))
    processed_files = []
    
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        tif_path = os.path.join(OUTPUT_DIR, base_name + '.tif')
        
        # Move ground truth text file to the output directory alongside images
        gt_path_src = os.path.join(GROUND_TRUTH_DIR, base_name + '.txt')
        gt_path_dst = os.path.join(OUTPUT_DIR, base_name + '.gt.txt')
        
        if not os.path.exists(gt_path_src):
            print(f"❌ Missing ground truth file for {base_name}. Skipping.")
            continue
            
        try:
            # Convert to TIFF and ensure a clean, binary format
            Image.open(img_path).convert('L').save(tif_path, compression='none')
            shutil.copy(gt_path_src, gt_path_dst)
            processed_files.append(tif_path)
            
        except Exception as e:
            print(f"⚠️ Failed to process {img_path}: {e}")
            continue

    if not processed_files:
        print("❌ Error: No images successfully processed to TIFF format.")
        sys.exit(1)
        
    # 2b. Generate the training list file (for lstmtraining)
    list_file_path = os.path.join(OUTPUT_DIR, f'{MODULE_NAME}.train.list')
    with open(list_file_path, 'w') as f:
        for tif_file in processed_files:
            # The format is image_name.tif
            base_name = os.path.splitext(os.path.basename(tif_file))[0]
            # Write the file name without extension, followed by .tif
            f.write(f'{base_name}.tif\n')
            
    print(f"✅ {len(processed_files)} documents prepared.")
    print("--------------------------------")


def run_training_commands():
    """Step 3: Execute the feature extraction and LSTMTraining iteration (fine-tuning)."""
    print("--- 3. Running Feature Extraction and Training ---")

    list_file_path = os.path.join(OUTPUT_DIR, f'{MODULE_NAME}.train.list')
    
    # 3a. Feature Extraction (Creates .lstmf files)
    print("-> 3a. Running Feature Extraction (Creating .lstmf files)...")
    extraction_command = [
        'tesseract',
        list_file_path,  # Use the list file
        '--psm', '11',  # Use PSM 11 (Sparse text) for training data
        '--oem', '1',   # Use OEM 1 (LSTM only)
        '--outputbase', os.path.join(OUTPUT_DIR, MODULE_NAME),
        '--maxpages', '0',
        'lstmbox'       # Command to generate LSTMF (feature) files
    ]
    
    try:
        # Run command from within the OUTPUT_DIR to keep paths local
        subprocess.run(extraction_command, check=True, cwd=OUTPUT_DIR)
        print("✅ Feature extraction (.lstmf files) complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Feature extraction failed. Error: {e}")
        sys.exit(1)
        
    # 3b. Fine-Tuning/LSTMTraining
    print(f"-> 3b. Starting LSTMTraining (Fine-Tuning, MAX 1000 Iterations)...")
    
    training_command = [
        'lstmtraining',
        '--continue_from', os.path.join(TESSDATA_PATH, f'{BASE_LANG}.traineddata'),
        '--traineddata', os.path.join(TESSDATA_PATH, f'{BASE_LANG}.traineddata'),
        '--train_listfile', list_file_path,
        '--max_iterations', '1000', # Set a reasonable limit for fine-tuning
        '--model_output', os.path.join(OUTPUT_DIR, f'{MODULE_NAME}_checkpoint'), 
        '--debug_interval', '0', # Disable debugging output
    ]

    try:
        # Run command from within the OUTPUT_DIR
        subprocess.run(training_command, check=True, cwd=OUTPUT_DIR)
        print("✅ Training iteration complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed. Error: {e}")
        sys.exit(1)
        
    print("--------------------------------")


def combine_data():
    """Step 4: Combine the best checkpoint data into the final .traineddata file."""
    print("--- 4. Combining Final Trained Data ---")
    
    # Find the latest checkpoint file generated
    checkpoint_files = sorted(glob(os.path.join(OUTPUT_DIR, f'{MODULE_NAME}_checkpoint_*.checkpoint')))
    if not checkpoint_files:
        print("❌ Error: Could not find any checkpoint files to combine.")
        sys.exit(1)
        
    latest_checkpoint_file = checkpoint_files[-1]
    final_output_filename = f'{MODULE_NAME}.traineddata'
    
    # Command to stop training and combine the checkpoint into the final module
    combine_command = [
        'lstmtraining',
        '--stop_training',
        '--continue_from', latest_checkpoint_file,
        '--traineddata', os.path.join(TESSDATA_PATH, f'{BASE_LANG}.traineddata'),
        '--model_output', final_output_filename,
    ]
    
    try:
        # Run command from within the OUTPUT_DIR
        subprocess.run(combine_command, check=True, cwd=OUTPUT_DIR)
        
        # Move the final output file to a clean, permanent directory
        final_module_dir = './custom_tessdata'
        os.makedirs(final_module_dir, exist_ok=True)
        shutil.move(os.path.join(OUTPUT_DIR, final_output_filename), os.path.join(final_module_dir, final_output_filename))
        
        print(f"✅ SUCCESS! New module created: {os.path.join(final_module_dir, final_output_filename)}")
        
        print("\n--- NEXT STEPS (CRITICAL) ---")
        print(f"1. Move the new module to the Tesseract system directory:")
        print(f"   sudo cp {os.path.join(final_module_dir, final_output_filename)} {TESSDATA_PATH}")
        print(f"2. Update ocr_master_wsl.py config to use: -l {MODULE_NAME}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Combination failed. Error: {e}")
        sys.exit(1)
    
    # Clean up temporary files
    shutil.rmtree(OUTPUT_DIR)
    print("✅ Temporary training files cleaned up.")
    print("--------------------------------")


# --- MAIN EXECUTION ----------------------------------------------------------

def run_full_training_pipeline():
    """Executes the entire Tesseract fine-tuning pipeline."""
    
    print("====================================================")
    print(f"  TESSERACT TRAINING PIPELINE for {MODULE_NAME}.traineddata ")
    print("====================================================")
    
    # Execute the steps sequentially
    check_dependencies()
    run_full_training_pipeline()
    
    # The actual pipeline execution
    prepare_data()
    run_training_commands()
    combine_data()

if __name__ == '__main__':
    run_full_training_pipeline()