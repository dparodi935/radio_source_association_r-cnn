""" Splits sources from selected folder into those which all, none and some encodings got correctly
"""

import pandas as pd
import os
import shutil

# --- 1. Set up your directories ---
script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", "cnn_data")

image_source_dir = os.path.join(data_folder, "inference_outputs - Copy") 
sorted_output_dir = os.path.join(data_folder, "sorted_cutouts")

# --- 2. Load the CSV Data ---
csv_path = os.path.join(image_source_dir, "inference.csv")
df = pd.read_csv(csv_path)

status_suffix = {
    "correct": "_rn",
    "too_many": "_tm",
    "too_few": "_tf",
    "mixed": "_ot"
}

# The specific encodings to keep for unanimous sources
ALLOWED_ENCODINGS = {"radio3", "radio3_w1", "radio3_w1m"}

# --- 3. Group by source and sort into folders ---
grouped = df.groupby("source_name")

files_copied = 0
files_missing = 0
files_skipped = 0

for source, group in grouped:
    n_total = len(group)
    n_right = (group["status"] == "correct").sum()
    n_wrong = n_total - n_right
    
    # Determine the target folder and if it is unanimous
    is_unanimous = False
    
    if n_right == 0:
        target_folder = os.path.join(sorted_output_dir, "1_All_Wrong")
        is_unanimous = True
    elif n_right == n_total:
        target_folder = os.path.join(sorted_output_dir, "3_All_Right")
        is_unanimous = True
    else:
        folder_name = f"{source}_{n_right}_right_{n_wrong}_wrong"
        target_folder = os.path.join(sorted_output_dir, "2_Mixed", folder_name)
        
    os.makedirs(target_folder, exist_ok=True)
    
    # --- 4. Reconstruct filenames and copy ---
    for _, row in group.iterrows():
        encoding = row["encoding"]
        status = row["status"]
        
        # If it's an all-right or all-wrong folder, skip unapproved encodings
        if is_unanimous and encoding not in ALLOWED_ENCODINGS:
            files_skipped += 1
            continue
            
        suffix = status_suffix.get(status, "_ot")
        img_name = f"{source}{suffix}_{encoding}.png"
        
        src_path = os.path.join(image_source_dir, img_name)
        dst_path = os.path.join(target_folder, img_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            files_copied += 1
        else:
            print(f"Warning: Could not find image -> {img_name}")
            files_missing += 1

print("\n--- Sorting Complete ---")
print(f"Images successfully copied : {files_copied}")
print(f"Images skipped (filtered)  : {files_skipped}")
print(f"Images missing/not found   : {files_missing}")
print(f"Check your results in      : {sorted_output_dir}")