""" Cleans inference csvs. Removes duplicates and sources with no ground truth equivalent
"""

import pandas as pd
import os
import glob

script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", "..", "cnn_data", "inference_outputs")

# Find all the individual encoding CSV files
csv_files = glob.glob(os.path.join(data_folder, "predictions_test_*.csv"))

if not csv_files:
    print("No CSV files found.")
else:
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        initial_rows = len(df)
        
        # 1. Delete duplicate rows based on source_name (keeping the most recent)
        df_no_dupes = df.drop_duplicates(subset=["source_name"], keep="last")
        dupes_removed = initial_rows - len(df_no_dupes)
        
        # 2. Delete rows with no ground truth equivalent (where n_true is 0)
        df_cleaned = df_no_dupes[df_no_dupes["n_true"] > 0]
        no_gt_removed = len(df_no_dupes) - len(df_cleaned)
        
        final_rows = len(df_cleaned)
        
        # 3. Overwrite the original file with the cleaned dataframe
        df_cleaned.to_csv(filepath, index=False)
        
        print(f"Processed {filename}:")
        print(f"  -> Initial rows: {initial_rows}")
        print(f"  -> Removed {dupes_removed} duplicate rows")
        print(f"  -> Removed {no_gt_removed} rows with 0 true components")
        print(f"  -> Final rows: {final_rows}\n")