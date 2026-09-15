""" Process the .log files from the inference runs on the dataset and collect them in a .json file  
    Will take the .log files stored in {results_foldername} 
"""

import re
import glob
import os
import json
from collections import defaultdict


results_foldername = "test_results"



def format_val_unc(mean, std):
    # [Keep your existing format_val_unc function exactly as it is]
    if std == 0.0:
        return f"{mean:.2f}", "0.00"
    sci_str = f"{std:e}"
    first_digit = sci_str[0]
    mag = int(sci_str.split('e')[1])
    sig_figs = 2 if first_digit == '1' else 1
    decimals = -mag + sig_figs - 1
    if decimals <= 0:
        round_pos = -decimals
        rounded_std = int(round(std, -round_pos))
        rounded_mean = int(round(mean, -round_pos))
        return f"{rounded_mean}", f"{rounded_std}"
    else:
        rounded_std = round(std, decimals)
        rounded_mean = round(mean, decimals)
        return f"{rounded_mean:.{decimals}f}", f"{rounded_std:.{decimals}f}"

def process_accuracy_files(folder_path, output_json_path):
    log_files = glob.glob(folder_path)

    if not log_files:
        print(f"No .log files found in: {folder_path}")
        return

    model_stats = defaultdict(dict)

    for file_path in log_files:
        with open(file_path, 'r') as file:
            text = file.read()

        gt_match = re.search(r"\|\s*(\d+)\s*with GT", text)
        acc_match = re.search(r"catalogue accuracy\s*:\s*[\d.]+%\s*\((\d+)/\d+\)", text)
        mc_match = re.search(r"multi-component\s*:\s*(\d+)/(\d+)", text)

        if not (gt_match and acc_match and mc_match):
            continue

        total_gt = int(gt_match.group(1))
        total_correct = int(acc_match.group(1))
        mc_correct = int(mc_match.group(1))
        mc_total = int(mc_match.group(2))

        sc_total = total_gt - mc_total
        sc_correct = total_correct - mc_correct

        if sc_total == 0 or mc_total == 0 or total_gt == 0:
            continue

        total_accuracy = (total_correct / total_gt) * 100
        sc_accuracy = (sc_correct / sc_total) * 100
        mc_accuracy = (mc_correct / mc_total) * 100

        name = os.path.basename(file_path).split("_ep10")[0]
        name = name.split("infer_")[1]
        
        if "_s" in name:
            model_name, seed_str = name.rsplit("_s", 1)
        else:
            model_name = name
            seed_str = "unknown"

        # Store accuracies in the dictionary for final average and pair calculation
        model_stats[model_name][seed_str] = {
            'total': total_accuracy,
            'sc': sc_accuracy,
            'mc': mc_accuracy
        }

    # ==========================================
    # SAVE DATA FOR PLOTTING SCRIPT
    # ==========================================
    with open(output_json_path, 'w') as f:
        json.dump(model_stats, f, indent=4)
    print(f"\nData successfully saved to {output_json_path}")

# === HOW TO USE ===
script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", results_foldername)
folder_path = os.path.join(data_folder, "*.log") 
output_json = os.path.join(script_dir, "..","results.json")

process_accuracy_files(folder_path, output_json)