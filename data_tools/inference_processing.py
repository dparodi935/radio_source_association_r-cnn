""" This will take the individual csv files from each inference run and join them into one overall csv file, dropping sources with no ground truth equivalent
"""
import pandas as pd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", "..", "cnn_data", "inference_outputs")

contents = os.listdir(data_folder)
csv_pd_list = []
    
for file in contents:
    if ".csv" not in file:
        continue
    filepath = os.path.join(data_folder,file)
    
    print(f"Processing {filepath}")
    csv_read_pd = pd.read_csv(filepath)
    csv_pd_list.append(csv_read_pd)


results_df = pd.concat(csv_pd_list, ignore_index=True)    

final_results_df = results_df[results_df["n_true"] > 0]

final_results_df.to_csv(os.path.join(data_folder,"inference.csv"), index=False)