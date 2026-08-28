""" Creates a table to count the number of sources which were solved by what number of encodings
"""
import pandas as pd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", "..", "cnn_data", "inference_outputs")

df = pd.read_csv(os.path.join(data_folder,"inference.csv"))

correct_df = df[df["status"]=="correct"]
incorrect_df = df[df["status"]!="correct"]

source_correct_counts = correct_df["source_name"].value_counts()
source_incorrect_counts = incorrect_df["source_name"].value_counts()

value_counts = source_correct_counts.value_counts().sort_index(ascending=False)
value_counts[0] = len(source_incorrect_counts[source_incorrect_counts.values == 5])

print(value_counts)

