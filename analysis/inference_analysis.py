""" Creates a table to count the number of sources which were solved by what combinations of encodings
"""
import pandas as pd
from tabulate import tabulate
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(script_dir, "..", "..", "cnn_data", "inference_outputs")

df = pd.read_csv(os.path.join(data_folder,"inference.csv"))


def calc_encoding_acc_mix(df):
    correct_df = df[df["status"]=="correct"]
    incorrect_df = df[df["status"]!="correct"]

    source_correct_counts = correct_df["source_name"].value_counts()
    source_incorrect_counts = incorrect_df["source_name"].value_counts()

    value_counts = source_correct_counts.value_counts().sort_index(ascending=False)
    value_counts[0] = len(source_incorrect_counts[source_incorrect_counts.values == 5])

    print(value_counts)

    r3_correct_df = correct_df[correct_df["encoding"] == "radio3"]
    w1_correct_df = correct_df[correct_df["encoding"] != "radio3"]

    # 1. Get the counts for w1 (you already have r3)
    r3_correct_counts = r3_correct_df["source_name"].value_counts()
    w1_correct_counts = w1_correct_df["source_name"].value_counts()

    # 2. Combine them into a single DataFrame
    combined_df = pd.DataFrame({
        "r3_count": r3_correct_counts,
        "w1_count": w1_correct_counts
    })

    # 3. Clean up the table
    final_table = (
        combined_df
        .fillna(0)                     # Fill missing values with 0 (if a source is in one but not the other)
        .astype(int)                   # Convert the counts back to integers (fillna converts them to floats)
        .rename_axis("source_name")    # Name the index
        .reset_index()                 # Pop the index out into a standard column
    )
    # 1. Map encodings into three distinct buckets
    def categorize_encoding(val):
        if val == "radio3":
            return "r3_count"
        elif "w1m" in val:
            return "w1m_count"
        elif "w1" in val:
            return "w1_count"
        return "other"

    # 2. Build final_table with r3, w1, and w1m columns
    groups = correct_df["encoding"].apply(categorize_encoding)
    final_table = pd.crosstab(correct_df["source_name"], groups).reset_index()

    # Ensure all expected columns exist even if a category has zero entries
    for col in ["r3_count", "w1_count", "w1m_count"]:
        if col not in final_table.columns:
            final_table[col] = 0

    # 3. Calculate category metrics (using .sum() directly on boolean masks)
    r3 = final_table["r3_count"]
    w1 = final_table["w1_count"]       # max possible is 2 (radio2_w1, radio3_w1)
    w1m = final_table["w1m_count"]     # max possible is 2 (radio2_w1m, radio3_w1m)
    total_w = w1 + w1m                 # max possible is 4

    metrics = {
        # Full agreement
        "All correct (r3=1, w1=2, w1m=2)": ((r3 == 1) & (w1 == 2) & (w1m == 2)).sum(),
        "All w1 variants correct, no r3": ((r3 == 0) & (total_w == 4)).sum(),
        
        # Isolated correctness
        "Only r3 correct": ((r3 == 1) & (total_w == 0)).sum(),
        "Only w1 correct (no w1m, no r3)": ((r3 == 0) & (w1 > 0) & (w1m == 0)).sum(),
        "Only w1m correct (no w1, no r3)": ((r3 == 0) & (w1m > 0) & (w1 == 0)).sum(),
        
        # Mixed combinations with r3
        "r3 + all w1 only (no w1m)": ((r3 == 1) & (w1 == 2) & (w1m == 0)).sum(),
        "r3 + all w1m only (no w1)": ((r3 == 1) & (w1m == 2) & (w1 == 0)).sum(),
        "r3 + partial w1/w1m (1 to 3 total)": ((r3 == 1) & total_w.between(1, 3)).sum(),
        
        # Partial without r3
        "Partial w1/w1m (1 to 3 total, no r3)": ((r3 == 0) & total_w.between(1, 3)).sum(),
    }

    # 4. Format and print table
    table_data = [[k, v] for k, v in metrics.items()]
    print(tabulate(table_data, headers=["Metric", "Count"], tablefmt="github"))
