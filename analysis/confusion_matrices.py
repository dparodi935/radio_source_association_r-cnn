import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap_matrices(csv_path):
    # 1. Load data
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 2. Identify correctness and component type
    df['is_correct'] = (df['status'] == 'correct')
    df['is_mc'] = (df['n_true'] > 1)
    
    # 3. Pivot the dataframe
    pivot_df = df.pivot_table(
        index='source_name', 
        columns='encoding', 
        values='is_correct', 
        aggfunc='first'
    )
    
    source_types = df.groupby('source_name')['is_mc'].first()
    merged_df = pivot_df.join(source_types)
    
    baseline = 'radio3'
    
    if baseline not in merged_df.columns:
        print(f"Error: Baseline '{baseline}' not found in the encodings.")
        return
        
    targets = [col for col in pivot_df.columns if col != baseline]
    
    # Helper to calculate matrix and accuracies
    def get_matrix_and_acc(data, target_name):
        valid_data = data[[baseline, target_name]].dropna()
        total = len(valid_data)
        
        if total == 0:
            return [[0, 0], [0, 0]], 0.0, 0.0
            
        both_correct = ((valid_data[baseline] == True) & (valid_data[target_name] == True)).sum()
        base_only    = ((valid_data[baseline] == True) & (valid_data[target_name] == False)).sum()
        targ_only    = ((valid_data[baseline] == False) & (valid_data[target_name] == True)).sum()
        both_wrong   = ((valid_data[baseline] == False) & (valid_data[target_name] == False)).sum()
        
        base_acc = ((both_correct + base_only) / total) * 100
        target_acc = ((both_correct + targ_only) / total) * 100
        
        # Format as 2x2 array
        matrix = [[both_correct, base_only], 
                  [targ_only, both_wrong]]
                  
        return matrix, base_acc, target_acc

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # 4. Iterate through encodings and generate a 1x3 figure for each
    for target in targets:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        subsets = [
            ("All Sources", merged_df),
            ("Single-Component (SC)", merged_df[merged_df['is_mc'] == False]),
            ("Multi-Component (MC)", merged_df[merged_df['is_mc'] == True])
        ]
        
        for i, (title, data) in enumerate(subsets):
            matrix, base_acc, targ_acc = get_matrix_and_acc(data, target)
            
            # Plot the heatmap matching the requested style
            sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False, 
                        square=True, ax=axes[i], annot_kws={"size": 16, "weight": "bold"}, 
                        linewidths=1, linecolor='white')
            
            # Format titles with accuracies
            axes[i].set_title(f"{title}\n{baseline}: {base_acc:.1f}% | {target}: {targ_acc:.1f}%", 
                              fontsize=13, pad=12, weight='bold')
            
            # Axis labels and ticks
            axes[i].set_xticklabels(['Correct', 'Wrong'], fontsize=12)
            axes[i].set_yticklabels(['Correct', 'Wrong'], fontsize=12, rotation=0)
            
            axes[i].set_xlabel(f'Predicted ({target})', fontsize=12, fontweight='bold')
            if i == 0:
                axes[i].set_ylabel(f'True ({baseline})', fontsize=12, fontweight='bold')
            else:
                axes[i].set_ylabel('')

        plt.tight_layout()
        save_path = os.path.join(script_dir, f"paired_matrices_{target}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close(fig)
        print(f"Saved: {save_path}")

# === HOW TO USE ===
script_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(script_dir, "..", "..", "cnn_data", "inference_outputs", "inference.csv") 

generate_heatmap_matrices(csv_path)