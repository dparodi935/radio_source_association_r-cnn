import json
import os
import statistics
import matplotlib.pyplot as plt
import numpy as np

def format_val_unc(mean, std):
    """
    Formats the mean and uncertainty so that uncertainty has 1 sig fig, 
    unless the first significant digit is 1, in which case it uses 2.
    """
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

def print_raw_table(raw_table_rows):
    print("\n" + "="*68)
    print("LATEX RAW PERFORMANCE TABLE (ALL MODELS)")
    print("="*68)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l|ccc}")
    print(r"Model & Total (\%) & SC (\%) & MC (\%) \\")
    print(r"\hline")
    for row in raw_table_rows:
        print(row)
    print(r"\end{tabular}")
    print(r"\caption{Absolute Performance (Mean $\pm$ Half-Range)}")
    print(r"\label{tab:raw_performance}")
    print(r"\end{table}")
    print("\n")

def print_delta_table(delta_table_rows, baseline_model):
    safe_base_name = baseline_model.replace("_", "\\_")
    print("="*68)
    print("LATEX PAIRED DELTAS TABLE (ERROR ADDITION)")
    print("="*68)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{l|ccc}")
    print(r"Model & $\Delta$ Total (\%) & $\Delta$ SC (\%) & $\Delta$ MC (\%) \\")
    print(r"\hline")
    for row in delta_table_rows:
        print(row)
    print(r"\end{tabular}")
    print(rf"\caption{{Accuracy Deltas vs Baseline ({safe_base_name}). Uncertainties calculated via absolute error addition.}}")
    print(r"\label{tab:paired_deltas}")
    print(r"\end{table}")
    print("\n")

def plot_raw_and_deltas(json_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run the processing script first.")
        return
        
    with open(json_path, 'r') as f:
        model_stats = json.load(f)

    baseline_model = "radio3"
    target_order = ['radio3', 'radio2_w1', 'radio2_w1m', 'radio3_w1', 'radio3_w1m']
    
    raw_data_store = {}
    
    # Store data for raw plots
    plot_data_tot = []
    plot_data_sc = []
    plot_data_mc = []
    plot_models = []

    # ==========================================
    # 1. CALCULATE RAW NUMBERS FOR ALL MODELS
    # ==========================================
    raw_table_rows = []
    
    for model in target_order:
        if model not in model_stats:
            continue
            
        seeds_dict = model_stats[model]
        tots = [metrics['total'] for metrics in seeds_dict.values()]
        scs = [metrics['sc'] for metrics in seeds_dict.values()]
        mcs = [metrics['mc'] for metrics in seeds_dict.values()]
        
        plot_data_tot.append(tots)
        plot_data_sc.append(scs)
        plot_data_mc.append(mcs)
        plot_models.append(model)
        
        row_strings = []
        model_results = {}
        
        for metric_name, vals in zip(['total', 'sc', 'mc'], [tots, scs, mcs]):
            if not vals:
                raise ValueError(f"Missing values for {model}")
                
            mean_val = statistics.mean(vals)
            range_err = (max(vals) - min(vals)) / 2.0 if len(vals) > 1 else 0.0
                
            model_results[metric_name] = {'mean': mean_val, 'err': range_err}
            
            m_str, e_str = format_val_unc(mean_val, range_err)
            row_strings.append(f"${m_str} \\pm {e_str}$")
            
        raw_data_store[model] = model_results
        safe_model_name = model.replace("_", "\\_")
        raw_table_rows.append(f"{safe_model_name} & {row_strings[0]} & {row_strings[1]} & {row_strings[2]} \\\\")

    print_raw_table(raw_table_rows)

    # ==========================================
    # 2. CALCULATE PAIRED DELTAS (PROPAGATING ERRORS)
    # ==========================================
    delta_table_rows = []
    base_data = raw_data_store.get(baseline_model)

    delta_models = []
    delta_means_tot, delta_errs_tot = [], []
    delta_means_sc, delta_errs_sc = [], []
    delta_means_mc, delta_errs_mc = [], []

    for model in target_order:
        if model == baseline_model or model not in raw_data_store:
            continue
            
        mod_data = raw_data_store[model]
        row_strings = []
        delta_models.append(model)
        
        for metric in ['total', 'sc', 'mc']:
            delta_mean = mod_data[metric]['mean'] - base_data[metric]['mean']
            delta_err = mod_data[metric]['err'] + base_data[metric]['err']
            
            if metric == 'total':
                delta_means_tot.append(delta_mean)
                delta_errs_tot.append(delta_err)
            elif metric == 'sc':
                delta_means_sc.append(delta_mean)
                delta_errs_sc.append(delta_err)
            elif metric == 'mc':
                delta_means_mc.append(delta_mean)
                delta_errs_mc.append(delta_err)
            
            m_str, e_str = format_val_unc(delta_mean, delta_err)
            row_strings.append(f"${m_str} \\pm {e_str}$")
            
        safe_model_name = model.replace("_", "\\_")
        delta_table_rows.append(f"{safe_model_name} & {row_strings[0]} & {row_strings[1]} & {row_strings[2]} \\\\")

    print_delta_table(delta_table_rows, baseline_model)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # ==========================================
    # 3. GENERATE RAW VALUE SCATTER PLOTS (3 files)
    # ==========================================
    x_positions = np.arange(len(plot_models))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(plot_models)]

    raw_metrics_to_plot = [
        ('total', 'Raw Total Accuracy', plot_data_tot),
        ('sc', 'Raw Single-Component (SC) Accuracy', plot_data_sc),
        ('mc', 'Raw Multi-Component (MC) Accuracy', plot_data_mc)
    ]

    for file_suffix, title, data_list in raw_metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, data_points in enumerate(data_list):
            ax.scatter([x_positions[i]] * len(data_points), data_points, color=colors[i], marker='x', 
                       s=60, alpha=0.8, linewidths=1.5)
            
        ax.set_ylabel(f'{title} (%)', fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_models, rotation=15, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        global_min = min([min(pts) for pts in data_list if pts])
        global_max = max([max(pts) for pts in data_list if pts])
        padding = (global_max - global_min) * 0.2 if global_max != global_min else 2.0
        ax.set_ylim(max(global_min - padding, 0), min(global_max + padding, 100))
        
        plt.tight_layout()
        save_path = os.path.join(script_dir, "..", "..", f"scatter_{file_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) 
        print(f"Saved plot: {save_path}")

    # ==========================================
    # 4. GENERATE INDIVIDUAL DELTA PLOTS (3 files)
    # ==========================================
    delta_x_positions = np.arange(len(delta_models))
    delta_colors = colors[1:len(delta_models) + 1]

    delta_metrics_to_plot = [
        ('delta_total', 'Δ Total Accuracy', delta_means_tot, delta_errs_tot),
        ('delta_sc', 'Δ Single-Component (SC) Accuracy', delta_means_sc, delta_errs_sc),
        ('delta_mc', 'Δ Multi-Component (MC) Accuracy', delta_means_mc, delta_errs_mc)
    ]

    for file_suffix, title, means, errs in delta_metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(len(delta_models)):
            ax.errorbar(delta_x_positions[i], means[i], yerr=errs[i],
                        fmt='o', color=delta_colors[i], ecolor=delta_colors[i],
                        elinewidth=1.5, capsize=5, capthick=1.5, markersize=7)

        ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
        ax.set_ylabel(f'{title} (%)', fontsize=12)
        ax.set_xticks(delta_x_positions)
        ax.set_xticklabels(delta_models, rotation=15, fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        d_lows = [m - e for m, e in zip(means, errs)]
        d_highs = [m + e for m, e in zip(means, errs)]
        y_min = min(min(d_lows), 0)
        y_max = max(max(d_highs), 0)
        padding = (y_max - y_min) * 0.25 if y_max != y_min else 1.0
        ax.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()
        save_path = os.path.join(script_dir, "..", "..", f"{file_suffix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(script_dir, "results.json")
    plot_raw_and_deltas(json_path)