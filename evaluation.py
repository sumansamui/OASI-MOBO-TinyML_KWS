import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pymoo for metrics
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
BASE_DIR = "mobo_results"
METHODS = ['Random', 'LHS', 'Sobol', 'OASI']

# Load all evaluation logs
data_dict = {}
all_points = []

for method in METHODS:
    file_path = os.path.join(BASE_DIR, f"all_evaluation_mobo_{method}.xlsx")
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Convert Accuracy to an error rate so both objectives are minimized
        df['Error_Rate'] = 1.0 - df['Accuracy'] 
        data_dict[method] = df
        all_points.append(df[['Error_Rate', 'Size_MB']].values)
    else:
        print(f"Warning: Data for {method} not found at {file_path}")

# Combine all points to find global bounds for normalization
global_points = np.vstack(all_points)
min_vals = global_points.min(axis=0)
max_vals = global_points.max(axis=0)

def normalize(points):
    """Min-max normalization to scale objectives between 0 and 1."""
    return (points - min_vals) / (max_vals - min_vals + 1e-8)

# ==========================================
# 2. Compute the "True" Reference Front
# ==========================================
# Pool all normalized evaluated points across all initialization methods
global_points_norm = normalize(global_points)
true_front_indices = NonDominatedSorting().do(global_points_norm, only_non_dominated_front=True)
true_pareto_front = global_points_norm[true_front_indices]

# Set a reference point for HV slightly above the worst normalized values (1.0, 1.0)
ref_point = np.array([1.1, 1.1])
hv_calculator = HV(ref_point=ref_point)
gd_calculator = GD(true_pareto_front)

# ==========================================
# 3. Calculate Metrics Over Time (Iterations)
# ==========================================
hv_history = {method: [] for method in data_dict.keys()}
final_gd = {}
final_fronts_raw = {}

for method, df in data_dict.items():
    max_iter = df['Iteration'].max()
    method_hv = []
    
    # Track metrics cumulatively up to each iteration
    for i in range(max_iter + 1):
        # Subset data up to current iteration
        current_data = df[df['Iteration'] <= i][['Error_Rate', 'Size_MB']].values
        current_data_norm = normalize(current_data)
        
        # Find Pareto front up to this iteration
        front_idx = NonDominatedSorting().do(current_data_norm, only_non_dominated_front=True)
        current_front_norm = current_data_norm[front_idx]
        
        # Calculate HV
        method_hv.append(hv_calculator.do(current_front_norm))
        
        # If it's the final iteration, calculate GD and save the raw front for boxplots
        if i == max_iter:
            final_gd[method] = gd_calculator.do(current_front_norm)
            final_fronts_raw[method] = df[df['Iteration'] <= i].iloc[front_idx]
            
    hv_history[method] = method_hv

# Print Quantitative Results
print("\n" + "="*40)
print("FINAL MULTI-OBJECTIVE METRICS")
print("="*40)
print(f"{'Method':<10} | {'Final HV (Higher=Better)':<25} | {'GD (Lower=Better)':<20}")
print("-" * 60)
for method in data_dict.keys():
    print(f"{method:<10} | {hv_history[method][-1]:<25.4f} | {final_gd[method]:<20.6f}")

# ==========================================
# 4. Visualizations
# ==========================================
os.makedirs("evaluation_plots", exist_ok=True)
sns.set_theme(style="whitegrid")

# --- Plot 1: Hypervolume Improvement over Iterations ---
plt.figure(figsize=(10, 6))
for method, hv_vals in hv_history.items():
    # X-axis is iteration (0 is the initial sampling phase)
    plt.plot(range(len(hv_vals)), hv_vals, marker='o', label=method, linewidth=2)

plt.title("Hypervolume Improvement over MOBO Iterations", fontsize=14)
plt.xlabel("Iteration (0 = Initial Samples)", fontsize=12)
plt.ylabel("Hypervolume (Normalized Space)", fontsize=12)
plt.legend(title="Initialization Method")
plt.tight_layout()
plt.savefig("evaluation_plots/hv_improvement_time.png", dpi=300)
plt.close()

# --- Plot 2: Box Plots of Pareto Front Distributions ---
# Since we only have one run per method, plotting a boxplot of a single HV score isn't possible.
# Instead, we boxplot the distribution of the Objective Values (Accuracy & Size) *within* the final Pareto fronts.
# This shows the spread/diversity of the models each technique found.

pareto_df_list = []
for method, f_df in final_fronts_raw.items():
    f_df = f_df.copy()
    f_df['Method'] = method
    pareto_df_list.append(f_df)

combined_pareto_df = pd.concat(pareto_df_list, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy Boxplot
sns.boxplot(data=combined_pareto_df, x='Method', y='Accuracy', ax=axes[0], palette="Set2")
axes[0].set_title("Spread of Accuracy in Final Pareto Front", fontsize=13)
axes[0].set_ylabel("Validation Accuracy")

# Size Boxplot
sns.boxplot(data=combined_pareto_df, x='Method', y='Size_MB', ax=axes[1], palette="Set2")
axes[1].set_title("Spread of Model Size in Final Pareto Front", fontsize=13)
axes[1].set_ylabel("Model Size (MB)")

plt.suptitle("Diversity of Solutions Discovered by Initialization Techniques", fontsize=15)
plt.tight_layout()
plt.savefig("evaluation_plots/pareto_distribution_boxplots.png", dpi=300)
plt.close()

print("\n✔️ Evaluation complete. Plots saved to the 'evaluation_plots' directory.")