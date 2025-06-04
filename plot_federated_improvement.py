# plot_federated_improvement.py
# This script generates plots to visualize global and local model performance in federated learning experiments.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs('plots_deeprvat', exist_ok=True)

# --- Load Data ---
global_df = pd.read_csv('global_model_metrics.csv')
local_df = pd.read_csv('local_model_metrics.csv')

# --- Parse Local-Only Baseline from log file ---
log_file = 'log_deeprvat.txt'
local_only_mse_init = []
local_only_r2_init = []
local_only_mse_final = []
local_only_r2_final = []
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            m = re.match(r"\s*Client (\d+) Local-Only Baseline: Initial MSE=([\d\.eE+-]+), R2=([\d\.eE+-]+); Final MSE=([\d\.eE+-]+), R2=([\d\.eE+-]+)", line)
            if m:
                local_only_mse_init.append(float(m.group(2)))
                local_only_r2_init.append(float(m.group(3)))
                local_only_mse_final.append(float(m.group(4)))
                local_only_r2_final.append(float(m.group(5)))

# --- Extract Metrics ---
# Initial global
initial_global = global_df.iloc[0]
# Final federated global
final_global = global_df.iloc[-1]
# Local-only (averaged)
local_only_mse_init_avg = np.mean(local_only_mse_init) if local_only_mse_init else np.nan
local_only_r2_init_avg = np.mean(local_only_r2_init) if local_only_r2_init else np.nan
local_only_mse_final_avg = np.mean(local_only_mse_final) if local_only_mse_final else np.nan
local_only_r2_final_avg = np.mean(local_only_r2_final) if local_only_r2_final else np.nan
# Federated local (averaged, last round)
final_local_df = local_df[local_df['round'] == local_df['round'].max()]
federated_local_mse_avg = final_local_df['mse'].mean()
federated_local_r2_avg = final_local_df['r2'].mean()

# --- Bar Plot: MSE ---
labels = ['Initial Global', 'Local-Only', 'Federated Global', 'Federated Local']
mse_values = [initial_global['mse'], local_only_mse_final_avg, final_global['mse'], federated_local_mse_avg]
colors = ['#bdbdbd', '#ffb347', '#77dd77', '#6ec6ff']
plt.figure(figsize=(10,6))
plt.bar(labels, mse_values, color=colors, edgecolor='black')
plt.ylabel('MSE (Lower is better)', fontsize=14)
plt.title('Model MSE Comparison', fontsize=16)
for i, v in enumerate(mse_values):
    plt.text(i, v + max(mse_values)*0.02, f'{v:.0f}', ha='center', va='bottom', fontsize=12)
plt.ylim(0, max(mse_values)*1.15)
plt.tight_layout()
plt.savefig('plots_deeprvat/model_mse_comparison.png')
plt.close()

"""# --- Bar Plot: R² ---
r2_values = [initial_global['r2'], local_only_r2_final_avg, final_global['r2'], federated_local_r2_avg]
plt.figure(figsize=(10,6))
plt.bar(labels, r2_values, color=colors, edgecolor='black')
plt.ylabel('R² (Higher is better)', fontsize=14)
plt.title('Model R² Comparison', fontsize=16)
for i, v in enumerate(r2_values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=12)
plt.ylim(min(r2_values)-0.2, max(r2_values)*1.15)
plt.tight_layout()
plt.savefig('plots_deeprvat/model_r2_comparison.png')
plt.close()
"""
# --- Line Plot: Global MSE vs. Round ---
plt.figure(figsize=(10,6))
plt.plot(global_df['round'], global_df['mse'], marker='o', color='#0072B2')
plt.xlabel('Round', fontsize=14)
plt.ylabel('Global Model MSE', fontsize=14)
plt.title('Global Model MSE per Round', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots_deeprvat/global_model_mse_per_round.png')
plt.close()

# --- Line Plot: Global R² vs. Round ---
plt.figure(figsize=(10,6))
plt.plot(global_df['round'], global_df['r2'], marker='o', color='#D55E00')
plt.xlabel('Round', fontsize=14)
plt.ylabel('Global Model R²', fontsize=14)
plt.title('Global Model R² per Round', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plots_deeprvat/global_model_r2_per_round.png')
plt.close()

"""
# --- Boxplot: Local Model MSE per Round ---
plt.figure(figsize=(12,6))
local_mse_data = [local_df[local_df['round'] == r]['mse'].values for r in sorted(local_df['round'].unique())]
plt.boxplot(local_mse_data, patch_artist=True, showmeans=True)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Local Model MSE', fontsize=14)
plt.title('Local Model MSE Distribution per Round', fontsize=16)
plt.xticks(ticks=np.arange(1, len(local_mse_data)+1), labels=sorted(local_df['round'].unique()))
plt.tight_layout()
plt.savefig('plots_deeprvat/local_model_mse_per_round.png')
plt.close()

# --- Boxplot: Local Model R² per Round ---
plt.figure(figsize=(12,6))
local_r2_data = [local_df[local_df['round'] == r]['r2'].values for r in sorted(local_df['round'].unique())]
plt.boxplot(local_r2_data, patch_artist=True, showmeans=True)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Local Model R²', fontsize=14)
plt.title('Local Model R² Distribution per Round', fontsize=16)
plt.xticks(ticks=np.arange(1, len(local_r2_data)+1), labels=sorted(local_df['round'].unique()))
plt.tight_layout()
plt.savefig('plots_deeprvat/local_model_r2_per_round.png')
plt.close()
"""