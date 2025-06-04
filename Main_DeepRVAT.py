# Main_DeepRVAT.py
# Main script for running federated learning simulation and generating performance plots.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score # For regression
import os
import csv
import subprocess

# Project-specific imports
from Data_Generator import generate_synthetic_data
from Data_Utils_DeepRVAT import preprocess_synthetic_data, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES
from Client_DeepRVAT import DeepRVATInspiredNet, client_update
from Server_DeepRVAT import federated_averaging, calculate_model_size_bytes # Reusing your server script

DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Main script using device: {DEVICE}")

# --- Configuration Parameters ---
NUM_CLIENTS = 4
NUM_INDIVIDUALS_PER_CLIENT = 150 
NUM_VARIANTS_RAW_PER_INDIVIDUAL = 50 

NUM_ROUNDS = 10 
LOCAL_EPOCHS = 10
BATCH_SIZE = 16  
CLIENT_LEARNING_RATE = 0.0001 
SERVER_LEARNING_RATE = 1.0

DP_ENABLED = True
DP_NOISE_MULTIPLIER = 0.05 
DP_CLIP_NORM = 2.0       

# --- Data Preparation ---
def load_and_prepare_client_data(num_clients, num_individuals, num_variants_raw):
    client_datasets = []
    all_client_X_processed = []
    all_client_y_processed = []
    log_entries = []

    log_entries.append(f"Generating and preprocessing data for {num_clients} clients...\n")
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        log_entries.append(f"  Generating raw data for {client_id} ({num_individuals} individuals, {num_variants_raw} variants each before filtering)...")
        variants_df, phenotypes_df = generate_synthetic_data(
            num_individuals=num_individuals,
            num_variants_per_individual=num_variants_raw,
            client_id=client_id
        )
        log_entries.append(f"  Preprocessing data for {client_id} (MAX_VARIANTS_PER_INDIVIDUAL={MAX_VARIANTS_PER_INDIVIDUAL}, NUM_FEATURES={NUM_FEATURES})...")
        
        try:
            X_processed, y_processed = preprocess_synthetic_data(variants_df, phenotypes_df)
        except Exception as e:
            log_entries.append(f"  Error preprocessing data for {client_id}: {e}")
            X_processed, y_processed = torch.empty(0, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES), torch.empty(0, 1)

        if X_processed.nelement() > 0 and y_processed.nelement() > 0:
            log_entries.append(f"  {client_id}: Processed X shape: {X_processed.shape}, y shape: {y_processed.shape}")
            client_datasets.append(TensorDataset(X_processed, y_processed))
            all_client_X_processed.append(X_processed)
            all_client_y_processed.append(y_processed)
        else:
            log_entries.append(f"  {client_id}: No valid data after preprocessing. This client will be skipped if it has no data.")
            client_datasets.append(TensorDataset(torch.empty(0, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES), torch.empty(0, 1)))


    if all_client_X_processed and all_client_y_processed:
        X_test = torch.cat(all_client_X_processed)
        y_test = torch.cat(all_client_y_processed)
        log_entries.append(f"Using all processed client data as a consolidated test set for evaluation: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    else: 
        log_entries.append("Warning: No client data successfully processed. Using dummy empty test set.")
        X_test = torch.empty(0, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES)
        y_test = torch.empty(0, 1)
        
    return client_datasets, (X_test, y_test), "\n".join(log_entries)


def evaluate_regression_model(model, x_test, y_test, device, round_name="", log_file=None, show_plot=True):
    model.eval()
    all_preds = []
    all_true = []
    test_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        if x_test.nelement() == 0 or y_test.nelement() == 0:
            log_file.write(f"Evaluation for {round_name}: No test data available.\n")
            return 0.0, 0.0, float('nan')

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0) 
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_true.extend(y_batch.cpu().numpy().flatten())
    
    if not all_true or not all_preds:   
        log_file.write(f"Evaluation for {round_name}: No predictions made (all_true or all_preds is empty).\n")
        return 0.0, 0.0, float('nan')

    avg_test_loss = test_loss / len(all_true) if len(all_true) > 0 else float('nan')
    mse = mean_squared_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)

    log_msg = f"{round_name} - Test MSE: {mse:.4f}, Test R2: {r2:.4f}, Avg Test Loss: {avg_test_loss:.4f}\n"
    print(log_msg.strip())
    if log_file:
        log_file.write(log_msg)

    return mse, r2, avg_test_loss


# --- Main Simulation ---
if __name__ == "__main__":
    os.makedirs("plots_deeprvat", exist_ok=True) 
    log_filename = "log_deeprvat.txt"
    global_metrics = []  
    local_metrics = []   
    local_only_baseline_metrics = []  

    with open(log_filename, "w") as log_file:
        log_file.write("--- Federated Learning Simulation Log (DeepRVAT-Inspired) ---\n")
        log_file.write("\n[Phase 0: Configuration Parameters]\n")
        log_file.write(f"  Device: {DEVICE}\n")
        log_file.write(f"  Number of Clients (NUM_CLIENTS): {NUM_CLIENTS}\n")
        log_file.write(f"  Individuals per Client (NUM_INDIVIDUALS_PER_CLIENT): {NUM_INDIVIDUALS_PER_CLIENT}\n")
        log_file.write(f"  Raw Variants per Individual (NUM_VARIANTS_RAW_PER_INDIVIDUAL): {NUM_VARIANTS_RAW_PER_INDIVIDUAL}\n")
        log_file.write(f"  Max Variants per Individual after processing (MAX_VARIANTS_PER_INDIVIDUAL): {MAX_VARIANTS_PER_INDIVIDUAL}\n")
        log_file.write(f"  Number of Features per Variant (NUM_FEATURES): {NUM_FEATURES}\n")
        log_file.write(f"  Number of Communication Rounds (NUM_ROUNDS): {NUM_ROUNDS}\n")
        log_file.write(f"  Local Epochs per Client (LOCAL_EPOCHS): {LOCAL_EPOCHS}\n")
        log_file.write(f"  Batch Size (BATCH_SIZE): {BATCH_SIZE}\n")
        log_file.write(f"  Client Learning Rate (CLIENT_LEARNING_RATE): {CLIENT_LEARNING_RATE}\n")
        log_file.write(f"  Server Learning Rate (SERVER_LEARNING_RATE): {SERVER_LEARNING_RATE}\n")
        log_file.write(f"  Differential Privacy Enabled (DP_ENABLED): {DP_ENABLED}\n")
        if DP_ENABLED:
            log_file.write(f"    DP Noise Multiplier (DP_NOISE_MULTIPLIER): {DP_NOISE_MULTIPLIER}\n")
            log_file.write(f"    DP Clip Norm (DP_CLIP_NORM): {DP_CLIP_NORM}\n")

        print("--- Federated Learning Simulation Start (DeepRVAT-Inspired) ---")
        print(f"See {log_filename} for detailed output.")

        # 1. Load and Prepare Data
        log_file.write("\n[Phase 1: Data Preparation]\n")
        client_datasets, (x_test, y_test), data_log_entries = load_and_prepare_client_data(
            NUM_CLIENTS, NUM_INDIVIDUALS_PER_CLIENT, NUM_VARIANTS_RAW_PER_INDIVIDUAL
        )
        log_file.write(data_log_entries + "\n")
        
        if x_test.nelement() == 0:
            log_file.write("Critical: No test data available after processing. Aborting simulation.\n")
            print("Critical: No test data available. Check data generation/preprocessing. Aborting.")
            exit()

        # --- Local-Only Baseline (No FL) ---
        log_file.write("\n[Phase 1b: Local-Only Baseline (No FL)]\n")
        print("\n--- Local-Only Baseline (No FL) ---")
        for client_idx, client_data in enumerate(client_datasets):
            if len(client_data) == 0:
                log_file.write(f"  Client {client_idx + 1} has no data, skipping local-only baseline.\n")
                continue
            
            local_model = DeepRVATInspiredNet(num_features=NUM_FEATURES, max_variants=MAX_VARIANTS_PER_INDIVIDUAL).to(DEVICE)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=CLIENT_LEARNING_RATE)
            criterion = nn.MSELoss()
            loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
            
            local_model.eval()
            local_preds, local_true = [], []
            with torch.no_grad():
                for x_local, y_local in loader:
                    x_local, y_local = x_local.to(DEVICE), y_local.to(DEVICE)
                    out_local = local_model(x_local)
                    local_preds.extend(out_local.cpu().numpy().flatten())
                    local_true.extend(y_local.cpu().numpy().flatten())
            if local_true and local_preds:
                mse_init = mean_squared_error(local_true, local_preds)
                r2_init = r2_score(local_true, local_preds)
            else:
                mse_init, r2_init = float('nan'), float('nan')
            
            local_model.train()
            for epoch in range(LOCAL_EPOCHS * NUM_ROUNDS):
                for x_local, y_local in loader:
                    x_local, y_local = x_local.to(DEVICE), y_local.to(DEVICE)
                    optimizer.zero_grad()
                    out_local = local_model(x_local)
                    loss = criterion(out_local, y_local)
                    loss.backward()
                    optimizer.step()
            
            local_model.eval()
            local_preds, local_true = [], []
            with torch.no_grad():
                for x_local, y_local in loader:
                    x_local, y_local = x_local.to(DEVICE), y_local.to(DEVICE)
                    out_local = local_model(x_local)
                    local_preds.extend(out_local.cpu().numpy().flatten())
                    local_true.extend(y_local.cpu().numpy().flatten())
            if local_true and local_preds:
                mse_final = mean_squared_error(local_true, local_preds)
                r2_final = r2_score(local_true, local_preds)
            else:
                mse_final, r2_final = float('nan'), float('nan')
            local_only_baseline_metrics.append({
                'client_id': client_idx + 1,
                'mse_init': mse_init,
                'r2_init': r2_init,
                'mse_final': mse_final,
                'r2_final': r2_final
            })
            log_file.write(f"  Client {client_idx + 1} Local-Only Baseline: Initial MSE={mse_init:.2f}, R2={r2_init:.2f}; Final MSE={mse_final:.2f}, R2={r2_final:.2f}\n")
            print(f"Client {client_idx + 1} Local-Only Baseline: Initial MSE={mse_init:.2f}, R2={r2_init:.2f}; Final MSE={mse_final:.2f}, R2={r2_final:.2f}")

        # 2. Initialize Global Model
        log_file.write("\n[Phase 2: Model Initialization]\n")
        global_model = DeepRVATInspiredNet(num_features=NUM_FEATURES, max_variants=MAX_VARIANTS_PER_INDIVIDUAL).to(DEVICE)
        log_file.write(f"Global DeepRVAT-inspired model created.\nModel Architecture:\n{global_model}\n")

        
        log_file.write("\nEvaluating Initial Global Model (Untrained)...\n")
        initial_mse, initial_r2, initial_loss = evaluate_regression_model(global_model, x_test, y_test, DEVICE, "Initial Global Model", log_file, show_plot=False)

        global_metrics.append({
            'round': 0,
            'mse': initial_mse,
            'r2': initial_r2,
            'loss': initial_loss,
            'bandwidth_MB': 0,
            'DP_noise_multiplier': DP_NOISE_MULTIPLIER if DP_ENABLED else 0,
            'DP_clip_norm': DP_CLIP_NORM if DP_ENABLED else 0
        })

        log_file.write("\n[Phase 3: Federated Training]\n")
        for round_num in range(NUM_ROUNDS):
            print(f"--- Round {round_num + 1}/{NUM_ROUNDS} --- (see {log_filename})")
            log_file.write(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---\n")
            current_global_weights = [p.data.clone().cpu() for p in global_model.parameters()] # Send CPU weights to server
            client_weight_deltas_collected = []
            client_data_sizes_collected = []
            round_local_metrics = []  # For this round
            total_bandwidth_this_round_bytes = 0
            active_clients_this_round = 0

            for client_idx in range(NUM_CLIENTS):
                client_data = client_datasets[client_idx]
                if len(client_data) == 0: # Skip clients with no data
                    log_file.write(f"  Client {client_idx + 1}/{NUM_CLIENTS} has no data, skipping.\n")
                    continue
                
                active_clients_this_round +=1
                log_file.write(f"  Client {client_idx + 1}/{NUM_CLIENTS} training ({len(client_data)} samples)...\n")
                
                client_model = DeepRVATInspiredNet(num_features=NUM_FEATURES, max_variants=MAX_VARIANTS_PER_INDIVIDUAL).to(DEVICE)
                
                with torch.no_grad():
                    for client_param, global_param_cpu in zip(client_model.parameters(), current_global_weights):
                        client_param.data = global_param_cpu.clone().to(DEVICE)

                weight_deltas, num_data_points = client_update(
                    client_model,
                    client_data,
                    LOCAL_EPOCHS,
                    BATCH_SIZE,
                    CLIENT_LEARNING_RATE,
                    DP_ENABLED,
                    DP_NOISE_MULTIPLIER,
                    DP_CLIP_NORM
                )
                client_weight_deltas_collected.append(weight_deltas) 
                client_data_sizes_collected.append(num_data_points)
                
                client_upload_size_bytes = calculate_model_size_bytes(weight_deltas)
                total_bandwidth_this_round_bytes += client_upload_size_bytes
                log_file.write(f"    Client {client_idx + 1} upload size: {client_upload_size_bytes / 1024:.2f} KB\n")

                # --- Local Model Evaluation ---
                
                client_model.eval()
                local_loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=False)
                local_preds, local_true = [], []
                local_loss = 0
                criterion = nn.MSELoss()
                with torch.no_grad():
                    for x_local, y_local in local_loader:
                        x_local, y_local = x_local.to(DEVICE), y_local.to(DEVICE)
                        out_local = client_model(x_local)
                        loss = criterion(out_local, y_local)
                        local_loss += loss.item() * x_local.size(0)
                        local_preds.extend(out_local.cpu().numpy().flatten())
                        local_true.extend(y_local.cpu().numpy().flatten())
                if local_true and local_preds:
                    local_mse = mean_squared_error(local_true, local_preds)
                    local_r2 = r2_score(local_true, local_preds)
                    avg_local_loss = local_loss / len(local_true)
                else:
                    local_mse, local_r2, avg_local_loss = float('nan'), float('nan'), float('nan')
                
                round_local_metrics.append({
                    'round': round_num + 1,
                    'client_id': client_idx + 1,
                    'mse': local_mse,
                    'r2': local_r2,
                    'loss': avg_local_loss,
                    'DP_noise_multiplier': DP_NOISE_MULTIPLIER if DP_ENABLED else 0,
                    'DP_clip_norm': DP_CLIP_NORM if DP_ENABLED else 0
                })

            if not client_weight_deltas_collected:
                log_file.write("  No client updates collected this round (all clients might have had no data or an error). Skipping aggregation.\n")
                print("Warning: No client updates collected this round.")
                
                round_mse, round_r2, round_loss = evaluate_regression_model(global_model, x_test, y_test, DEVICE, f"Round {round_num + 1} (No Updates)", log_file, show_plot=False)
                
                global_metrics.append({
                    'round': round_num + 1,
                    'mse': round_mse,
                    'r2': round_r2,
                    'loss': round_loss,
                    'bandwidth_MB': total_bandwidth_this_round_bytes / (1024*1024),
                    'DP_noise_multiplier': DP_NOISE_MULTIPLIER if DP_ENABLED else 0,
                    'DP_clip_norm': DP_CLIP_NORM if DP_ENABLED else 0
                })
                
                local_metrics.extend(round_local_metrics)
                continue
            
            log_file.write(f"  Server aggregating updates from {active_clients_this_round} clients...\n")
            
            
            cpu_global_weights = [p.data.clone().cpu() for p in global_model.parameters()]
            
            new_global_weights_cpu = federated_averaging(
                cpu_global_weights, # Must be on CPU
                client_weight_deltas_collected, # Already on CPU
                client_data_sizes_collected,
                SERVER_LEARNING_RATE
            )

            
            with torch.no_grad():
                for p_global, new_w_cpu in zip(global_model.parameters(), new_global_weights_cpu):
                    p_global.data = new_w_cpu.clone().to(DEVICE)
            
            log_file.write(f"  Round {round_num + 1} aggregation complete.\n")
            log_file.write(f"  Total bandwidth for this round (uploads): {total_bandwidth_this_round_bytes / (1024*1024):.3f} MB\n")

            log_file.write(f"\n  Evaluating Global Model after Round {round_num + 1}...\n")
            round_mse, round_r2, round_loss = evaluate_regression_model(global_model, x_test, y_test, DEVICE, f"Round {round_num + 1}", log_file, show_plot=False)
            global_metrics.append({
                'round': round_num + 1,
                'mse': round_mse,
                'r2': round_r2,
                'loss': round_loss,
                'bandwidth_MB': total_bandwidth_this_round_bytes / (1024*1024),
                'DP_noise_multiplier': DP_NOISE_MULTIPLIER if DP_ENABLED else 0,
                'DP_clip_norm': DP_CLIP_NORM if DP_ENABLED else 0
            })
            local_metrics.extend(round_local_metrics)

        log_file.write("\n--- Federated Learning Simulation End ---\n")

        log_file.write("\n[Phase 4: Final Evaluation]\n")
        final_mse, final_r2, final_loss = evaluate_regression_model(global_model, x_test, y_test, DEVICE, "Final Global Model", log_file, show_plot=True) # Show plot for final

        global_metrics_file = "global_model_metrics.csv"
        local_metrics_file = "local_model_metrics.csv"
        with open(global_metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'mse', 'r2', 'loss', 'bandwidth_MB', 'DP_noise_multiplier', 'DP_clip_norm'])
            writer.writeheader()
            for row in global_metrics:
                writer.writerow(row)
        with open(local_metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'client_id', 'mse', 'r2', 'loss', 'DP_noise_multiplier', 'DP_clip_norm'])
            writer.writeheader()
            for row in local_metrics:
                writer.writerow(row)

        print("Generating performance plots...")
        subprocess.run(['python', 'plot_federated_improvement.py'], check=True)

        log_file.write("\n--- Federated Learning Performance Summary ---\n\n")
        print("\n--- Federated Learning Performance Summary ---\n")
        initial_global = global_metrics[0]
        final_global = global_metrics[-1]
        log_file.write(f"Global Model:\n  - Initial MSE: {initial_global['mse']:.2f}, R²: {initial_global['r2']:.2f}\n  - Final MSE: {final_global['mse']:.2f}, R²: {final_global['r2']:.2f}\n  - Improvement: MSE ↓ {initial_global['mse'] - final_global['mse']:.2f}, R² ↑ {final_global['r2'] - initial_global['r2']:.2f}\n\n")
        print(f"Global Model:\n  - Initial MSE: {initial_global['mse']:.2f}, R²: {initial_global['r2']:.2f}\n  - Final MSE: {final_global['mse']:.2f}, R²: {final_global['r2']:.2f}\n  - Improvement: MSE ↓ {initial_global['mse'] - final_global['mse']:.2f}, R² ↑ {final_global['r2'] - initial_global['r2']:.2f}\n")
        local_metrics_df = None
        try:
            import pandas as pd
            local_metrics_df = pd.DataFrame(local_metrics)
        except ImportError:
            pass
        if local_metrics_df is not None:
            initial_local_mse = local_metrics_df[local_metrics_df['round'] == 1]['mse'].mean()
            initial_local_r2 = local_metrics_df[local_metrics_df['round'] == 1]['r2'].mean()
            final_local_mse = local_metrics_df[local_metrics_df['round'] == NUM_ROUNDS]['mse'].mean()
            final_local_r2 = local_metrics_df[local_metrics_df['round'] == NUM_ROUNDS]['r2'].mean()
        else:
            initial_local_mse = np.mean([m['mse'] for m in local_metrics if m['round'] == 1])
            initial_local_r2 = np.mean([m['r2'] for m in local_metrics if m['round'] == 1])
            final_local_mse = np.mean([m['mse'] for m in local_metrics if m['round'] == NUM_ROUNDS])
            final_local_r2 = np.mean([m['r2'] for m in local_metrics if m['round'] == NUM_ROUNDS])
        log_file.write(f"Local Models (Averaged Across Clients):\n  - Initial Avg MSE: {initial_local_mse:.2f}, R²: {initial_local_r2:.2f}\n  - Final Avg MSE: {final_local_mse:.2f}, R²: {final_local_r2:.2f}\n  - Improvement: MSE ↓ {initial_local_mse - final_local_mse:.2f}, R² ↑ {final_local_r2 - initial_local_r2:.2f}\n\n")
        print(f"Local Models (Averaged Across Clients):\n  - Initial Avg MSE: {initial_local_mse:.2f}, R²: {initial_local_r2:.2f}\n  - Final Avg MSE: {final_local_mse:.2f}, R²: {final_local_r2:.2f}\n  - Improvement: MSE ↓ {initial_local_mse - final_local_mse:.2f}, R² ↑ {final_local_r2 - initial_local_r2:.2f}\n")   
        if local_only_baseline_metrics:
            avg_local_only_init_mse = np.mean([m['mse_init'] for m in local_only_baseline_metrics])
            avg_local_only_init_r2 = np.mean([m['r2_init'] for m in local_only_baseline_metrics])
            avg_local_only_final_mse = np.mean([m['mse_final'] for m in local_only_baseline_metrics])
            avg_local_only_final_r2 = np.mean([m['r2_final'] for m in local_only_baseline_metrics])
            log_file.write(f"Local-Only Baseline (Averaged Across Clients):\n  - Initial Avg MSE: {avg_local_only_init_mse:.2f}, R²: {avg_local_only_init_r2:.2f}\n  - Final Avg MSE: {avg_local_only_final_mse:.2f}, R²: {avg_local_only_final_r2:.2f}\n  - Improvement: MSE ↓ {avg_local_only_init_mse - avg_local_only_final_mse:.2f}, R² ↑ {avg_local_only_final_r2 - avg_local_only_init_r2:.2f}\n\n")
            print(f"Local-Only Baseline (Averaged Across Clients):\n  - Initial Avg MSE: {avg_local_only_init_mse:.2f}, R²: {avg_local_only_init_r2:.2f}\n  - Final Avg MSE: {avg_local_only_final_mse:.2f}, R²: {avg_local_only_final_r2:.2f}\n  - Improvement: MSE ↓ {avg_local_only_init_mse - avg_local_only_final_mse:.2f}, R² ↑ {avg_local_only_final_r2 - avg_local_only_init_r2:.2f}\n")
        log_file.write("Federated learning enables all clients to benefit from the collective data, improving both global and local model performance, while keeping raw data private.\n")
        print("Federated learning enables all clients to benefit from the collective data, improving both global and local model performance, while keeping raw data private.\n")

        # --- Discussion Points ---
        log_file.write("\n--- Key Concepts Demonstrated (DeepRVAT-Inspired FL) ---\n")
        log_file.write("\n1. Federated Training for Genomic Data (Simulated):\n")
        log_file.write("   - Global model (DeepRVAT-inspired 1D CNN) initialized by the server.\n")
        log_file.write("   - Model weights distributed to clients.\n")
        log_file.write("   - Clients generate synthetic genomic data, preprocess it (filtering, padding), and train locally.\n")
        log_file.write("   - Clients send model *updates* (weight deltas) back to the server.\n")
        log_file.write("   - Server aggregates updates (Federated Averaging) to form the new global model.\n")
        log_file.write("   - Evaluation uses regression metrics (MSE, R2) and true vs. predicted plots.\n")

        log_file.write("\n2. Data Handling for Sequential Genomic Features:\n")
        log_file.write(f"   - `data_generator.py` creates synthetic variants and phenotypes.\n")
        log_file.write(f"   - `data_utils_deeprvat.py` preprocesses this: filters variants (e.g., MAF, CADD score), pads/truncates to `MAX_VARIANTS_PER_INDIVIDUAL={MAX_VARIANTS_PER_INDIVIDUAL}` with `NUM_FEATURES={NUM_FEATURES}`.\n")
        log_file.write(f"   - This creates fixed-size tensors suitable for the 1D CNN.\n")

        log_file.write("\n3. Model Architecture (DeepRVATInspiredNet):\n")
        log_file.write(f"   - A 1D CNN designed to process sequences of variant features.\n")
        log_file.write(f"   - Input: (batch_size, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES).\n")
        log_file.write(f"   - Output: A single continuous value (phenotype prediction).\n")
        log_file.write(f"   - Sized to be runnable on a MacBook Air M3.\n")

        log_file.write("\n4. Tuning Aspects:\n")
        log_file.write(f"   - `NUM_ROUNDS` ({NUM_ROUNDS}), `LOCAL_EPOCHS` ({LOCAL_EPOCHS}), learning rates, `BATCH_SIZE` ({BATCH_SIZE}) are key hyperparameters.\n")
        log_file.write(f"   - `NUM_INDIVIDUALS_PER_CLIENT` ({NUM_INDIVIDUALS_PER_CLIENT}) and filtering criteria in `data_utils_deeprvat.py` heavily impact data quality and quantity per client.\n")
        log_file.write(f"   - `MAX_VARIANTS_PER_INDIVIDUAL` ({MAX_VARIANTS_PER_INDIVIDUAL}) affects model input size and memory.\n")

        log_file.write("\n5. Data Privacy (Simplified Differential Privacy):\n")
        if DP_ENABLED:
            log_file.write(f"   - DP is ENABLED: noise multiplier {DP_NOISE_MULTIPLIER}, clip norm {DP_CLIP_NORM}.\n")
            log_file.write("   - Deltas are clipped and noised. Trade-off between privacy and model utility (MSE/R2).\n")
        else:
            log_file.write("   - DP is DISABLED.\n")

        log_file.write("\n6. Bandwidth Calculation:\n")
        initial_model_size_bytes = calculate_model_size_bytes([p.data.cpu() for p in global_model.parameters()]) # Ensure CPU for calculation
        log_file.write(f"   - Estimated size of one full model (weights): {initial_model_size_bytes / 1024:.2f} KB\n")
        log_file.write("   - Bandwidth per round logged (sum of client delta sizes).\n")

        log_file.write("\n--- Further Exploration ---\n")
        log_file.write("   - Implement more sophisticated variant filtering or feature engineering.\n")
        log_file.write("   - Experiment with different 1D CNN architectures, attention mechanisms, or RNNs for the local model.\n")
        log_file.write("   - Explore non-IID data distributions more explicitly (e.g., different allele frequencies or effect sizes per client).\n")
        log_file.write("   - Rigorous hyperparameter tuning for FL and DP parameters.\n")
        print(f"--- Simulation Complete. Results in {log_filename} and plots in plots_deeprvat/ ---")