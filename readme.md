# FederatedLearning_DeepRVAT Project

This project demonstrates federated learning for rare variant association studies using a DeepRVAT-inspired 1D CNN model. It simulates multiple clients, each with their own synthetic genomic data, and shows how federated learning improves both global and local model performance while preserving data privacy.

## Project Structure

- **Main_DeepRVAT.py**: Main script to run the federated learning simulation and generate performance plots.
- **Client_DeepRVAT.py**: Defines the client-side model and local update logic.
- **Server_DeepRVAT.py**: Implements federated averaging and model size calculation.
- **Data_Generator.py**: Generates synthetic variant and phenotype data for each client.
- **Data_Utils_DeepRVAT.py**: Preprocessing utilities for filtering, padding, and tensor conversion.
- **plot_federated_improvement.py**: Generates professional plots to visualize global and local model performance.
- **requirements.txt**: Lists all required Python packages.
- **plots_deeprvat/**: Output directory for all generated plots.
- **global_model_metrics.csv, local_model_metrics.csv**: Metrics collected during the simulation.
- **log_deeprvat.txt**: Detailed log of the simulation.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main simulation:
   ```bash
   python Main_DeepRVAT.py
   ```
3. After the run, plots will be saved in the `plots_deeprvat/` directory.

## Key Features

- Simulates federated learning with multiple clients and synthetic data.
- Tracks and saves global and local model performance after each round.
- Automatically generates clear, plots for all key metrics.
- Demonstrates the benefit of federated learning over local-only training.

For more details, see comments in each script. 