# Data_Utils_DeepRVAT.py
# Preprocessing utilities for DeepRVAT-inspired federated learning experiments.

import pandas as pd
import numpy as np
import torch

# Constants based on DeepRVAT documentation and data_generator.py
NUM_ANNOTATIONS = 33
EXPECTED_COLUMNS = [f"Annotation{i}" for i in range(1, NUM_ANNOTATIONS + 1)] + ['maf', 'cadd_phred']
NUM_FEATURES = len(EXPECTED_COLUMNS) # 33 annotations + MAF + CADD_PHRED

# Define a maximum number of variants per individual for padding/truncating
# This is important for creating fixed-size tensors for batching.
# Needs to be chosen based on typical output of data_generator and memory constraints.
# The generator makes 50, but many might be filtered. Let's aim for a smaller number.
MAX_VARIANTS_PER_INDIVIDUAL = 25 # Tunable parameter

def preprocess_synthetic_data(variants_df, phenotypes_df):
    """
    Preprocesses synthetic variant data and phenotypes for the DeepRVAT-inspired model.
    1. Filters variants based on MAF and CADD scores.
    2. Selects relevant features.
    3. Groups variants by individual.
    4. Pads/truncates variant sequences to a fixed length.
    5. Converts to PyTorch tensors.
    """
    # Filter variants (example criteria, similar to typical rare variant analysis)
    # DeepRVAT often focuses on rare (MAF < 0.01) and potentially deleterious (CADD > 15) variants
    filtered_variants_df = variants_df[
        (variants_df['maf'] < 0.01) & (variants_df['cadd_phred'] > 15.0)
    ]

    if filtered_variants_df.empty:
        print("Warning: No variants left after filtering. Check filter criteria or data generation.")
        # Return empty tensors with correct last dimension for features if no data
        return torch.empty(0, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES), torch.empty(0, 1)

    # Select features
    feature_df = filtered_variants_df[['individual_id'] + EXPECTED_COLUMNS]

    all_individual_data = []
    all_phenotypes = []

    # Ensure phenotypes_df is indexed by individual_id for quick lookup
    phenotypes_df = phenotypes_df.set_index('individual_id')

    for individual_id, group in feature_df.groupby('individual_id'):
        if individual_id not in phenotypes_df.index:
            # print(f"Warning: Individual {individual_id} found in variants but not in phenotypes. Skipping.")
            continue

        # Extract features, drop individual_id for the numerical data
        variants_data = group[EXPECTED_COLUMNS].values.astype(np.float32)

        # Pad or truncate variants
        if variants_data.shape[0] > MAX_VARIANTS_PER_INDIVIDUAL:
            # Truncate (e.g., take the first MAX_VARIANTS_PER_INDIVIDUAL)
            # More sophisticated selection could be implemented (e.g., by CADD score)
            padded_variants = variants_data[:MAX_VARIANTS_PER_INDIVIDUAL, :]
        elif variants_data.shape[0] < MAX_VARIANTS_PER_INDIVIDUAL:
            # Pad with zeros
            padding = np.zeros((MAX_VARIANTS_PER_INDIVIDUAL - variants_data.shape[0], NUM_FEATURES), dtype=np.float32)
            padded_variants = np.vstack((variants_data, padding))
        else:
            padded_variants = variants_data

        all_individual_data.append(padded_variants)
        all_phenotypes.append(phenotypes_df.loc[individual_id, 'phenotype'])

    if not all_individual_data: # Check if list is empty after processing all groups
        print("Warning: No individuals with valid data after processing. Returning empty tensors.")
        return torch.empty(0, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES), torch.empty(0, 1)

    # Convert to PyTorch tensors
    # X shape: (num_individuals, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES)
    X_tensor = torch.tensor(np.array(all_individual_data), dtype=torch.float32)
    # y shape: (num_individuals, 1) for regression
    y_tensor = torch.tensor(np.array(all_phenotypes), dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor

if __name__ == '__main__':
    # Example Usage with your data_generator
    from data_generator import generate_synthetic_data

    print("Generating synthetic data for testing preprocessing...")
    raw_variants, raw_phenotypes = generate_synthetic_data(num_individuals=5, num_variants_per_individual=50, client_id='test_client')
    print(f"Raw variants: {raw_variants.shape}, Raw phenotypes: {raw_phenotypes.shape}")
    print(raw_variants.head())
    print(raw_phenotypes.head())

    print(f"\nPreprocessing with MAX_VARIANTS_PER_INDIVIDUAL={MAX_VARIANTS_PER_INDIVIDUAL}, NUM_FEATURES={NUM_FEATURES}")
    X, y = preprocess_synthetic_data(raw_variants, raw_phenotypes)

    if X.nelement() > 0 and y.nelement() > 0:
        print(f"\nProcessed X shape: {X.shape}") # Expected: (num_individuals_after_filtering, MAX_VARIANTS_PER_INDIVIDUAL, NUM_FEATURES)
        print(f"Processed y shape: {y.shape}")   # Expected: (num_individuals_after_filtering, 1)
        print("\nFirst processed individual's data (X[0]):")
        print(X[0])
        print("\nFirst processed individual's phenotype (y[0]):")
        print(y[0])
    else:
        print("\nNo data available after preprocessing in the test run.")