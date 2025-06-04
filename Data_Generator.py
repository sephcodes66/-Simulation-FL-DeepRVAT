# Data_Generator.py
# Generates synthetic variant and phenotype data for federated learning experiments.

import pandas as pd
import numpy as np

ENABLE_PLOTTING = False

def generate_synthetic_data(num_individuals=10, num_variants_per_individual=50, client_id=None):
    """Generates synthetic variant data and phenotypes for a single client."""
    # Based on DeepRVAT documentation, it expects 33 annotations + MAF + CADD_PHRED
    required_annotations = [f"Annotation{i}" for i in range(1, 34)]
    
    data = []
    phenotypes = []

    for i in range(num_individuals):
        individual_variants = []
        for j in range(num_variants_per_individual):
            annotations = {ann: np.random.rand() for ann in required_annotations}
            
            # Ensure some variants meet filtering criteria
            maf = np.random.rand() * 0.009 if np.random.rand() < 0.2 else np.random.rand() * 0.49 + 0.01
            cadd_phred = np.random.rand() * 14.9 + 5.1 if np.random.rand() < 0.3 else np.random.rand() * 5
            
            variant_data = {
                'id': f"var_{client_id}_{i}_{j}",
                'maf': maf,
                'cadd_phred': cadd_phred,
                **annotations
            }
            individual_variants.append(variant_data)
        
        # Flatten variants for DataFrame and add individual ID
        for var_idx, var in enumerate(individual_variants):
            row = {'individual_id': f"ind_{client_id}_{i}", **var}
            data.append(row)
            
        phenotypes.append({'individual_id': f"ind_{client_id}_{i}", 'phenotype': np.random.rand() * 90 + 10})

    variants_df = pd.DataFrame(data)
    phenotypes_df = pd.DataFrame(phenotypes)
    
    return variants_df, phenotypes_df

if __name__ == "__main__":
    variants_df, phenotypes_df = generate_synthetic_data(client_id=1)
    print(f"Generated Variants for Client 1 ({len(variants_df)} rows):\n{variants_df.head()}")
    print(f"\nGenerated Phenotypes for Client 1 ({len(phenotypes_df)} rows):\n{phenotypes_df.head()}")