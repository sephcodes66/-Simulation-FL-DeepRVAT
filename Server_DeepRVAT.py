# Server_DeepRVAT.py
# Server-side logic for federated averaging and model size calculation in federated learning.

import torch
import numpy as np
from Data_Generator import generate_synthetic_data
        
def federated_averaging(global_model_weights, client_weight_deltas, client_data_sizes, server_lr):
    """Aggregates client weight deltas using Federated Averaging (FedAvg)."""
    total_data_points = sum(client_data_sizes)
    if not client_weight_deltas:
        return global_model_weights
    weighted_deltas_sum = [torch.zeros_like(delta_layer) for delta_layer in client_weight_deltas[0]]
    for i, deltas in enumerate(client_weight_deltas):
        weight = client_data_sizes[i] / total_data_points
        for layer_idx, delta_layer in enumerate(deltas):
            weighted_deltas_sum[layer_idx] += weight * delta_layer
    new_global_weights = [global_layer + server_lr * delta_sum_layer 
                         for global_layer, delta_sum_layer in zip(global_model_weights, weighted_deltas_sum)]
    return new_global_weights

def calculate_model_size_bytes(model_weights):
    """Calculates the size of model weights in bytes."""
    import sys
    total_bytes = 0
    for layer_weights in model_weights:
        arr = layer_weights.cpu().numpy()
        total_bytes += sys.getsizeof(arr)
    return total_bytes 