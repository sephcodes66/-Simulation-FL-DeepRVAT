# Client_DeepRVAT.py
# Defines the client-side model and local update logic for federated learning.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Client-side using device: {DEVICE}")


# --- Client-Side: Model and Local Update ---
class DeepRVATInspiredNet(nn.Module):
    """1D CNN model inspired by DeepRVAT for variant data regression."""
    def __init__(self, num_features, max_variants, output_dim=1):
        super().__init__()
        # Input shape: (batch_size, max_variants, num_features)

        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=4) 
        flattened_size = 64 * 4 

        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.3) 
        self.fc2 = nn.Linear(128, output_dim) 

    def forward(self, x):
        # x shape: (batch_size, max_variants, num_features)
        x = x.permute(0, 2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        return x

def client_update(model, client_dataset, local_epochs, batch_size, learning_rate,
                  dp_enabled, noise_multiplier, clip_norm):
    """Simulates a client's local training process for regression."""
    model = model.to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 
    loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    initial_weights = [p.data.clone() for p in model.parameters()]

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x, y in loader:
            if x.nelement() == 0 or y.nelement() == 0: # Skip empty batches if any
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        # print(f"Client Epoch {epoch+1}/{local_epochs}, Avg Loss: {epoch_loss/num_batches if num_batches > 0 else float('nan')}")


    updated_weights = [p.data.clone() for p in model.parameters()]
    weight_deltas = [u - i for u, i in zip(updated_weights, initial_weights)]

    if dp_enabled:
        noisy_deltas = []
        for delta in weight_deltas:
            clipped_delta = delta.clone()
           
            norm = torch.norm(clipped_delta.flatten(), p=2) # L2 norm
            if norm > clip_norm and norm > 0: # Add norm > 0 to avoid division by zero if delta is all zeros
                clipped_delta = clipped_delta * (clip_norm / norm)
            
            actual_noise_std = max(clip_norm * noise_multiplier, 1e-5)

            noise = torch.normal(0, actual_noise_std, size=clipped_delta.shape, device=clipped_delta.device)
            noisy_delta = clipped_delta + noise
            noisy_deltas.append(noisy_delta.cpu())
        return noisy_deltas, len(client_dataset)
    else:
        return [d.cpu() for d in weight_deltas], len(client_dataset)