"""
Memory-Optimized House Price Prediction Training Script
"""
import os
from lib.utils import get_train_split_data, load_all_resale_data, get_cleaned_normalized_data
from lib.eval import get_regression_metrics
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# Configuration
MODEL_SAVE_PATH = 'house_price_predictor.pth'
BATCH_SIZE = 1024
ACCUMULATION_STEPS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('medium')

# --- Helper Functions ---
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def df_to_tensor(df):
    """Keep data on CPU with float32 precision"""
    return torch.from_numpy(df.astype('float32').values)

# --- Memory-Efficient Model ---
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# --- Training Functions ---
def train_model(model, train_loader, optimizer, criterion, device):
    """Memory-optimized training loop"""
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        # Move only current batch to GPU
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) / ACCUMULATION_STEPS
        
        # Backward pass
        loss.backward()
        total_loss += loss.item() * ACCUMULATION_STEPS
        
        # Gradient accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Free memory immediately
            del batch_X, batch_y, outputs, loss
            torch.cuda.empty_cache()
    
    return total_loss / len(train_loader.dataset)

# --- Main Execution ---
if __name__ == "__main__":
    device = get_device()
    
    # Load and preprocess data
    X, y = load_all_resale_data()
    X, y = get_cleaned_normalized_data(X, y)
    X_train, X_test, y_train, y_test = get_train_split_data(X, y, 0.2)
    
    # Convert to CPU tensors
    train_dataset = TensorDataset(
        df_to_tensor(X_train),
        df_to_tensor(y_train).reshape(-1, 1)
    )
    test_dataset = TensorDataset(
        df_to_tensor(X_test),
        df_to_tensor(y_test).reshape(-1, 1)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        pin_memory=True,
        num_workers=4
    )
    
    # Initialize model and optimizer
    model = HousePricePredictor(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler('cuda')
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        avg_loss = train_model(model, train_loader, optimizer, criterion, device)
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()
                del X_batch, y_batch, outputs
        
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
    
    # Final evaluation
    model.eval()
    final_preds = []
    final_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            final_preds.extend(preds)
            final_true.extend(y_batch.numpy())
    
    metrics = get_regression_metrics(final_true, final_preds)
    print("\nFinal Test Metrics:")
    print(pd.DataFrame([metrics]))
    
    # Cleanup
    del train_loader, test_loader, model
    torch.cuda.empty_cache()