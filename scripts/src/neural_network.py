"""
House Price Prediction Training Script
"""
from lib.utils import get_train_split_data, load_all_resale_data, get_cleaned_normalized_data
from lib.eval import get_regression_metrics
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

# Configuration
SEED = 42
MODEL_SAVE_PATH = 'house_price_predictor.pth'
BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Helper Functions ---
def get_device():
    """Get available device (GPU/CPU)"""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def df_to_tensor(df):
    """Convert DataFrame to PyTorch tensor with proper dtype handling"""
    device = get_device()
    assert not df.isnull().values.any(), "NaNs in dataframe"
    return torch.from_numpy(df.astype('float32').values).to(device)

# --- Model Architecture ---
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# --- Training Functions ---
def train_model(model, train_loader, test_data, optimizer, criterion, num_epochs):
    """Main training loop with validation tracking"""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    best_loss = float('inf')
    device = get_device()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        # Validation
        val_loss = evaluate_model(model, *test_data, criterion)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{num_epochs} | "
                  f"Train Loss: {epoch_loss/len(train_loader.dataset):.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {lr:.2e}")

def evaluate_model(model, X, y, criterion=None):
    """Evaluate model performance"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        y = y.to(device)
        preds = model(X.to(device))
        loss = criterion(preds, y) if criterion else None
        metrics = get_regression_metrics(y.cpu().numpy(), preds.cpu().numpy())
    return metrics, (loss.item() if loss else None)

# --- Main Execution ---
if __name__ == "__main__":
    # Load and prepare data
    X, y = load_all_resale_data()
    X, y = get_cleaned_normalized_data(X, y)
    X_train, X_test, y_train, y_test = get_train_split_data(X, y, 0.2)
    
    # Convert to tensors
    device = get_device()
    X_train_tensor = df_to_tensor(X_train)
    y_train_tensor = df_to_tensor(y_train).reshape(-1, 1)
    X_test_tensor = df_to_tensor(X_test)
    y_test_tensor = df_to_tensor(y_test).reshape(-1, 1)
    
    # Verify data integrity
    assert X_train_tensor.shape[0] == y_train_tensor.shape[0], "Training data mismatch"
    assert X_test_tensor.shape[0] == y_test_tensor.shape[0], "Test data mismatch"
    
    # Initialize model
    model = HousePricePredictor(X_train_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        test_data=(X_test_tensor, y_test_tensor),
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=NUM_EPOCHS
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    test_metrics, test_loss = evaluate_model(model, X_test_tensor, y_test_tensor, criterion)
    print(pd.DataFrame([test_metrics]))
    
    # Cleanup
    del X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None