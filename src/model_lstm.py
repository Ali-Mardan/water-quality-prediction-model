import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class WaterQualityDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (num_samples, seq_length, num_features)
            y: numpy array of shape (num_samples,)
        """
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # If Univariate, X might be (num_samples, seq_length), we need (num_samples, seq_length, 1)
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(-1)
            
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WaterQualityLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(WaterQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train_model(X_train, y_train, epochs=20, batch_size=32, lr=0.001):
    print("Initializing PyTorch LSTM model...")
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and loader
    dataset = WaterQualityDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model parameters
    input_size = dataset.X.shape[2] # Number of features
    
    model = WaterQualityLSTM(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
        
    return model

if __name__ == "__main__":
    print("Loading prepared sequence data...")
    try:
        X = np.load('X_sequences.npy')
        y = np.load('y_targets.npy')
        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
        
        # Simple train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("Starting training...")
        model = train_model(X_train, y_train, epochs=10)
        print("Training complete.")
        
        # Evaluation
        print("\nEvaluating on Test Set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            if len(X_test_tensor.shape) == 2:
                X_test_tensor = X_test_tensor.unsqueeze(-1)
            y_pred = model(X_test_tensor).cpu().numpy()
            
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Test R²: {r2:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        
    except FileNotFoundError:
        print("Sequence files not found. Please run data_sequence_prep.py first.")
