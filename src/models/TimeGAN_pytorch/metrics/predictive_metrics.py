import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple dataset for the predictive task
class PredictiveDataset(Dataset):
    def __init__(self, data, seq_len, dim):
        """
        Args:
            data (np.ndarray): list/array of sequences of shape (seq_len, dim)
            We'll use the first (seq_len - 1) time steps and first (dim - 1) features as X,
            and the next step of the last feature as Y.
        """
        self.X = []
        self.Y = []
        for seq in data:
            # Use all time steps except last as input (and all but last feature)
            self.X.append(seq[:-1, :dim-1])
            # Use the last feature from time step 2 to end as prediction
            self.Y.append(seq[1:, dim-1].reshape(-1, 1))
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.Y[idx], dtype=torch.float32))

# Post-hoc predictor using a GRU network
class PostHocPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim (int): Number of input features (dim - 1)
            hidden_dim (int): Hidden size (e.g., int(dim/2))
        """
        super(PostHocPredictor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, seq_len-1, input_dim)
        out, _ = self.gru(x)
        # Apply FC layer on all timesteps
        y_hat = self.fc(out)
        return y_hat

def predictive_score_metrics(ori_data, generated_data, iterations=250, batch_size=128, lr=0.001):
    """
    Trains a post-hoc RNN predictor on synthetic data to predict one-step ahead on the last feature,
    then evaluates its performance on the original data.
    
    Args:
        ori_data (list or np.ndarray): List/array of original sequences (each of shape (seq_len, dim)).
        generated_data (list or np.ndarray): List/array of synthetic sequences.
        iterations (int): Number of training iterations.
        batch_size (int): Batch size.
        lr (float): Learning rate.
    
    Returns:
        predictive_score (float): The MAE of the predictions on the original data.
    """
    # Assume ori_data and generated_data are lists or arrays of sequences
    # Get dimensions from original data
    seq_len, dim = np.asarray(ori_data)[0].shape  # assuming uniform shape
    input_dim = dim - 1  # use all features except last as input
    hidden_dim = int(dim / 2)
    
    # Create training dataset using synthetic data
    train_dataset = PredictiveDataset(generated_data, seq_len, dim)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define predictor network and move to GPU
    predictor_net = PostHocPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    predictor_net.train()
    
    optimizer = optim.Adam(predictor_net.parameters(), lr=lr)
    mae_loss = nn.L1Loss()

    # Training loop on synthetic data
    for it in range(iterations):
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move batch to GPU
            
            optimizer.zero_grad()
            y_pred = predictor_net(X_batch)
            loss = mae_loss(y_pred, Y_batch)
            loss.backward()
            optimizer.step()
    
    # Now evaluate predictor on the original data
    test_dataset = PredictiveDataset(ori_data, seq_len, dim)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictor_net.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)  # Move batch to GPU
            y_pred = predictor_net(X_batch)
            all_pred.append(y_pred.cpu().numpy())  # Move predictions back to CPU
            all_true.append(Y_batch.cpu().numpy())  # Move ground truth back to CPU
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    # Compute MAE for each sequence and average them
    N = all_true.shape[0]
    total_mae = 0.0
    for i in range(N):
        total_mae += mean_absolute_error(all_true[i], all_pred[i])
    predictive_score = total_mae / N
    
    return predictive_score

# Example usage:
# synthetic_data = model.generate(num_samples=len(ori_data))  # from your TimeGAN model
# pred_score = predictive_score_metrics(ori_data, synthetic_data)
# print("Predictive Score (MAE):", pred_score)