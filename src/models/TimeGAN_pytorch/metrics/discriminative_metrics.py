import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A simple PyTorch Dataset for evaluation (real and fake samples with labels)
class EvalDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.ndarray): array of shape (N, seq_len, dim)
            labels (np.ndarray): binary labels (1 for real, 0 for fake)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Post-hoc RNN discriminator using GRU
class PostHocDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim (int): Number of features (dim)
            hidden_dim (int): Hidden size (typically int(dim/2))
        """
        super(PostHocDiscriminator, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # We use the last hidden state for classification
        out, h = self.gru(x)  # h: (1, batch, hidden_dim)
        h = h.squeeze(0)      # (batch, hidden_dim)
        logits = self.fc(h)   # (batch, 1)
        prob = torch.sigmoid(logits)
        return logits, prob

def discriminative_score_metrics(ori_data, generated_data, iterations=250, batch_size=128, lr=0.001):
    """
    Compute the discriminative score using a post-hoc RNN discriminator.
    
    Args:
        ori_data (np.ndarray): Original data of shape (N, seq_len, dim)
        generated_data (np.ndarray): Synthetic data of same shape (or list of arrays)
        iterations (int): Number of training iterations
        batch_size (int): Batch size for training
        lr (float): Learning rate for the discriminator optimizer
        
    Returns:
        discriminative_score (float): np.abs(classification accuracy - 0.5)
    """
    # Assume ori_data and generated_data have same shape:
    N, seq_len, dim = np.asarray(ori_data).shape
    
    # Create labels: 1 for real, 0 for fake.
    real_labels = np.ones((len(ori_data),), dtype=np.float32)
    fake_labels = np.zeros((len(generated_data),), dtype=np.float32)
    
    # Combine and then split into train/test (e.g., 80/20)
    all_data = np.concatenate((ori_data, generated_data), axis=0)
    all_labels = np.concatenate((real_labels, fake_labels), axis=0)
    
    # Shuffle data
    idx = np.random.permutation(len(all_data))
    all_data = all_data[idx]
    all_labels = all_labels[idx]
    
    # Train-test split
    split = int(0.8 * len(all_data))
    train_data, test_data = all_data[:split], all_data[split:]
    train_labels, test_labels = all_labels[:split], all_labels[split:]
    
    # Create DataLoaders
    train_dataset = EvalDataset(train_data, train_labels)
    test_dataset = EvalDataset(test_data, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Define network: use hidden_dim = int(dim/2)
    hidden_dim = int(dim / 2)
    discriminator_net = PostHocDiscriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    discriminator_net.train()
    
    optimizer = optim.Adam(discriminator_net.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Training loop
    for it in range(iterations):
        for batch_x, batch_labels in train_loader:
            batch_x, batch_labels = batch_x.to(device), batch_labels.to(device)  # Move to GPU
            
            optimizer.zero_grad()
            logits, _ = discriminator_net(batch_x)
            loss = bce_loss(logits, batch_labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate on test set
    discriminator_net.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_x, batch_labels in test_loader:
            batch_x = batch_x.to(device)  # Move to GPU
            logits, probs = discriminator_net(batch_x)
            preds = (probs > 0.5).float().cpu().numpy().squeeze()
            all_preds.extend(preds.tolist())
            all_true.extend(batch_labels.cpu().numpy().squeeze().tolist())
    
    acc = accuracy_score(all_true, all_preds)
    discriminative_score = np.abs(acc - 0.5)
    
    return discriminative_score