import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import MinMaxScaler

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (multi-GPU)

    # Ensure deterministic behavior in CUDA (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slows down training but ensures reproducibility

    print(f"Random seed set to: {seed}")

class TimeSeriesDataset:
    def __init__(self, data, seq_len, n_seq, train_ratio=0.8, val_ratio=0.1, scaler_path='models/scaler.pkl'):
        """
        Preprocesses and splits the time-series dataset into train, validation, and test sets.

        Args:
            data (numpy array): Original time-series data of shape (n_samples, n_features).
            seq_len (int): Length of the sequences.
            n_seq (int): Number of sequences (features).
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.

        Returns:
            train_dataset, val_dataset, test_dataset: Datasets for training, validation, and testing.
        """
        print("Initializing TimeSeriesDataset...")

        self.seq_len = seq_len
        self.n_seq = n_seq

        # Ensure correct shape
        assert data.shape[1] == n_seq, "Data dimensions do not match n_seq"

        # Split indices
        n_train = int(len(data) * train_ratio)
        n_val = int(len(data) * val_ratio)
        n_test = len(data) - n_train - n_val

        print(f"Data split: {n_train} train, {n_val} validation, {n_test} test")

        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        # Fit scaler on training data and transform all sets
        self.scaler = MinMaxScaler()
        train_data_scaled = self.scaler.fit_transform(train_data)
        val_data_scaled = self.scaler.transform(val_data)
        test_data_scaled = self.scaler.transform(test_data)

        # Convert data into sequences
        self.train_data = self._create_sequences(train_data_scaled)
        self.val_data = self._create_sequences(val_data_scaled)
        self.test_data = self._create_sequences(test_data_scaled)

        # Save the fitted scaler
        joblib.dump(self.scaler, scaler_path)

        print(f"Scaler saved at {scaler_path}")

    def _create_sequences(self, data):
        sequences = []
        print(len(data))
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i:i + self.seq_len])
            print(f'[{i+1}/{len(data)}]', end='\r')
        return np.array(sequences)

    def get_datasets(self):
        """Returns train, val, and test datasets as PyTorch Dataset objects."""
        #return TimeSeriesData(self.train_data), TimeSeriesData(self.val_data), TimeSeriesData(self.test_data)
        return self.train_data, self.val_data, self.test_data


class TimeSeriesData(Dataset):
    def __init__(self, data):
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]