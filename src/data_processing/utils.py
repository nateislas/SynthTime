import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os
from sklearn.preprocessing import OneHotEncoder

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
        """
        Internal method to create sequences from data array.

        Args:
            data (numpy array): The data from which to form sequences.

        Returns:
            sequences (numpy array): An array of shape (num_sequences, seq_len, n_features).
        """
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)

    def get_datasets(self):
        """Returns train, val, and test datasets as PyTorch Dataset objects."""
        #return TimeSeriesData(self.train_data), TimeSeriesData(self.val_data), TimeSeriesData(self.test_data)
        return self.train_data, self.val_data, self.test_data


class TimeSeriesData(Dataset):
    """
    Dataset for handling time-series data sequences.

    Args:
        data (numpy array): Data to be loaded, shaped (num_sequences, seq_len, n_features).
    """
    def __init__(self, data):
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
class KFoldTimeSeries:
    def __init__(self, data, seq_len, n_splits=5):
        """
        Initializes a KFoldTimeSeries object for performing time-series cross-validation.

        Args:
            data (numpy array): The time-series data to split, shaped (n_samples, n_features).
            seq_len (int): The length of each sequence.
            n_splits (int): The number of folds or splits.
        """
        self.data = data
        self.seq_len = seq_len
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def get_fold(self, fold_index):
        """
        Retrieves training and validation datasets for a given fold index using TimeSeriesSplit.

        Args:
            fold_index (int): The index of the fold (0 to n_splits-1).

        Returns:
            train_dataset (TimeSeriesData): The training dataset for the fold.
            val_dataset (TimeSeriesData): The validation dataset for the fold.
        """
        for i, (train_indices, val_indices) in enumerate(self.tscv.split(self.data)):
            if i == fold_index:
                # Fit scaler on the training data only
                scaler = MinMaxScaler()
                train_data_scaled = scaler.fit_transform(self.data.iloc[train_indices])
                val_data_scaled = scaler.transform(self.data.iloc[val_indices])

                # Create sequences
                train_sequences = self._create_sequences(train_data_scaled)
                val_sequences = self._create_sequences(val_data_scaled)

                # Return as Dataset objects
                return np.array(train_sequences), np.array(val_sequences)

    def _create_sequences(self, data):
        """
        Internal method to create sequences from data array.

        Args:
            data (numpy array): The data from which to form sequences.

        Returns:
            sequences (numpy array): An array of shape (num_sequences, seq_len, n_features).
        """
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)
    
def save_synth_data(synth_data: np.ndarray,
                    save_dir: str = './data/synthetic_data',
                    dataset_name: str = 'unnamed_dataset',
                    col_names: list = None):
    """
    Save synthetic time-series data to Parquet files.

    Args:
        synth_data: numpy array of shape (n, seq_len, n_seq), where:
            - n is the number of generated samples,
            - seq_len is the number of continuous trading days,
            - n_seq is the number of features (e.g., OPEN, LOW, CLOSE, HIGH).
        save_dir: Directory where the Parquet files will be saved.
        dataset_name: Name of the dataset (used for naming the Parquet files).
        col_names: List of column names. If empty, default names are generated for each feature.
    """
    n_samples = str(synth_data.shape[0])
    seq_len = str(synth_data.shape[1])
    
    # Ensure the directory exists
    save_path = os.path.join(save_dir, dataset_name, seq_len)
    
    os.makedirs(save_path, exist_ok=True)

    # Check if column names are provided
    if col_names is None:
        col_names = [f'feature_{i}' for i in range(synth_data.shape[2])]  # Assuming n_seq is the third dimension

    # Iterate over each sample and save it as a Parquet file
    for i in range(synth_data.shape[0]):
        df = pd.DataFrame(synth_data[i], columns=col_names)
        file_path = os.path.join(save_path, f"{dataset_name}_sample_{i}.csv")
        df.to_csv(file_path, index=False)
    
    print(f"Saved {n_samples} {dataset_name} (seq_len={seq_len}) generated synthetic samples to {save_path}")

def prepare_timegan_data_forecasting(df, seq_len=21, train_ratio=0.8, val_ratio=0.1):
    """
    Prepares data for Conditional TimeGAN for forecasting and splits into train, validation, and test sets.
    The scaler is fit only on the training data to avoid data leakage.

    Args:
        df (pd.DataFrame): DataFrame with columns [OPEN, HIGH, LOW, CLOSE, Market Regime, Monthly Return]
        seq_len (int): Sequence length for time-series data.
        train_ratio (float): Proportion of data used for training.
        val_ratio (float): Proportion of data used for validation.

    Returns:
        tuple: (train_data, val_data, test_data, scaler)
            - train_data: (time_series_train, condition_train)
            - val_data: (time_series_val, condition_val)
            - test_data: (time_series_test, condition_test)
            - scaler: MinMaxScaler fitted on train data (for inverse transformation if needed)
    """
    time_series_cols = ["OPEN", "HIGH", "LOW", "CLOSE"]
    condition_cols = ["Market Regime", "Monthly Return"]

    # Convert time-series data to NumPy array
    time_series_data = df[time_series_cols].values

    # Split data indices for train, val, test
    n = len(time_series_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # **Fit the scaler only on the training set to prevent data leakage**
    scaler = MinMaxScaler()
    time_series_train = scaler.fit_transform(time_series_data[:train_end])  # Fit on train
    time_series_val = scaler.transform(time_series_data[train_end:val_end])  # Transform only
    time_series_test = scaler.transform(time_series_data[val_end:])  # Transform only

    # Combine back for sequence extraction
    time_series_data_scaled = np.vstack([time_series_train, time_series_val, time_series_test])

    # Process the condition data
    condition_data = df[condition_cols].copy()

    # One-Hot Encode Market Regime
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    market_regime_one_hot = encoder.fit_transform(condition_data[["Market Regime"]])  # Shape (n, 3)

    # Ensure Monthly Return is also 2D and apply log transform
    monthly_return = np.log2(condition_data[["Monthly Return"]].values + 1)

    # Combine market regime and monthly return into a single condition array
    condition_data = np.hstack((market_regime_one_hot, monthly_return))  # Shape (n, 4)

    # Create sequences
    time_series_seq, condition_seq = [], []

    for i in range(len(df) - seq_len + 1):
        time_series_seq.append(time_series_data_scaled[i:i + seq_len])  # Shape (seq_len, 4)
        condition_seq.append(condition_data[i])  # Use condition at time t

    time_series_seq, condition_seq = np.array(time_series_seq), np.array(condition_seq)

    # Split sequences into train, val, test
    time_series_train, condition_train = time_series_seq[:train_end], condition_seq[:train_end]
    time_series_val, condition_val = time_series_seq[train_end:val_end], condition_seq[train_end:val_end]
    time_series_test, condition_test = time_series_seq[val_end:], condition_seq[val_end:]

    return (time_series_train, condition_train), (time_series_val, condition_val), (time_series_test, condition_test)


class CondTimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        time_series_data: np.ndarray of shape (n, seq_len, n_seq)
        cond_data: np.ndarray of shape (n, cond_dim)
        """

        self.time_series_data = torch.tensor(data[0], dtype=torch.float32)
        self.cond_data = torch.tensor(data[1], dtype=torch.float32)

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        return self.time_series_data[idx], self.cond_data[idx]
