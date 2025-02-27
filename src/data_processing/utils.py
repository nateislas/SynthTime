import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

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
                    col_names: list = []):
    """
    Save synthetic time-series data to a Parquet file.

    Args:
        synth_data: numpy array of shape (n, seq_len, n_seq), where:
            - n is the number of generated samples,
            - seq_len is the number of continuous trading days,
            - n_seq is the number of features (e.g., OPEN, LOW, CLOSE, HIGH).
        save_dir: Directory where the Parquet file will be saved.
        dataset_name: Name of the dataset (used for naming the Parquet file).
        col_names: List of column names. If empty, default names are generated.
    """
    # Ensure the directory exists
    save_path = os.path.join(save_dir, dataset_name, str(synth_data.shape[1]))  # seq_len
    os.makedirs(save_path, exist_ok=True)

    # Flatten the synthetic data to shape (n, seq_len * n_seq)
    n, seq_len, n_seq = synth_data.shape
    synth_data_flat = synth_data.reshape(n, seq_len * n_seq)

    # Generate default column names if col_names is not provided
    if not col_names:
        col_names = [f"day_{i//n_seq + 1}_feature_{i % n_seq + 1}" for i in range(seq_len * n_seq)]
    
    # Ensure the length of col_names matches the flattened data
    assert len(col_names) == seq_len * n_seq, f"Expected {seq_len * n_seq} column names, got {len(col_names)}."

    # Convert to pandas DataFrame
    df = pd.DataFrame(synth_data_flat, columns=col_names)

    # Define the Parquet file path
    file_path = os.path.join(save_path, f"{dataset_name}_synth_data.parquet")

    # Save to Parquet file
    df.to_parquet(file_path, index=False)
    print(f"Synthetic data saved to {file_path}")