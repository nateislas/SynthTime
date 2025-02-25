import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm, trange
import joblib
import logging
from datetime import datetime
import uuid
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import spectral_norm


def create_logger():
    # Ensure the 'logs' directory exists
    os.makedirs("logs", exist_ok=True)  # Creates the 'logs' directory if it doesn't exist
    
    # Configure the logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all log messages
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/model_training.log"),  # Save logs to file
            logging.StreamHandler()  # Also print logs to console
        ]
    )
    
    # Create a logger instance
    logger = logging.getLogger(__name__)
    
    logger.info("Logger setup complete. Logging will be written to logs/model_training.log")
    print("Logger setup complete. Logging will be written to logs/model_training.log")
    
    return logger
    
def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    Computes Maximum Mean Discrepancy (MMD) between real (x) and generated (y) distributions.
    
    Args:
        x (Tensor): Samples from real distribution (batch_size, seq_len, features).
        y (Tensor): Samples from generated distribution (batch_size, seq_len, features).
        kernel (str): Kernel type ('rbf' or 'polynomial').
        sigma (float): Bandwidth for RBF kernel.
    
    Returns:
        Tensor: Scalar loss measuring MMD.
    """
    def rbf_kernel(x, y, sigma):
        gamma = 1.0 / (2 * sigma ** 2)
        xx = torch.matmul(x, x.transpose(-2, -1))
        yy = torch.matmul(y, y.transpose(-2, -1))
        xy = torch.matmul(x, y.transpose(-2, -1))

        x2 = torch.diagonal(xx, dim1=-2, dim2=-1).unsqueeze(-1)
        y2 = torch.diagonal(yy, dim1=-2, dim2=-1).unsqueeze(-1)

        K_xx = torch.exp(-gamma * (x2 + x2.transpose(-2, -1) - 2 * xx))
        K_yy = torch.exp(-gamma * (y2 + y2.transpose(-2, -1) - 2 * yy))
        K_xy = torch.exp(-gamma * (x2 + y2.transpose(-2, -1) - 2 * xy))
        
        return K_xx, K_yy, K_xy

    def polynomial_kernel(x, y, degree=3, coef=1):
        K_xx = (torch.matmul(x, x.transpose(-2, -1)) + coef) ** degree
        K_yy = (torch.matmul(y, y.transpose(-2, -1)) + coef) ** degree
        K_xy = (torch.matmul(x, y.transpose(-2, -1)) + coef) ** degree
        return K_xx, K_yy, K_xy

    if kernel == 'rbf':
        K_xx, K_yy, K_xy = rbf_kernel(x, y, sigma)
    elif kernel == 'polynomial':
        K_xx, K_yy, K_xy = polynomial_kernel(x, y)
    else:
        raise ValueError("Unsupported kernel type")

    mmd_loss = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd_loss

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
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)

    def get_datasets(self):
        """Returns train, val, and test datasets as PyTorch Dataset objects."""
        return StockDataset(self.train_data), StockDataset(self.val_data), StockDataset(self.test_data)


class StockDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.2, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # 🔥 Add LayerNorm
        self.dropout = nn.Dropout(0.4)  # or some other rate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.layer_norm(out)  # 🔥 Normalize the hidden states
        out = self.dropout(out)     # Additional dropout on entire GRU output
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional=False):
        super(Encoder, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers, bidirectional)
    
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional=False):
        super(Decoder, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, output_dim, n_layers, bidirectional)
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional=False):
        super(Generator, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers, bidirectional)
    
    def forward(self, x):
        return self.model(x)

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, n_layers, bidirectional=False):
        super(Supervisor, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, hidden_dim, n_layers, bidirectional=bidirectional)

    def forward(self, x):
        out = self.model(x)
        return out 

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, n_layers, bidirectional=False):
        super(Discriminator, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, 1, n_layers, bidirectional)
        self.model.fc = spectral_norm(self.model.fc)
    
    def forward(self, x):
        return self.model(x)
        
        
class TimeGAN:
    def __init__(self, input_dim, hidden_dim, seq_len, n_seq, batch_size=128, lr=0.01, gamma=0.25, n_layers=3, dataset_name = '', device='cuda'):
        self.logger = create_logger()
        
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.encoder = Encoder(input_dim, hidden_dim, n_layers).to(device)
        self.decoder = Decoder(hidden_dim, input_dim, n_layers).to(device)
        self.generator = Generator(n_seq, hidden_dim, n_layers).to(device)
        self.supervisor = Supervisor(hidden_dim, n_layers).to(device) # original paper uses n-1 layers for the Supervisor
        self.discriminator = Discriminator(hidden_dim, n_layers).to(device)
        self.autoencoder = nn.Sequential(self.encoder, self.decoder).to(device)
        
        # After building all modules, do:
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.generator.apply(self._init_weights)
        self.supervisor.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
        
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-6)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=1e-6)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr, weight_decay=1e-6)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=lr, weight_decay=1e-6)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr*0.1, weight_decay=1e-6)

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        
        self.scaler = joblib.load("models/scaler.pkl")  # Load the saved scaler
        
        # Generate a unique model ID using current date and UUID
        self.model_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]})_n_layers{n_layers}_seq_len{seq_len}_n_seq{n_seq}"
        
        self.logger.info(f"Model initialized with ID: {self.model_id}")
        print(f"Model initialized with ID: {self.model_id}")
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
        elif isinstance(m, (nn.GRU, nn.LSTM)):  # Include LSTM
            for name, param in m.named_parameters():
                if 'weight_ih' in name:  # Input-to-hidden weights
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:  # Hidden-to-hidden weights
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:  # Bias terms
                    nn.init.constant_(param.data, 0)
                    
                    # Special handling for LSTM biases (forget gate initialization)
                    if isinstance(m, nn.LSTM):
                        hidden_size = param.shape[0] // 4  # LSTM has 4 gates: input, forget, cell, output
                        param.data[hidden_size:hidden_size*2].fill_(1)  # Initialize forget gate bias to 1
    
    def trainTimeGAN(self, train_loader, test_loader):
        model.train_autoencoder(train_loader, epochs=train_steps)
        
        model.train_supervisor(train_loader, epochs=train_steps)
        
        model.train_joint_network(train_loader, epochs=train_steps)
        
    def load_model(self, 
                   encoder_path='models/best_encoder_final.pth', 
                   decoder_path='models/best_decoder_final.pth', 
                   generator_path='models/best_generator_final.pth', 
                   supervisor_path='models/best_supervisor_final.pth',
                   discriminator_path='models/best_discriminator_final.pth'):
        """
        Loads pre-trained models (Encoder, Decoder, Generator, Supervisor, Discriminator) 
        from saved checkpoints.
    
        Args:
            encoder_path (str): Path to the best encoder model.
            decoder_path (str): Path to the best decoder model.
            generator_path (str): Path to the best generator model.
            supervisor_path (str): Path to the best supervisor model.
            discriminator_path (str): Path to the best discriminator model.
    
        Returns:
            None
        """
    
        # Load Encoder
        if os.path.exists(encoder_path):
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device, weights_only=True))
            self.encoder.to(self.device)
            self.encoder.eval()  # Freeze encoder
            self.logger.info(f"Loaded best Encoder from {encoder_path}")
            print(f"Loaded best Encoder from {encoder_path}")
        else:
            self.logger.info(f"Warning: Encoder model not found at {encoder_path}")
            print(f"Warning: Encoder model not found at {encoder_path}")

        # Load Decoder
        if os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device, weights_only=True))
            self.decoder.to(self.device)
            self.decoder.eval()  # Freeze decoder
            self.logger.info(f"Loaded best Decoder from {decoder_path}")
            print(f"Loaded best Decoder from {decoder_path}")
        else:
            self.logger.info(f"Warning: Decoder model not found at {decoder_path}")
            print(f"Warning: Decoder model not found at {decoder_path}")
    
        # Load Generator
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device, weights_only=True))
            self.generator.to(self.device)
            self.generator.eval()  # Freeze generator
            self.logger.info(f"Loaded best Generator from {generator_path}")
            print(f"Loaded best Generator from {generator_path}")
        else:
            self.logger.info(f"Warning: Generator model not found at {generator_path}")
            print(f"Warning: Generator model not found at {generator_path}")
    
        # Load Supervisor
        if os.path.exists(supervisor_path):
            self.supervisor.load_state_dict(torch.load(supervisor_path, map_location=self.device, weights_only=True))
            self.supervisor.to(self.device)
            self.supervisor.eval()  # Freeze supervisor
            self.logger.info(f"Loaded best Supervisor from {supervisor_path}")
            print(f"Loaded best Supervisor from {supervisor_path}")
        else:
            self.logger.info(f"Warning: Supervisor model not found at {supervisor_path}")
            print(f"Warning: Supervisor model not found at {supervisor_path}")
    
        # Load Discriminator
        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device, weights_only=True))
            self.discriminator.to(self.device)
            self.discriminator.eval()  # Freeze discriminator
            self.logger.info(f"Loaded best Discriminator from {discriminator_path}")
            print(f"Loaded best Discriminator from {discriminator_path}")
        else:
            self.logger.info(f"Warning: Discriminator model not found at {discriminator_path}")
            print(f"Warning: Discriminator model not found at {discriminator_path}")
        
    def train_autoencoder(self, train_loader, val_loader=None,epochs=50, best_encoder_path="models/best_encoder.pth", best_decoder_path="models/best_decoder.pth"):
        """
        Trains the autoencoder (Encoder + Decoder) over multiple epochs,
        iterating through the entire training dataset in batches.
    
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset. Default is None.
            epochs (int, optional): Number of epochs to train. Default is 50.
            best_encoder_path (str, optional): File path to save the best encoder based on validation loss.
            best_decoder_path (str, optional): File path to save the best decoder based on validation loss.
    
        Returns:
            list: A history of losses for monitoring.
        """
        self.logger.info(f"Training AutoEncoder")
        print(f"Training AutoEncoder")
        
        # Set models to training mode
        self.encoder.train()
        self.decoder.train()
        
        scheduler_encoder = lr_scheduler.CosineAnnealingLR(self.opt_encoder, T_max=epochs, eta_min=1e-4)
        scheduler_decoder = lr_scheduler.CosineAnnealingLR(self.opt_decoder, T_max=epochs, eta_min=1e-4)
        
        # Track best model based on lowest loss
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
    
            # Iterate through dataset in batches
            for batch_idx, x in enumerate(train_loader):
                x = x.to(self.device)  # Move data to device

                # Zero gradients
                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()
    
                # Forward pass: Encode and reconstruct
                x_latent = self.encoder(x)  
                x_recon = self.decoder(x_latent)  
                
                embedding_loss_t0 = self.loss_mse(x, x_recon)

                # Backpropagation
                embedding_loss_t0.backward()

                # Update weights
                self.opt_encoder.step()
                self.opt_decoder.step()
    
                # Track loss
                total_loss += embedding_loss_t0.item()
                num_batches += 1
    
            # Compute average loss for the epoch
            avg_train_loss = total_loss / num_batches

            # Validation phase (if val_loader is provided)
            avg_val_loss = None
            if val_loader is not None:
                self.encoder.eval()
                self.decoder.eval()
                total_val_loss = 0
                num_val_batches = 0
    
                with torch.no_grad():
                    for x_val in val_loader:
                        x_val = x_val.to(self.device)
                        
                        x_latent_val = self.encoder(x_val)
                        x_recon_val = self.decoder(x_latent_val)
    
                        val_loss = self.loss_mse(x_val, x_recon_val)
                        total_val_loss += val_loss.item()
                        num_val_batches += 1
    
                avg_val_loss = total_val_loss / num_val_batches
                
                # Set back to training mode
                self.encoder.train()
                self.decoder.train()
                
            # Adjust LR based on validation loss
            scheduler_encoder.step()
            scheduler_decoder.step()
            
            last_lr = scheduler_encoder.get_last_lr()[0]
            
            if val_loader is not None:
                self.logger.info(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}, lr: {last_lr}")
                print(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}, lr: {last_lr}")
            else:
                self.logger.info(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, lr: {last_lr}")
                print(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, lr: {last_lr}")
            
            # Determine if we should save the model (use validation loss if available, otherwise train loss)
            current_best_loss = avg_val_loss if val_loader is not None else avg_train_loss
            if current_best_loss < best_loss:
                best_loss = current_best_loss
                torch.save(self.encoder.state_dict(), best_encoder_path)
                torch.save(self.decoder.state_dict(), best_decoder_path)
    
                self.logger.info(f"Best autoencoder model saved with loss {best_loss:.6f}")
                print(f"Best autoencoder model saved with loss {best_loss:.6f}")

        self.logger.info("Autoencoder training complete.")
        print("Autoencoder training complete.")
        
        # Load Best encoder and decoder into the TimeGAN attributes
        if os.path.exists(best_encoder_path):
            self.encoder.load_state_dict(torch.load(best_encoder_path, map_location=self.device, weights_only=True))
            self.encoder.to(self.device)
            self.encoder.eval()  # Freeze encoder
        
        if os.path.exists(best_decoder_path):
            self.decoder.load_state_dict(torch.load(best_decoder_path, map_location=self.device, weights_only=True))
            self.decoder.to(self.device)
            self.decoder.eval()  # Freeze decoder
            
        self.logger.info(f"Loaded best encoder and decoder")
        print(f"Loaded best encoder and decoder")

    def train_supervisor(self, train_loader, val_loader=None, epochs=50, save_path="models/best_supervisor.pth"):
        """
        Trains the Supervisor network over multiple epochs using a pre-trained encoder.
        If a validation loader is provided, the model is evaluated after each epoch.
    
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset. Default is None.
            epochs (int, optional): Number of epochs to train. Default is 50.
            save_path (str, optional): File path to save the best Supervisor model.
            scheduler_type (str, optional): Type of learning rate scheduler. Default is "ReduceLROnPlateau".
    
        Returns:
            dict: Training and validation loss history.
        """
        self.logger.info("Training Supervisor model...")
        print("Training Supervisor model...")
    
        # Set Supervisor to training mode
        self.supervisor.train()
    
        # Track best validation loss
        best_loss = float('inf')
        
        scheduler = lr_scheduler.CosineAnnealingLR(self.opt_supervisor, T_max=epochs, eta_min=1e-4)
    
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
    
            # Training Phase
            for batch_idx, x in enumerate(train_loader):
                x = x.to(self.device)  # Move data to device
    
                # Zero gradients
                self.opt_generator.zero_grad()
                self.opt_supervisor.zero_grad()
    
                # Forward pass: Encode input sequence using the pre-trained encoder
                with torch.no_grad():  # Prevent encoder from updating
                    h = self.encoder(x)
    
                # Supervisor predicts next latent state
                h_hat_supervised = self.supervisor(h)
    
                # Compute supervised loss (MSE)
                # Inject some noise to prevent supervisor from collapsing
                #g_loss_s = self.loss_mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])
                g_loss_s = self.loss_mse(h[:, 1:, :], h_hat_supervised[:, :-1, :]) 
                g_loss_s += torch.randn_like(g_loss_s) * 0.001
    
                # Backpropagation
                g_loss_s.backward()
    
                # Update Supervisor weights
                self.opt_supervisor.step()
                self.opt_generator.step()
    
                # Track loss
                total_loss += g_loss_s.item()
                num_batches += 1
    
            # Compute average training loss for the epoch
            avg_train_loss = total_loss / num_batches

            # Validation Phase
            avg_val_loss = None
            if val_loader is not None:
                self.supervisor.eval()  # Set to evaluation mode
                total_val_loss = 0
                num_val_batches = 0
    
                with torch.no_grad():
                    for x_val in val_loader:
                        x_val = x_val.to(self.device)
    
                        # Encode input sequence
                        h_val = self.encoder(x_val)
    
                        # Supervisor predicts next latent state
                        h_hat_val = self.supervisor(h_val)
    
                        # Compute supervised loss
                        val_loss = self.loss_mse(h_val[:, 1:, :], h_hat_val[:, :-1, :])
                        total_val_loss += val_loss.item()
                        num_val_batches += 1
    
                avg_val_loss = total_val_loss / num_val_batches
                
                scheduler.step()  # Adjust LR based on validation loss
                
                last_lr = scheduler.get_last_lr()[0]

                self.logger.info(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}, lr: {last_lr}")
                print(f"epoch [{epoch+1}/{epochs}], train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}, lr: {last_lr}")
    
                # Save the best model based on validation loss
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(self.supervisor.state_dict(), save_path)
                    self.logger.info(f"Best Supervisor model saved with validation loss {best_loss:.6f}")
                    print(f"Best Supervisor model saved with validation loss {best_loss:.6f}")
    
                self.supervisor.train()  # Switch back to training mode
    
            else:
                avg_val_loss = avg_train_loss  # If no validation set, monitor training loss instead
                
                scheduler.step()
    
                # Save the best model based on training loss (if no validation set is used)
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    torch.save(self.supervisor.state_dict(), save_path)
                    self.logger.info(f"Best Supervisor model saved with training loss {best_loss:.6f}")
    
        self.logger.info("Supervisor training complete.")
    
        # Load the best Supervisor model (only if validation was used)
        if val_loader is not None and os.path.exists(save_path):
            self.supervisor.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))
            self.supervisor.to(self.device)
            self.supervisor.eval()  # Set to evaluation mode
            self.logger.info("Loaded best Supervisor model based on validation loss.")
            print("Loaded best Supervisor model based on validation loss.")
    

    def train_joint_network(self, train_loader, val_loader=None, epochs=100, device='cuda', save_path="models/"):
        """
        Jointly trains the Generator, Encoder, and Discriminator networks over multiple epochs.
        Saves the best models based on the lowest total loss.
    
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset. Default is None.
            epochs (int): Number of training iterations (matches original paper).
            device (torch.device): The device to use for training (e.g., "cuda" or "cpu").
            save_path (str): Directory where the best models will be saved.
    
        Returns:
            None
        """
        self.logger.info("Training Joint Network.")
        print("Training Joint Network.")
    
        # Track the best model based on loss
        best_loss = float('inf')
    
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()
    
        for epoch in range(epochs):
            for step, X_real in enumerate(train_loader):
                X_real = X_real.to(device)
    
                # --- Train Generator and Supervisor Twice (like the original) ---
                for _ in range(2):  # Train Generator two times before Discriminator
                    Z_noise = self.random_generator(X_real.shape[0], self.seq_len, self.n_seq, self.device)
                    
                    # Train the generator (returns three loss components)
                    step_g_loss_u, step_g_loss_s, step_g_loss_v, step_g_loss_mmd = self.train_generator(X_real, Z_noise)
    
                    # Train the encoder (autoencoder training)
                    step_e_loss_t0 = self.train_encoder(X_real)
    
                # --- Train the Discriminator Only if Loss is Too High ---
                # Z_disc = torch.rand_like(X_real).to(device)  # Generate another batch of noise
                Z_disc  = self.random_generator(X_real.shape[0], self.seq_len, self.n_seq, self.device)
                
    
                # Compute discriminator loss
                step_d_loss = self.discriminator_loss(X_real, Z_disc)
    
                if step_d_loss > 0.15:  # Matches the original paper
                    step_d_loss = self.train_discriminator(X_real, Z_disc)
    
                # Compute total loss for monitoring
                #total_loss = step_g_loss_u + step_g_loss_s + step_g_loss_v + g_loss_mmd + step_e_loss_t0 + step_d_loss
                
                total_loss = (
                            0.5 * step_g_loss_u +  # Primary generator loss
                            0.3 * step_g_loss_s +  # Supervisor loss (important but not dominant)
                            0.05 * step_g_loss_v +  # Generator moment matching loss
                            0.05 * step_g_loss_mmd +  # Generator moment matching loss
                            0.3 * step_e_loss_t0 + # Autoencoder loss (reconstruction quality)
                            0.1 * step_d_loss      # Discriminator loss (shouldn't overpower)
                        )
    
                # Print losses at intervals (every 1000 steps like the original)
                if step % 1000 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], "
                          f"d_loss: {step_d_loss:.4f}, g_loss_u: {step_g_loss_u:.4f}, "
                          f"g_loss_s: {step_g_loss_s:.4f}, g_loss_v: {step_g_loss_v:.4f}, "
                          f"e_loss_t0: {step_e_loss_t0:.4f}")
    
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], "
                          f"d_loss: {step_d_loss:.4f}, g_loss_u: {step_g_loss_u:.4f}, "
                          f"g_loss_s: {step_g_loss_s:.4f}, g_loss_v: {step_g_loss_v:.4f}, "
                          f"e_loss_t0: {step_e_loss_t0:.4f}")
    
            # --- Validation Phase ---
            avg_val_loss = None
            if val_loader is not None:
                self.encoder.eval()
                self.decoder.eval()
                self.generator.eval()
                self.supervisor.eval()
                self.discriminator.eval()
    
                total_val_loss = 0
                num_val_batches = 0
    
                with torch.no_grad():
                    for X_val in val_loader:
                        X_val = X_val.to(device)
    
                        Z_noise_val = torch.rand((X_val.shape[0], self.seq_len, self.n_seq)).to(self.device)
    
                        # Compute validation losses (same as train loop)
                        val_g_loss_u, val_g_loss_s, val_g_loss_v, val_g_loss_mmd = self.train_generator(X_val, Z_noise_val, training=False)
                        val_e_loss_t0 = self.train_encoder(X_val, training=False)
                        Z_disc_val = torch.rand_like(X_val).to(device)
                        val_d_loss = self.train_discriminator(X_val, Z_disc_val, training=False)
    
                        total_val_loss += (
                            0.5 * val_g_loss_u +  # Primary generator loss
                            0.3 * val_g_loss_s +  # Supervisor loss (important but not dominant)
                            0.05 * val_g_loss_v +  # Generator moment matching loss
                            0.05 * val_g_loss_mmd +  # Generator moment matching loss
                            0.3 * val_e_loss_t0 + # Autoencoder loss (reconstruction quality)
                            0.1 * val_d_loss      # Discriminator loss (shouldn't overpower)
                        )
                        
                        num_val_batches += 1
    
                avg_val_loss = total_val_loss / num_val_batches
    
                self.logger.info(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.6f}")
    
                # Restore models to training mode
                self.encoder.train()
                self.decoder.train()
                self.generator.train()
                self.supervisor.train()
                self.discriminator.train()
    
            # --- Save best models based on lowest loss ---
            current_best_loss = avg_val_loss if val_loader is not None else total_loss
            if current_best_loss < best_loss:
                best_loss = current_best_loss
                torch.save(self.generator.state_dict(), f"{save_path}/best_generator_final.pth")
                torch.save(self.encoder.state_dict(), f"{save_path}/best_encoder_final.pth")
                torch.save(self.decoder.state_dict(), f"{save_path}/best_decoder_final.pth")
                torch.save(self.supervisor.state_dict(), f"{save_path}/best_supervisor_final.pth")
                torch.save(self.discriminator.state_dict(), f"{save_path}/best_discriminator_final.pth")
                self.logger.info(f"Best models saved with loss {best_loss:.6f}")
                print(f"Best models saved with loss {best_loss:.6f}")
    
        self.logger.info("Joint networks training complete.")
        print("Joint networks training complete.")
        
    def train_encoder(self, x, training=True):
        """
        Trains the encoder if `training=True`, otherwise computes the loss without updating weights.
        
        Args:
            x (torch.Tensor): Real input data.
            training (bool): Whether to update weights. If False, only returns loss (for validation).
        
        Returns:
            torch.Tensor: Embedding loss.
        """
        if training:
            self.opt_encoder.zero_grad()
            self.opt_decoder.zero_grad()
    
        with torch.set_grad_enabled(training):
            h = self.encoder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.loss_mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])
    
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.loss_mse(x, x_tilde)
            e_loss = embedding_loss_t0 + 0.1 * generator_loss_supervised
           
            if training:
                e_loss.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()
    
        return e_loss

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        # Collapse (batch_size, seq_len) -> single mean/var per feature dimension
        y_true_mean = torch.mean(y_true, dim=(0,1))  # shape (n_features,)
        y_pred_mean = torch.mean(y_pred, dim=(0,1))
        
        y_true_var = torch.var(y_true, dim=(0,1))
        y_pred_var = torch.var(y_pred, dim=(0,1))
        
        g_loss_mean = torch.mean(torch.abs(y_true_mean - y_pred_mean))
        g_loss_var = torch.mean(torch.abs(torch.sqrt(y_true_var + 1e-6) - torch.sqrt(y_pred_var + 1e-6)))
        #g_loss_var = torch.mean(torch.abs(y_true_var  - y_pred_var + 1e-6))

        return g_loss_mean + g_loss_var
    
    def train_generator(self, x, z, training=True):
        """
        Trains the generator using adversarial loss, supervised loss, and MMD loss.
        
        Args:
            x (Tensor): Real input data.
            z (Tensor): Random noise input.
            training (bool): Whether to update weights. If False, only computes loss.
        
        Returns:
            tuple: Generator losses (unsupervised, supervised, moment, MMD).
        """
        if training:
            self.opt_generator.zero_grad()
            self.opt_supervisor.zero_grad()
    
        with torch.set_grad_enabled(training):
            # Generate synthetic sequences
            z_noisy = z + torch.randn_like(z) * 0.1  # Inject noise for robustness
            h_fake = self.generator(z_noisy)
            x_fake = self.decoder(self.supervisor(h_fake))
    
            # Adversarial loss (fake samples should fool discriminator)
            y_fake = self.discriminator(self.supervisor(h_fake))
            g_loss_u = self.loss_bce(y_fake, torch.ones_like(y_fake))
    
            # Supervised loss (next-step prediction)
            h_real = self.encoder(x)
            h_hat_supervised = self.supervisor(h_real)
            g_loss_s = self.loss_mse(h_real[:, 1:, :], h_hat_supervised[:, :-1, :])
    
            # Moment loss (mean and variance alignment)
            g_loss_v = self.calc_generator_moments_loss(x, x_fake)
    
            # **MMD Loss (distribution alignment)**
            g_loss_mmd = compute_mmd(x, x_fake, kernel='rbf', sigma=1.0)
    
            # Total generator loss
            g_loss = (
                g_loss_u +                      # Adversarial loss
                self.gamma * g_loss_s +         # Supervised loss
                100 * g_loss_v +                # Moment loss
                50 * g_loss_mmd                 # MMD loss (adjust weight)
            )
    
            if training:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.opt_generator.step()
                self.opt_supervisor.step()
    
        return g_loss_u, g_loss_s, g_loss_v, g_loss_mmd
        
    def discriminator_loss(self, x, z):
        y_real = self.discriminator(self.encoder(x))
        discriminator_loss_real = self.loss_bce(y_real, torch.ones_like(y_real))

        y_fake = self.discriminator(self.supervisor(self.generator(z)))
        discriminator_loss_fake = self.loss_bce(y_fake, torch.zeros_like(y_fake))

        y_fake_e = self.discriminator(self.generator(z))
        discriminator_loss_fake_e = self.loss_bce(y_fake_e, torch.zeros_like(y_fake_e))

        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    def train_discriminator(self, x, z, training=True):
        """
        Trains the discriminator if training=True, otherwise computes the loss without updating weights.
        
        Args:
            x (torch.Tensor): Real input data.
            z (torch.Tensor): Fake input (generated noise).
            training (bool): Whether to update weights. If False, only returns loss (for validation).
        
        Returns:
            torch.Tensor: Discriminator loss.
        """
        if training:
            self.opt_discriminator.zero_grad()  # Reset gradients
        
        # Correct gradient handling
        with torch.no_grad() if not training else torch.enable_grad():
            discriminator_loss = self.discriminator_loss(x, z)  # Compute loss
        
        if training:
            discriminator_loss.backward()  # Backpropagation
            
            # **Gradient Clipping (Prevents Exploding Gradients)**
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
    
            self.opt_discriminator.step()  # Update weights
        
        return discriminator_loss.detach()  # Return a non-tracked tensor
        
    def random_generator(self, batch_size, seq_len, dim, device='cuda'):
        # Uniform [0, 1], matching the official code's style
        return torch.rand(batch_size, seq_len, dim, device=device)


    def sample(self, n_samples):
        """
        Generates synthetic time-series data using the trained TimeGAN model.
    
        Args:
            n_samples (int): Number of samples to generate.
    
        Returns:
            numpy array: Generated time-series data in the original scale.
        """
        self.load_model()  # Ensure models are loaded
        
        steps = n_samples // self.batch_size + 1
        generated_data = []
    
        for _ in tqdm(range(steps), desc='Synthetic data generation'):
            # Generate random noise (Z_mb in the original implementation)
            Z_ = self.random_generator(self.batch_size, self.seq_len, self.n_seq, self.device)

            # Generate latent space representation (H_hat in the original implementation)
            H_hat = self.generator(Z_)
            
            # Convert latent space representation to time-series data (X_hat)
            X_hat = self.decoder(H_hat).cpu().detach().numpy()  # Convert to numpy
    
            # Append to generated data list
            generated_data.append(X_hat)
    
        # Concatenate all generated sequences
        generated_data = np.vstack(generated_data)[:n_samples]  # Trim to exact n_samples
    
        # Renormalization (convert back to original scale)
        generated_data = self.scaler.inverse_transform(generated_data.reshape(-1, self.n_seq)).reshape(n_samples, self.seq_len, self.n_seq)
    
        return generated_data