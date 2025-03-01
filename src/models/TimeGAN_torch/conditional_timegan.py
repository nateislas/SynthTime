import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import uuid
from datetime import datetime
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from typing import Optional

def create_logger():
    import logging
    logger = logging.getLogger("timegan")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger

# ----------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim + cond_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, x, cond):
        cond_expanded = cond.unsqueeze(1).repeat(1, x.shape[1], 1)  # Expand and repeat across sequence length
        x = torch.cat((x, cond_expanded), dim=-1)  # Concatenate the correctly expanded conditioning variable
        out, _ = self.rnn(x)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, cond_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim + cond_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()

    def forward(self, h, cond):
        cond_expanded = cond.unsqueeze(1).repeat(1, h.shape[1], 1)
        h = torch.cat((h, cond_expanded), dim=-1)
        out, _ = self.rnn(h)
        out = self.fc(out)
        return self.act(out)

class Generator(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim + cond_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, z, cond):
        cond_expanded = cond.unsqueeze(1).repeat(1, z.shape[1], 1)  # Expand and repeat across sequence length
        z = torch.cat((z, cond_expanded), dim=-1)
        out, _ = self.rnn(z)
        return self.act(self.fc(out))

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, cond_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim + cond_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, h, cond):
        cond_expanded = cond.unsqueeze(1).repeat(1, h.shape[1], 1)
        h = torch.cat((h, cond_expanded), dim=-1)
        out, _ = self.rnn(h)
        return self.act(self.fc(out))

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, cond_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim + cond_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h, cond):
        cond_expanded = cond.unsqueeze(1).repeat(1, h.shape[1], 1)  # Expand and repeat across sequence length
        h = torch.cat((h, cond_expanded), dim=-1)
        out, _ = self.rnn(h)
        logits = self.fc(out)
        return logits
# ----------------------------------------------------------------------------


class CondTimeGAN:
    def __init__(self, parameters):
        self.logger = create_logger()
        
        # Extract parameters
        self.dataset_name = parameters["dataset_name"]
        self.seq_len = parameters["seq_len"]
        self.n_seq = parameters["n_seq"]
        self.hidden_dim = parameters["hidden_dim"]
        self.num_layers = parameters["num_layers"]
        self.cond_dim = parameters["cond_dim"]  # New conditioning variable dimension
        self.gamma = parameters.get("gamma", 1.0)
        self.batch_size = parameters.get("batch_size", 32)
        self.lr = parameters.get("lr", 2e-4)
        self.device = parameters.get("device", torch.device("cpu"))
        
        # Initialize models with conditioning support
        self.encoder = Encoder(self.n_seq, self.cond_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.decoder = Decoder(self.hidden_dim, self.n_seq, self.cond_dim, self.num_layers).to(self.device)
        self.generator = Generator(self.n_seq, self.cond_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.supervisor = Supervisor(self.hidden_dim, self.cond_dim, self.num_layers).to(self.device)
        self.discriminator = Discriminator(self.hidden_dim, self.cond_dim, self.num_layers).to(self.device)
        
        # Initialize weights
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.generator.apply(self._init_weights)
        self.supervisor.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)

        # Initialize optimizers
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=self.lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # Loss functions
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCEWithLogitsLoss()
        
        # Generate a unique model ID using current date and UUID
        self.model_id = f"{self.dataset_name}_" \
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_" \
                        f"{uuid.uuid4().hex[:8]}_" \
                        f"n_layers{self.num_layers}_seq_len{self.seq_len}_n_seq{self.n_seq}_hidden_dim{self.hidden_dim}_cond_dim{self.cond_dim}"
                
        self.logger.info(f"Model initialized with ID: {self.model_id}")

    def _init_weights(self, m: nn.Module):
        """
        Weight initialization, similar to the original code’s approach.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:  # input-to-hidden
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:  # hidden-to-hidden
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:  # biases
                    nn.init.constant_(param.data, 0.0)
                    # For LSTM, might want to init forget gate bias = 1
                    if isinstance(m, nn.LSTM):
                        hidden_size = param.shape[0] // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def _random_generator(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates random noise and random conditioning values.
        """
        Z = torch.randn(batch_size, self.seq_len, self.n_seq, device=self.device)
        return Z
    
    def train_autoencoder(self, 
                          train_data: np.ndarray, 
                          val_data: np.ndarray = None, 
                          epoch: int = 250, 
                          patience: int = 10):
        """
        Phase 1: Train the autoencoder (encoder + decoder) to learn initial embeddings.

        Uses early stopping if `val_data` is provided.

        Args:
            - train_data: Training dataset (numpy array of shape [N, seq_len, n_seq])
            - val_data: Validation dataset (optional, same shape as train_data)
            - epoch: Maximum number of training epochs
            - patience: Number of epochs to wait for improvement before stopping early
        """

        self.encoder.train()
        self.decoder.train()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0  # Track epochs without improvement
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        if val_data:
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        
        for step in range(epoch):
            train_loss = 0.0
            num_batches = 0

            for time_series_batch, cond_batch in train_loader:                
                X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)

                # Forward pass: encode -> decode
                H = self.encoder(X_mb_torch, cond_torch)
                X_tilde = self.decoder(H, cond_torch)

                # Compute loss
                mse = self.loss_mse(X_tilde, X_mb_torch)
                loss_ae = 10.0 * torch.sqrt(mse + 1e-7)  # Original paper scales loss

                # Backpropagation
                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()
                loss_ae.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()

                # Track loss for reporting
                train_loss += loss_ae.item()
                num_batches += 1

            train_loss /= max(num_batches, 1)  # Avoid division by zero

            # Compute validation loss if val_data is provided
            val_loss = None
            if val_data is not None:
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_batches = 0
                    for time_series_batch, cond_batch in val_loader:   
                        X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                        cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)
                        
                        H = self.encoder(X_mb_torch, cond_torch)
                        X_tilde = self.decoder(H, cond_torch)

                        mse_val = self.loss_mse(X_tilde, X_mb_torch)
                        val_loss += 10.0 * torch.sqrt(mse_val + 1e-7).item()
                        val_batches += 1

                    val_loss /= max(val_batches, 1)

                # Restore training mode
                self.encoder.train()
                self.decoder.train()

            # Logging
            if step % 10 == 0 or step == epoch - 1:
                if val_data is not None:
                    self.logger.info(f"[Autoencoder] Epoch {step}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    self.logger.info(f"[Autoencoder] Epoch {step}/{epoch}, Train Loss: {train_loss:.4f}")

            # Early stopping logic
            if val_data is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "opt_encoder": self.opt_encoder.state_dict(),
                        "opt_decoder": self.opt_decoder.state_dict(),
                    }
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience:
                    self.logger.info(f"[Autoencoder] Early stopping at epoch {step} with best Val Loss: {best_val_loss:.4f}")
                    break  # Stop training if patience is exceeded

        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.encoder.load_state_dict(best_model_state["encoder"])
            self.decoder.load_state_dict(best_model_state["decoder"])
            self.opt_encoder.load_state_dict(best_model_state["opt_encoder"])
            self.opt_decoder.load_state_dict(best_model_state["opt_decoder"])
            self.logger.info("[Autoencoder] Restored best model from early stopping.")
    
    def train_supervisor(self,
                     train_data: np.ndarray,
                     val_data: np.ndarray = None,
                     epoch: int = 250,
                     patience: int = 10):
        """
        Phase 2: Train the supervisor (and possibly generator) with the supervised loss:
        MSE(H[:,1:,:], H_hat_supervise[:,:-1,:])

        Implements early stopping based on validation loss.

        Args:
            - train_data: Training dataset (numpy array of shape [N, seq_len, n_seq])
            - val_data: Validation dataset (optional, same shape as train_data)
            - epoch: Maximum number of training epochs
            - patience: Number of epochs to wait for improvement before stopping early
        """

        self.encoder.eval()  # Freeze encoder from Phase 1
        self.supervisor.train()
        self.generator.train()

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0  # Tracks epochs without improvement

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        if val_data:
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        for step in range(epoch):
            train_loss = 0.0
            num_batches = 0

            for time_series_batch, cond_batch in train_loader:                
                X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)

                # Get real embeddings from the (frozen) encoder
                with torch.no_grad():
                    H = self.encoder(X_mb_torch, cond_torch)

                # Supervisor tries to predict next-step embeddings
                H_hat_supervise = self.supervisor(H, cond_torch)

                # The supervised loss is MSE of next-step embeddings
                if self.seq_len > 1:
                    sup_loss = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                else:
                    sup_loss = torch.tensor(0.0, device=self.device)

                # Backprop
                self.opt_generator.zero_grad()
                self.opt_supervisor.zero_grad()
                sup_loss.backward()
                self.opt_generator.step()
                self.opt_supervisor.step()

                # Track loss for reporting
                train_loss += sup_loss.item()
                num_batches += 1

            train_loss /= max(num_batches, 1)  # Avoid division by zero

            # Compute validation loss if val_data is provided
            val_loss = None
            if val_data is not None:
                self.supervisor.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_batches = 0
                    for time_series_batch, cond_batch in val_loader:                
                        X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                        cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)

                        H = self.encoder(X_mb_torch, cond_torch)
                        H_hat_supervise = self.supervisor(H, cond_torch)

                        if self.seq_len > 1:
                            sup_loss_val = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                        else:
                            sup_loss_val = torch.tensor(0.0, device=self.device)

                        val_loss += sup_loss_val.item()
                        val_batches += 1

                    val_loss /= max(val_batches, 1)

                # Restore training mode
                self.supervisor.train()

            # Logging
            if step % 10 == 0 or step == epoch - 1:
                if val_data is not None:
                    self.logger.info(f"[Supervisor] Epoch {step}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    self.logger.info(f"[Supervisor] Epoch {step}/{epoch}, Train Loss: {train_loss:.4f}")

            # Early stopping logic
            if val_data is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        "supervisor": self.supervisor.state_dict(),
                        "generator": self.generator.state_dict(),
                        "opt_supervisor": self.opt_supervisor.state_dict(),
                        "opt_generator": self.opt_generator.state_dict(),
                    }
                    patience_counter = 0  # Reset counter
                else:
                    patience_counter += 1  # Increment counter

                if patience_counter >= patience:
                    self.logger.info(f"[Supervisor] Early stopping at epoch {step} with best Val Loss: {best_val_loss:.4f}")
                    break  # Stop training if patience is exceeded

        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.supervisor.load_state_dict(best_model_state["supervisor"])
            self.generator.load_state_dict(best_model_state["generator"])
            self.opt_supervisor.load_state_dict(best_model_state["opt_supervisor"])
            self.opt_generator.load_state_dict(best_model_state["opt_generator"])
            self.logger.info("[Supervisor] Restored best model from early stopping.")
            
    def _compute_validation_loss(self, val_data: np.ndarray) -> float:
        """
        Compute a scalar validation loss for early stopping.
        For instance, sum (G_loss + D_loss) across the val set, 
        or just G_loss, etc.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.generator.eval()
        self.supervisor.eval()
        self.discriminator.eval()

        total_loss = 0.0
        num_batches = 0
        
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for time_series_batch, cond_batch in val_loader:                
                X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)

                # 1. Encode real data
                H = self.encoder(X_mb_torch, cond_torch)

                # 2. Generate synthetic data
                Z = self._random_generator(batch_size=X_mb_torch.size(0))
                E_hat = self.generator(Z, cond_torch)
                H_hat = self.supervisor(E_hat, cond_torch)
                X_hat = self.decoder(H_hat, cond_torch)

                # 3. Compute G-related losses
                Y_fake = self.discriminator(H_hat, cond_torch)
                Y_fake_e = self.discriminator(E_hat, cond_torch)
                valid = torch.ones_like(Y_fake, device=self.device)
                g_loss_u = self.loss_bce(Y_fake, valid)
                g_loss_u_e = self.loss_bce(Y_fake_e, valid)

                H_hat_supervise = self.supervisor(H, cond_torch)
                if self.seq_len > 1:
                    g_loss_s = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                else:
                    g_loss_s = torch.tensor(0.0, device=self.device)

                x_hat_mean = torch.mean(X_hat, dim=0)
                x_mean = torch.mean(X_mb_torch, dim=0)
                x_hat_var = torch.var(X_hat, dim=0)
                x_var = torch.var(X_mb_torch, dim=0)
                g_loss_v1 = torch.mean(torch.abs(torch.sqrt(x_hat_var + 1e-6) - torch.sqrt(x_var + 1e-6)))
                g_loss_v2 = torch.mean(torch.abs(x_hat_mean - x_mean))
                g_loss_v = g_loss_v1 + g_loss_v2

                G_loss = (g_loss_u 
                          + self.gamma * g_loss_u_e 
                          + 100.0 * torch.sqrt(g_loss_s + 1e-7) 
                          + 100.0 * g_loss_v)

                # 4. Compute D-related losses
                Y_real = self.discriminator(H, cond_torch)
                d_loss_real = self.loss_bce(Y_real, torch.ones_like(Y_real, device=self.device))
                d_loss_fake = self.loss_bce(Y_fake, torch.zeros_like(Y_fake, device=self.device))
                d_loss_fake_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e, device=self.device))
                D_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e

                # 5. Combine G_loss & D_loss
                val_step_loss = G_loss + D_loss
                total_loss += val_step_loss.item()
                num_batches += 1

        # Restore training mode
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()

        return total_loss / max(num_batches, 1)
    
    def train_joint_network(self, 
                       train_data: np.ndarray, 
                       val_data: np.ndarray = None, 
                       epoch: int = 250, 
                       patience: int = 10):
        """
        Phase 3: Joint training of generator, supervisor, encoder (fine-tuning), and discriminator.
        This follows the main TimeGAN logic:
          - Update G, S, E steps more often
          - Update D step if D_loss > threshold (0.15, etc.)

        Includes early stopping based on a chosen validation metric (by default, 
        G_loss + D_loss) if `val_data` is provided.

        Args:
            - train_data: Training dataset [N, seq_len, n_seq]
            - val_data: Validation dataset (optional), same shape as train_data
            - epoch: Maximum number of epochs
            - patience: Number of epochs to wait for improvement before early stopping
        """

        # Unfreeze everything for joint training
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()

        # Early stopping trackers
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        
        for step in range(epoch):
            # ------------------------------------------------------------------
            # (A) Train generator (and supervisor, embedder) more frequently
            # ------------------------------------------------------------------
            # Typically, we train G twice per D iteration
            
            train_iter = iter(train_loader)  # Create an iterator
            
            for kk in range(2):
                for time_series_batch, cond_batch in train_loader:                
                    X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
                    cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)

                    # 1. Real embeddings (Encoder)
                    H = self.encoder(X_mb_torch, cond_torch)

                    # 2. Random noise -> Generator -> Supervisor
                    #    We combine the noise (Z) with the REAL condition from the batch
                    Z = self._random_generator(batch_size=X_mb_torch.size(0))
                    E_hat = self.generator(Z, cond_torch)
                    H_hat = self.supervisor(E_hat, cond_torch)

                    # 3. Reconstruct X via decode(H_hat)
                    X_hat = self.decoder(H_hat, cond_torch)

                    # 4. Discriminator predictions
                    Y_fake = self.discriminator(H_hat, cond_torch)
                    Y_real = self.discriminator(H, cond_torch)
                    Y_fake_e = self.discriminator(E_hat, cond_torch)

                    # ---------------------
                    # Generator Losses
                    # ---------------------
                    valid = torch.ones_like(Y_fake, device=self.device)
                    g_loss_u = self.loss_bce(Y_fake, valid)      # Fool D with H_hat
                    g_loss_u_e = self.loss_bce(Y_fake_e, valid)  # Fool D with E_hat

                    # Supervised loss
                    H_hat_supervise = self.supervisor(H, cond_torch)
                    if self.seq_len > 1:
                        g_loss_s = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                    else:
                        g_loss_s = torch.tensor(0.0, device=self.device)

                    # Two-moment matching
                    x_hat_mean = torch.mean(X_hat, dim=0)
                    x_mean = torch.mean(X_mb_torch, dim=0)
                    x_hat_var = torch.var(X_hat, dim=0)
                    x_var = torch.var(X_mb_torch, dim=0)
                    g_loss_v1 = torch.mean(torch.abs(torch.sqrt(x_hat_var + 1e-6) - torch.sqrt(x_var + 1e-6)))
                    g_loss_v2 = torch.mean(torch.abs(x_hat_mean - x_mean))
                    g_loss_v = g_loss_v1 + g_loss_v2

                    # Total Generator loss
                    G_loss = (g_loss_u 
                              + self.gamma * g_loss_u_e 
                              + 100.0 * torch.sqrt(g_loss_s + 1e-7) 
                              + 100.0 * g_loss_v)

                    # ---------------------
                    # Backprop Generator
                    # ---------------------
                    self.opt_generator.zero_grad()
                    self.opt_supervisor.zero_grad()
                    self.opt_encoder.zero_grad()
                    self.opt_decoder.zero_grad()

                    G_loss.backward()

                    self.opt_generator.step()
                    self.opt_supervisor.step()
                    self.opt_encoder.step()
                    self.opt_decoder.step()

            # ------------------------------------------------------------------
            # (B) Train discriminator (1 step)
            # ------------------------------------------------------------------
            time_series_batch, cond_batch = next(train_iter)
            if (time_series_batch is None) or (cond_batch is None):
                # If there's no more data in the generator, we skip
                continue
                
            X_mb_torch = torch.tensor(time_series_batch, dtype=torch.float32, device=self.device)
            cond_torch = torch.tensor(cond_batch, dtype=torch.float32, device=self.device)
            H = self.encoder(X_mb_torch, cond_torch)

            Z = self._random_generator(batch_size=X_mb_torch.size(0))
            E_hat = self.generator(Z, cond_torch)
            H_hat = self.supervisor(E_hat, cond_torch)

            Y_real = self.discriminator(H, cond_torch)
            Y_fake = self.discriminator(H_hat, cond_torch)
            Y_fake_e = self.discriminator(E_hat, cond_torch)

            d_loss_real = self.loss_bce(Y_real, torch.ones_like(Y_real, device=self.device))
            d_loss_fake = self.loss_bce(Y_fake, torch.zeros_like(Y_fake, device=self.device))
            d_loss_fake_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e, device=self.device))
            D_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e

            if D_loss.item() > 0.15:
                self.opt_discriminator.zero_grad()
                D_loss.backward()
                self.opt_discriminator.step()

            # ---------------------------------
            # (C) Validation & Early Stopping
            # ---------------------------------
            # We do this once per epoch (after finishing training steps)
            val_total_loss = None
            if val_data is not None:
                val_total_loss = self._compute_validation_loss(val_data)

                # Check if current val loss is better
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    # Save the model state
                    best_model_state = {
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "generator": self.generator.state_dict(),
                        "supervisor": self.supervisor.state_dict(),
                        "discriminator": self.discriminator.state_dict(),
                        "opt_encoder": self.opt_encoder.state_dict(),
                        "opt_decoder": self.opt_decoder.state_dict(),
                        "opt_generator": self.opt_generator.state_dict(),
                        "opt_supervisor": self.opt_supervisor.state_dict(),
                        "opt_discriminator": self.opt_discriminator.state_dict(),
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"[Joint] Early stopping at epoch {step} with best validation loss: {best_val_loss:.4f}")
                    break

            # ---------------------------------
            # (D) Logging
            # ---------------------------------
            if step % 10 == 0 or step == epoch - 1:
                log_msg = (f"[Joint] step {step}/{epoch}, D_loss={D_loss.item():.4f}, "
                           f"G_loss_u={g_loss_u.item():.4f}, "
                           f"G_loss_s={g_loss_s.item():.4f}, "
                           f"G_loss_v={g_loss_v.item():.4f}")
                if val_data is not None and val_total_loss is not None:
                    log_msg += f", Val Loss={val_total_loss:.4f}"
                self.logger.info(log_msg)

        # Restore best model if early stopping occurred
        if best_model_state is not None and patience_counter < patience:
            # If we never triggered early stopping, the best model is from the last improvement
            self.encoder.load_state_dict(best_model_state["encoder"])
            self.decoder.load_state_dict(best_model_state["decoder"])
            self.generator.load_state_dict(best_model_state["generator"])
            self.supervisor.load_state_dict(best_model_state["supervisor"])
            self.discriminator.load_state_dict(best_model_state["discriminator"])
            self.opt_encoder.load_state_dict(best_model_state["opt_encoder"])
            self.opt_decoder.load_state_dict(best_model_state["opt_decoder"])
            self.opt_generator.load_state_dict(best_model_state["opt_generator"])
            self.opt_supervisor.load_state_dict(best_model_state["opt_supervisor"])
            self.opt_discriminator.load_state_dict(best_model_state["opt_discriminator"])

            self.logger.info("[Joint] Restored best model from early stopping.")
            
        if val_data:
            return best_val_loss
            
    def save_model(self, 
                   save_dir: str = './model_checkpoints/'):
        """
        Saves the trained TimeGAN model, including:
        - Model architecture & weights (encoder, decoder, generator, supervisor, discriminator)
        - Optimizer states (for continued training)
        - Model ID (for reference)

        Args:
            model_path (str): Path to save the model (e.g., "checkpoints/timegan.pth").
        """
        save_dir = os.path.join(save_dir, self.dataset_name)
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, self.model_id)

        # Prepare dictionary of model state
        model_state = {
            "model_id": self.model_id,  # Save model ID for tracking
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "generator": self.generator.state_dict(),
            "supervisor": self.supervisor.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_encoder": self.opt_encoder.state_dict(),
            "opt_decoder": self.opt_decoder.state_dict(),
            "opt_generator": self.opt_generator.state_dict(),
            "opt_supervisor": self.opt_supervisor.state_dict(),
            "opt_discriminator": self.opt_discriminator.state_dict(),
        }
        
        # Save model state
        torch.save(model_state, model_path)

        self.logger.info(f"[Model Saved] Successfully saved model to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Loads a saved TimeGAN model, including:
        - Model weights (encoder, decoder, generator, supervisor, discriminator)
        - Optimizer states (to resume training if needed)
        - Model ID

        Args:
            model_path (str): Path to the saved model file (e.g., "checkpoints/timegan.pth").
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        # Load checkpoint
        model_state = torch.load(model_path, map_location=self.device)

        # Restore model ID
        self.model_id = model_state.get("model_id", "unknown_model")

        # Restore model weights
        self.encoder.load_state_dict(model_state["encoder"])
        self.decoder.load_state_dict(model_state["decoder"])
        self.generator.load_state_dict(model_state["generator"])
        self.supervisor.load_state_dict(model_state["supervisor"])
        self.discriminator.load_state_dict(model_state["discriminator"])

        # Restore optimizers
        self.opt_encoder.load_state_dict(model_state["opt_encoder"])
        self.opt_decoder.load_state_dict(model_state["opt_decoder"])
        self.opt_generator.load_state_dict(model_state["opt_generator"])
        self.opt_supervisor.load_state_dict(model_state["opt_supervisor"])
        self.opt_discriminator.load_state_dict(model_state["opt_discriminator"])

        self.logger.info(f"[Model Loaded] Successfully loaded model from {model_path} (ID: {self.model_id})")
            
    def train(self, train_data: np.ndarray,
              val_data: np.ndarray = None,
              ae_iters: int = 250,
              sup_iters: int = 250,
              joint_iters: int = 250,
              patience: int = 15,
              save_dir: str = './model_checkpoints/'):
        """
        Wrapper to run the three phases in sequence.
        
        train_data: tuple, first element is the time-series data that is np.array(n, seq_len, n_seq), second element is np.array(n, cond_dim)
        
        """
        # Phase 1: Autoencoder training
        self.logger.info("[Training] Phase 1: Autoencoder")
        self.train_autoencoder(train_data, val_data, epoch=ae_iters, patience=patience)
        
        # Phase 2: Supervised training
        self.logger.info("[Training] Phase 2: Supervisor")
        self.train_supervisor(train_data, val_data, epoch=sup_iters, patience=patience)
        
        # Phase 3: Joint training
        self.logger.info("[Training] Phase 3: Joint")
        if val_data:
            best_val_loss = self.train_joint_network(train_data, val_data, epoch=joint_iters, patience=patience)
        else:
            self.train_joint_network(train_data, val_data, epoch=joint_iters, patience=patience)
                
        # Save model
        self.save_model(save_dir)
        
        if val_data:
            return best_val_loss
    
    def generate(self, num_samples: int, condition: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate synthetic sequences using the trained generator + supervisor + decoder.

        Args:
            - num_samples (int): Number of sequences to generate.
            - condition (np.ndarray or None): Conditioning variable (shape: [num_samples, cond_dim]). 
              If None, a random condition will be generated.

        Returns:
            - syn_data (np.ndarray): Generated synthetic sequences of shape [num_samples, seq_len, n_seq].
        """
        self.generator.eval()
        self.supervisor.eval()
        self.decoder.eval()

        with torch.no_grad():
            # Generate random noise for generator input
            Z = torch.randn(num_samples, self.seq_len, self.n_seq, device=self.device)

            # Handle conditioning
            if condition is None:
                cond = torch.randn(num_samples, self.cond_dim, device=self.device)  # Random condition
            else:
                # Convert provided condition to tensor
                cond = torch.from_numpy(condition).float().to(self.device)
                if cond.shape[0] != num_samples or cond.shape[1] != self.cond_dim:
                    raise ValueError(f"Condition shape mismatch: Expected ({num_samples}, {self.cond_dim}), got {cond.shape}")

            # Generate synthetic latent embeddings
            E_hat = self.generator(Z, cond)
            H_hat = self.supervisor(E_hat, cond)
            X_hat = self.decoder(H_hat, cond)  # shape: [num_samples, seq_len, n_seq]

        return X_hat.cpu().numpy()