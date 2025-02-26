import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import uuid
from datetime import datetime
from typing import Tuple

from sympy import simplify
from torch.utils.data import DataLoader, Dataset


# Example placeholder or import from your own logger utility
def create_logger():
    import logging
    logger = logging.getLogger("timegan")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger

# ----------------------------------------------------------------------------
# Example placeholder classes. Replace these with your actual model definitions.
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, _ = self.rnn(x)
        # last output state is out[:, -1, :] if needed
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Sigmoid()  # or your chosen activation
    def forward(self, h):
        out, _ = self.rnn(h)
        # map back to original dimension
        out = self.fc(out)
        out = self.act(out)
        return out

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()
    def forward(self, z):
        out, _ = self.rnn(z)
        out = self.act(self.fc(out))
        return out

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # As in TimeGAN, use (num_layers - 1) or simply reuse the same layering
        effective_layers = max(1, num_layers - 1)
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                           num_layers=effective_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()
    def forward(self, h):
        out, _ = self.rnn(h)
        out = self.act(self.fc(out))
        return out

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        out, _ = self.rnn(h)
        logits = self.fc(out)  # shape: [batch, seq_len, 1]
        return logits
# ----------------------------------------------------------------------------


class TimeGAN:
    def __init__(self, parameters):
        """
        parameters: dict with fields like:
            - seq_len
            - n_seq
            - hidden_dim
            - num_layers
            - lr
            - gamma
            - batch_size
            - device
            - dataset_name (optional)
            ...
        """
        self.logger = create_logger()
        
        # Extract parameters (update as needed)
        self.seq_len = parameters["seq_len"]
        self.n_seq = parameters["n_seq"]
        self.hidden_dim = parameters["hidden_dim"]
        self.num_layers = parameters["num_layers"]
        self.gamma = parameters.get("gamma", 1.0)
        self.batch_size = parameters.get("batch_size", 64)
        self.lr = parameters.get("lr", 0.001)
        self.dataset_name = parameters.get("dataset_name", "custom")
        
        self.device = parameters.get("device", torch.device("cpu"))
        
        # Build modules
        self.encoder = Encoder(self.n_seq, self.hidden_dim, self.num_layers).to(self.device)
        self.decoder = Decoder(self.hidden_dim, self.n_seq, self.num_layers).to(self.device)
        self.generator = Generator(self.n_seq, self.hidden_dim, self.num_layers).to(self.device)
        self.supervisor = Supervisor(self.hidden_dim, self.num_layers).to(self.device)
        self.discriminator = Discriminator(self.hidden_dim, self.num_layers).to(self.device)
        
        # You can chain encoder->decoder for an autoencoder if you wish
        # but it's often simpler to call them separately in training steps
        self.autoencoder = nn.Sequential(self.encoder, self.decoder).to(self.device)
        
        # Initialize weights
        self.encoder.apply(self._init_weights)
        self.decoder.apply(self._init_weights)
        self.generator.apply(self._init_weights)
        self.supervisor.apply(self._init_weights)
        self.discriminator.apply(self._init_weights)
        
        # Define optimizers
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=self.lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        # Alternatively, if you prefer a combined generator-supervisor optimizer:
        # self.opt_gen_super = optim.Adam(
        #     list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=self.lr
        # )
        
        # Loss functions
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCEWithLogitsLoss()  # typical for discriminator
        
        # If you have a stored scaler, load it here. Otherwise, remove this line.
        # from sklearn.externals import joblib  # or import joblib
        # self.scaler = joblib.load("models/scaler.pkl")
        
        # Generate a unique model ID using current date and UUID
        self.model_id = f"{self.dataset_name}_" \
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_" \
                        f"{uuid.uuid4().hex[:8]}_" \
                        f"n_layers{self.num_layers}_seq_len{self.seq_len}_n_seq{self.n_seq}"
        
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
                    # For LSTM, you might want to init forget gate bias = 1
                    if isinstance(m, nn.LSTM):
                        hidden_size = param.shape[0] // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def _batch_generator(self, data: np.ndarray, batch_size: int):
        """
        Simple example of a batch generator.
        data is assumed to be a numpy array of shape [N, seq_len, n_seq].
        Yields X_mb in random mini-batches.
        """
        N = data.shape[0]
        idx = np.random.permutation(N)
        for i in range(0, N, batch_size):
            batch_idx = idx[i : i + batch_size]
            X_mb = data[batch_idx]  # shape [batch, seq_len, n_seq]
            yield X_mb
    
    def _random_generator(self, batch_size: int) -> torch.Tensor:
        """
        Generates random noise of shape [batch_size, seq_len, n_seq].
        (In TimeGAN, z_dim is often the same as n_seq. Adjust if needed.)
        """
        return torch.randn(batch_size, self.seq_len, self.n_seq, device=self.device)
    
    def train_autoencoder(self,
                          data: np.ndarray,
                          epoch: int = 250):
        """
        Phase 1: Train the autoencoder (encoder + decoder) to learn initial embeddings.
        """
        self.encoder.train()
        self.decoder.train()
        
        for step in range(epoch):
            for X_mb in self._batch_generator(data, self.batch_size):
                X_mb_torch = torch.tensor(X_mb, dtype=torch.float32, device=self.device)
                
                # Forward pass: encode -> decode
                H = self.encoder(X_mb_torch)
                X_tilde = self.decoder(H)
                
                # MSE loss
                mse = self.loss_mse(X_tilde, X_mb_torch)
                # Original code often used a scaled version, e.g. 10*sqrt(mse)
                loss_ae = 10.0 * torch.sqrt(mse + 1e-7)
                
                # Backprop
                self.opt_encoder.zero_grad()
                self.opt_decoder.zero_grad()
                loss_ae.backward()
                self.opt_encoder.step()
                self.opt_decoder.step()
                
            # Print log occasionally
            if step % 10 == 0:
                self.logger.info(f"[Autoencoder] step {step}/{epoch}, loss_ae={loss_ae.item():.4f}")
    
    def train_supervisor(self,
                         data: np.ndarray,
                         epoch: int = 250):
        """
        Phase 2: Train the supervisor (and possibly generator) with the supervised loss:
        MSE(H[:,1:,:], H_hat_supervise[:,:-1,:])
        """
        self.encoder.eval()  # We freeze the encoder’s weights from Phase 1
        self.supervisor.train()
        self.generator.train()
        
        # If you prefer a single optimizer for generator + supervisor:
        # opt_gs = optim.Adam(
        #     list(self.generator.parameters()) + list(self.supervisor.parameters()),
        #     lr=self.lr
        # )
        
        for step in range(epoch):
            for X_mb in self._batch_generator(data, self.batch_size):
                X_mb_torch = torch.tensor(X_mb, dtype=torch.float32, device=self.device)
                
                # Get real embeddings from the (frozen) encoder
                with torch.no_grad():
                    H = self.encoder(X_mb_torch)  # shape: [batch, seq_len, hidden_dim]
                
                # Supervisor tries to predict next-step embeddings
                H_hat_supervise = self.supervisor(H)
                
                # The supervised loss is MSE of next-step embeddings
                if self.seq_len > 1:
                    sup_loss = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                else:
                    sup_loss = torch.tensor(0.0, device=self.device)
                
                # You may also want to incorporate generator’s random input:
                # Z = self._random_generator(batch_size=X_mb_torch.size(0))
                # E_hat = self.generator(Z)
                # H_hat_gen = self.supervisor(E_hat)
                # ... but in the original code, Phase 2 only trains supervised loss on real embeddings.
                
                # Backprop
                self.opt_generator.zero_grad()
                self.opt_supervisor.zero_grad()
                sup_loss.backward()
                self.opt_generator.step()
                self.opt_supervisor.step()
                
            # Print log occasionally
            if step % 10 == 0:
                self.logger.info(f"[Supervisor] step {step}/{epoch}, sup_loss={sup_loss.item():.4f}")
    
    def train_joint_network(self, data: np.ndarray, epoch: int = 250):
        """
        Phase 3: Joint training of generator, supervisor, encoder (fine-tuning), and discriminator.
        This follows the main TimeGAN logic:
          - Update G, S, E steps more often
          - Update D step if D_loss > threshold (0.15, etc.)
        """
        self.encoder.train()       # unfreeze for joint training
        self.decoder.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()
        
        for step in range(epoch):
            # ------------------------------------------------------------------
            # (A) Train generator (and supervisor, embedder) more frequently
            # ------------------------------------------------------------------
            for kk in range(2):  # Generator is trained twice as much as the discriminator
                for X_mb in self._batch_generator(data, self.batch_size):
                    X_mb_torch = torch.tensor(X_mb, dtype=torch.float32, device=self.device)
                    
                    # 1. Get real embeddings
                    H = self.encoder(X_mb_torch)
                    
                    # 2. Generate random noise and pass through G + S
                    Z = self._random_generator(batch_size=X_mb_torch.size(0))
                    E_hat = self.generator(Z)
                    H_hat = self.supervisor(E_hat)
                    
                    # 3. Reconstruct X via decode(H_hat)
                    X_hat = self.decoder(H_hat)
                    
                    # 4. Discriminator predictions
                    #    Y_fake = D(H_hat), Y_real = D(H), Y_fake_e = D(E_hat)
                    Y_fake = self.discriminator(H_hat)
                    Y_real = self.discriminator(H)
                    Y_fake_e = self.discriminator(E_hat)
                    
                    # Generator losses
                    valid = torch.ones_like(Y_fake, device=self.device)
                    fake = torch.zeros_like(Y_fake, device=self.device)
                    
                    g_loss_u = self.loss_bce(Y_fake, valid)     # G tries to fool D
                    g_loss_u_e = self.loss_bce(Y_fake_e, valid) # E_hat as well
    
                    # Supervised loss
                    H_hat_supervise = self.supervisor(H)
                    if self.seq_len > 1:
                        g_loss_s = self.loss_mse(H_hat_supervise[:, :-1, :], H[:, 1:, :])
                    else:
                        g_loss_s = torch.tensor(0.0, device=self.device)
                    
                    # Two-moment matching (means & variances)
                    x_hat_mean = torch.mean(X_hat, dim=0)
                    x_mean = torch.mean(X_mb_torch, dim=0)
                    x_hat_var = torch.var(X_hat, dim=0)
                    x_var = torch.var(X_mb_torch, dim=0)
                    g_loss_v1 = torch.mean(torch.abs(torch.sqrt(x_hat_var + 1e-6) - torch.sqrt(x_var + 1e-6)))
                    g_loss_v2 = torch.mean(torch.abs(x_hat_mean - x_mean))
                    g_loss_v = g_loss_v1 + g_loss_v2
                    
                    # Total Generator loss
                    G_loss = g_loss_u + self.gamma * g_loss_u_e + 100.0 * torch.sqrt(g_loss_s + 1e-7) + 100.0 * g_loss_v
                    
                    # Compute only one backward pass for G loss
                    self.opt_generator.zero_grad()
                    self.opt_supervisor.zero_grad()
                    self.opt_encoder.zero_grad()
                    self.opt_decoder.zero_grad()
                    
                    G_loss.backward()  # No retain_graph=True
                    
                    self.opt_generator.step()
                    self.opt_supervisor.step()
                    self.opt_encoder.step()
                    self.opt_decoder.step()
            
            # ------------------------------------------------------------------
            # (B) Train discriminator (1 step)
            # ------------------------------------------------------------------
            X_mb = next(self._batch_generator(data, self.batch_size), None)
            if X_mb is None:
                continue  # no more data in generator
            X_mb_torch = torch.tensor(X_mb, dtype=torch.float32, device=self.device)
            H = self.encoder(X_mb_torch)
            
            Z = self._random_generator(batch_size=X_mb_torch.size(0))
            E_hat = self.generator(Z)
            H_hat = self.supervisor(E_hat)
            
            Y_real = self.discriminator(H)
            Y_fake = self.discriminator(H_hat)
            Y_fake_e = self.discriminator(E_hat)
            
            d_loss_real = self.loss_bce(Y_real, torch.ones_like(Y_real, device=self.device))
            d_loss_fake = self.loss_bce(Y_fake, torch.zeros_like(Y_fake, device=self.device))
            d_loss_fake_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e, device=self.device))
            D_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e
            
            if D_loss.item() > 0.15:
                self.opt_discriminator.zero_grad()
                D_loss.backward()  # No retain_graph=True
                self.opt_discriminator.step()
            
            # Logging
            if step % 10 == 0:
                self.logger.info(
                    f"[Joint] step {step}/{epoch}, D_loss={D_loss.item():.4f}, "
                    f"G_loss_u={g_loss_u.item():.4f}, G_loss_s={g_loss_s.item():.4f}, "
                    f"G_loss_v={g_loss_v.item():.4f}"
                )
            
    def train(self, data: np.ndarray,
              ae_iters: int = 250,
              sup_iters: int = 250,
              joint_iters: int = 250):
        """
        Wrapper to run the three phases in sequence.
        """
        # Phase 1: Autoencoder training
        self.logger.info("[Training] Phase 1: Autoencoder")
        self.train_autoencoder(data, epoch=ae_iters)
        
        # Phase 2: Supervised training
        self.logger.info("[Training] Phase 2: Supervisor")
        self.train_supervisor(data, epoch=sup_iters)
        
        # Phase 3: Joint training
        self.logger.info("[Training] Phase 3: Joint")
        self.train_joint_network(data, epoch=joint_iters)
    
    def generate(self, num_samples: int) -> np.ndarray:
        """
        Generate synthetic sequences using the trained generator + supervisor + decoder.
        """
        self.generator.eval()
        self.supervisor.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            Z = torch.randn(num_samples, self.seq_len, self.n_seq, device=self.device)
            E_hat = self.generator(Z)
            H_hat = self.supervisor(E_hat)
            X_hat = self.decoder(H_hat)  # shape: [num_samples, seq_len, n_seq]
            
        syn_data = X_hat.cpu().numpy()
        
        # If you want to invert scaling:
        # syn_data = self.scaler.inverse_transform(... )

        return syn_data