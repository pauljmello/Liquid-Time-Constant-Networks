import numpy as np
import torch
import torch.nn as nn


class LTCN:

    class Cell(nn.Module):
        """
        Liquid Time-Constant Cell:
        dx(t)/dt = -[1/τ + f(x(t), I(t))]x(t) + f(x(t), I(t))A
        """

        def __init__(self, input_dim, hidden_dim, tau, device):
            super().__init__()
            self.tau = nn.Parameter(torch.ones(hidden_dim, device=device) * tau)
            self.gamma = nn.Parameter(torch.randn(hidden_dim, input_dim, device=device) / np.sqrt(input_dim))
            self.gamma_r = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device) / np.sqrt(hidden_dim))
            self.mu = nn.Parameter(torch.zeros(hidden_dim, device=device))
            self.A = nn.Parameter(torch.randn(hidden_dim, device=device) / np.sqrt(hidden_dim))
            self.device = device

        def forward(self, x, I, dt):
            x = x.to(self.device)
            I = I.to(self.device)

            recurrent_term = x @ self.gamma_r.T
            input_term = I @ self.gamma.T
            f_val = torch.tanh(recurrent_term + input_term + self.mu)

            # Fused ODE solver update
            numerator = x + dt * f_val * self.A
            denominator = 1.0 + dt * ((1.0 / self.tau) + f_val)
            denominator = torch.clamp(denominator, min=1e-8)
            return numerator / denominator

    class Network(nn.Module):
        """Liquid Time-Constant Networks"""

        def __init__(self, input_dim, hidden_dim, output_dim, tau, dt, steps, device):
            super().__init__()
            self.ltc = LTCN.Cell(input_dim, hidden_dim, tau, device)
            self.output_layer = nn.Linear(hidden_dim, output_dim, device=device)
            self.dt = dt
            self.steps = steps
            self.device = device
            self.train_losses = []
            self.val_losses = []

        def forward(self, x_seq):
            x_seq = x_seq.to(self.device)
            batch_size, seq_len, _ = x_seq.shape
            h = torch.zeros(batch_size, self.ltc.mu.shape[0], device=self.device)
            outputs = torch.zeros(batch_size, seq_len, self.output_layer.out_features, device=self.device)

            for t in range(seq_len):
                I_t = x_seq[:, t, :]
                effective_dt = self.dt / self.steps
                for s in range(self.steps):
                    h = self.ltc(h, I_t, effective_dt)
                outputs[:, t, :] = self.output_layer(h)

            return outputs


    @staticmethod
    def train(model, train_loader, val_loader, epochs, learning_rate, print_every, device):
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        max_grad_norm = 1.0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        model.train_losses = train_losses
        model.val_losses = val_losses

        return model, {'train': train_losses, 'val': val_losses}

    @staticmethod
    def create(input_dim, hidden_dim, output_dim, tau, dt, steps, device):
        return LTCN.Network(input_dim, hidden_dim, output_dim, tau, dt, steps, device)