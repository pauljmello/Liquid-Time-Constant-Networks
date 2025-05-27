import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from kalmanfilter import KalmanFilter
from kan import KAN
from ltcn import LTCN
from utils import prepare_data, ensure_output_dir


class Testbench:

    @staticmethod
    def save_loss_comparison(ltcn_losses, kan_losses, save_dir, filename='loss_comparison.png'):
        filepath = os.path.join(save_dir, filename)
        plt.figure(figsize=(10, 6))
        plt.plot(ltcn_losses['train'], 'b-', label='LTCN - Train')
        plt.plot(ltcn_losses['val'], 'b--', label='LTCN - Val')
        plt.plot(kan_losses['train'], 'r-', label='KAN - Train')
        plt.plot(kan_losses['val'], 'r--', label='KAN - Val')
        plt.title('Training and Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.savefig(filepath)
        plt.close()
        print(f"Saved loss comparison to {filepath}")

    @staticmethod
    def save_prediction_comparison(ltcn_model, kan_model, kf_model, sample_data, device, save_dir, filename='prediction_comparison.png'):
        filepath = os.path.join(save_dir, filename)
        X_sample, y_true = sample_data
        X_sample = X_sample.to(device)

        ltcn_model.eval()
        kan_model.eval()
        kf_model.eval()

        with torch.no_grad():
            # LTCN predictions
            ltcn_preds = ltcn_model(X_sample)

            # KAN predictions
            batch_size, seq_len, feat_dim = X_sample.shape
            X_reshaped = X_sample.reshape(batch_size * seq_len, feat_dim)
            kan_preds_flat = kan_model(X_reshaped)
            kan_preds = kan_preds_flat.reshape(batch_size, seq_len, feat_dim)

            # Kalman Filter predictions
            kf_preds = kf_model(X_sample)

        y_true = y_true.cpu().numpy()
        ltcn_preds = ltcn_preds.cpu().numpy()
        kan_preds = kan_preds.cpu().numpy()
        kf_preds = kf_preds.cpu().numpy()

        plt.figure(figsize=(12, 12))

        plt.subplot(3, 1, 1)
        plt.plot(y_true[0, :, 0], 'k-', label='True')
        plt.plot(ltcn_preds[0, :, 0], 'b--', label='LTCN')
        plt.title('LTCN Model Predictions')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(y_true[0, :, 0], 'k-', label='True')
        plt.plot(kan_preds[0, :, 0], 'r--', label='KAN')
        plt.title('KAN Model Predictions')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(y_true[0, :, 0], 'k-', label='True')
        plt.plot(kf_preds[0, :, 0], 'g--', label='Kalman Filter')
        plt.title('Kalman Filter Predictions (Included For Fun)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"Saved prediction comparison to {filepath}")


def run_benchmark():
    # Config
    seed = 1001
    torch.manual_seed(seed)
    np.random.seed(seed)
    save_dir = '../images'
    print_every = 10

    # Dataset Parameters
    num_samples = 1000
    seq_length = 100
    max_time = 10.0
    train_split = 0.7

    # Shared Model Parameters
    input_dim = 1
    output_dim = 1

    # Shared Training Parameters
    epochs = 250
    batch_size = 256
    learning_rate = 0.01

    # LTCN Parameters
    ltcn_hidden_dim = 32
    ltcn_tau = 5.0      # Base time constant
    ltcn_dt = 0.01      # Time step for ODE solver
    ltcn_steps = 10     # Number of solver steps per time point

    # KAN Parameters
    kan_width = [input_dim, 8, 8, output_dim]   # [input dims, <- hidden layer dims ->, output dims]
    kan_grid_size = 5       # Grid Resolution
    kan_spline_degree = 3   # Spline Degree: Smoothness
    reg_lambda = 1e-4

    # Kalman Filter Parameters
    kf_dt = 0.01
    kf_damping = 0.1         # Damping factor for oscillations
    kf_frequency = 1.0       # Frequency of oscillations
    kf_process_noise = 0.01
    kf_obs_noise = 0.03
    kf_state_dim = 3         # State dimension default: [position, velocity, damping], generalized to higher dimensions
    kf_obs_dim = 1           # Observation dimension default: [position], generalized to higher dimensions

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ensure_output_dir(save_dir)

    # Data
    print("Preparing data...")
    train_loader, val_loader, vis_sample = prepare_data(num_samples, seq_length, batch_size, max_time, train_split, seed)

    # LTCN
    print("\nCreating and training LTCN model...")
    ltcn_model = LTCN.create(input_dim, ltcn_hidden_dim, output_dim, ltcn_tau, ltcn_dt, ltcn_steps, device)
    ltcn_params = sum(p.numel() for p in ltcn_model.parameters())
    print(f"LTCN parameters: {ltcn_params}")
    ltcn_model, ltcn_losses = LTCN.train(ltcn_model, train_loader, val_loader, epochs, learning_rate, print_every, device)

    # KAN
    print("\nCreating and training KAN model...")
    kan_model = KAN.create(kan_width, kan_grid_size, kan_spline_degree, device)
    kan_params = sum(p.numel() for p in kan_model.parameters())
    print(f"KAN parameters: {kan_params}")
    kan_model, kan_losses = KAN.train(kan_model, train_loader, val_loader, epochs, learning_rate, print_every, device, reg_lambda)

    # Kalman Filter
    print("\nApplying Kalman Filter...")
    kf_model = KalmanFilter.create( kf_dt, kf_damping, kf_frequency, kf_process_noise, kf_obs_noise, kf_state_dim, kf_obs_dim, device)
    kf_params = sum(p.numel() for p in kf_model.parameters())
    print(f"Kalman Filter parameters: {kf_params}")

    # Compare
    print("\nGenerating comparison visualizations...")
    Testbench.save_loss_comparison(ltcn_losses, kan_losses, save_dir)
    Testbench.save_prediction_comparison(ltcn_model, kan_model, kf_model, vis_sample, device, save_dir)

    print(f"\nFinal results:")
    print(f"LTCN validation loss: {ltcn_losses['val'][-1]:.6f}")
    print(f"KAN validation loss: {kan_losses['val'][-1]:.6f}")

    # Parameter Counts
    print(f"\nModel size comparison:")
    print(f"LTCN: {ltcn_params} parameters")
    print(f"KAN: {kan_params} parameters")
    print(f"Kalman Filter: {kf_params} parameters (state_dim={kf_state_dim}, obs_dim={kf_obs_dim})")

    print("\nNote:")
    print("The Kalman Filter is included for fun. \nIt is fundamentally different than networks like KAN and LTCN, \n...but not so different.")

    return {'ltc_model': ltcn_model, 'kan_model': kan_model, 'kf_model': kf_model, 'ltc_losses': ltcn_losses, 'kan_losses': kan_losses}


if __name__ == '__main__':
    run_benchmark()