import numpy as np
import torch

from ltcn import LTCN
from utils import prepare_data, visualize_results, ensure_output_dir


def main():
    # Configs
    seed = 1001
    torch.manual_seed(seed)
    np.random.seed(seed)
    save_dir = 'images'
    print_every = 10

    # Dataset Parameters
    num_samples = 1000
    seq_length = 100
    max_time = 10.0
    train_split = 0.7

    # Model Parameters
    input_dim = 1
    hidden_dim = 32
    output_dim = 1
    tau = 5.0              # Base time constant
    dt = 0.01              # Time step for ODE solver
    steps = 10             # Number of solver steps per time point

    # Training Parameters
    batch_size = 512
    epochs = 250
    learning_rate = 0.003

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    ensure_output_dir(save_dir)

    print("Preparing data...")
    train_loader, val_loader, vis_sample = prepare_data(num_samples, seq_length, batch_size, max_time, train_split, seed)

    print("Creating model...")
    model = LTCN.create(input_dim, hidden_dim, output_dim, tau, dt, steps, device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    print("Training model...")
    model, losses = LTCN.train(model, train_loader, val_loader, epochs, learning_rate, print_every, device)

    print("Visualizing results...")
    visualize_results(model, vis_sample, save_dir, device)


if __name__ == "__main__":
    main()