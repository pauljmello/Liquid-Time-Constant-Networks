import torch


class KalmanFilter:
    class Model(torch.nn.Module):
        def __init__(self, state_dim, obs_dim, device):
            super().__init__()
            self.state_dim = state_dim
            self.obs_dim = obs_dim
            self.device = device

            self.F = torch.nn.Parameter(torch.eye(state_dim), requires_grad=False)  # State transition
            self.H = torch.nn.Parameter(torch.zeros(obs_dim, state_dim), requires_grad=False)  # Observation
            self.Q = torch.nn.Parameter(torch.eye(state_dim), requires_grad=False)  # Process noise
            self.R = torch.nn.Parameter(torch.eye(obs_dim), requires_grad=False)  # Observation noise
            self.train_losses = []
            self.val_losses = []

        def forward(self, x_seq):
            """ Apply Kalman Filter
                1. State Prediction:     x_pred = F·x                           "predict next state from prior state"
                2. Error Covariance:     P_pred = F·P·F^T + Q                   "project error covariance matrix"
                3. Kalman Gain:          K = P_pred·H^T·inv(H·P_pred·H^T + R)   "optimal Kalman gain"
                4. State Update:         x = x_pred + K·(z - H·x_pred)          "measurement update"
            """
            x_seq = x_seq.to(self.device)
            batch_size, seq_len, input_features = x_seq.shape
            predictions = torch.zeros_like(x_seq)

            for b in range(batch_size):
                # Initial state estimate and error covariance
                x = torch.zeros(self.state_dim, 1, device=self.device)
                P = torch.eye(self.state_dim, device=self.device)

                # Process each time step
                for t in range(seq_len):
                    # Get observation z at time t
                    z = x_seq[b, t, :self.obs_dim].reshape(-1, 1)

                    # Prediction Step
                    x_pred = self.F @ x                         # State Prediction: F·x
                    P_pred = self.F @ P @ self.F.t() + self.Q   # Covariance Prediction: F·P·F^T + Q

                    # Update Step
                    y = z - self.H @ x_pred                     # z - H·x_pred
                    S = self.H @ P_pred @ self.H.t() + self.R   # H·P_pred·H^T + R
                    K = P_pred @ self.H.t() @ torch.inverse(S)  # Kalman Gain: P_pred·H^T·S^(-1)
                    x = x_pred + K @ y                          # State Update: x_pred + K·y

                    # Joseph covariance update (more stable)
                    I_KH = torch.eye(self.state_dim, device=self.device) - K @ self.H
                    P = I_KH @ P_pred @ I_KH.t() + K @ self.R @ K.t()       # (I-KH)·P_pred·(I-KH)^T + K·R·K^T

                    # Map state estimate to observation space for output
                    predicted_obs = self.H @ x

                    # Store predictions
                    pred_len = min(predicted_obs.shape[0], input_features)
                    predictions[b, t, :pred_len] = predicted_obs[:pred_len, 0]

            return predictions

    @staticmethod
    def create(dt, damping, frequency, process_noise, observation_noise, state_dim=3 , obs_dim=1, device='cpu'):
        model = KalmanFilter.Model(state_dim, obs_dim, device)

        # State Transition Matrix
        if state_dim >= 3:
            # Damped Oscillator for 3 Dimensions
            F = torch.eye(state_dim, device=device)
            F[0, 1] = dt                            # position += velocity * dt
            F[1, 0] = -frequency * frequency * dt   # velocity -= spring_force * dt
            F[1, 1] = 1 - 2 * damping * dt          # velocity *= damping_factor
            model.F.data = F
        else:
            # Dynamics subset for 1-2 Dimensions
            F = torch.eye(state_dim, device=device)
            if state_dim >= 2:
                F[0, 1] = dt                            # position += velocity * dt
                F[1, 0] = -frequency * frequency * dt   # velocity -= spring_force * dt
                F[1, 1] = 1 - 2 * damping * dt          # velocity *= damping_factor
            model.F.data = F

        H = torch.zeros(obs_dim, state_dim, device=device)
        for i in range(min(obs_dim, state_dim)):
            H[i, i] = 1.0
        model.H.data = H  # Identity Matrix

        # Initialize noise covariances
        model.Q.data = torch.eye(state_dim, device=device) * process_noise
        model.R.data = torch.eye(obs_dim, device=device) * observation_noise

        return model