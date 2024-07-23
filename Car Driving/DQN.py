# Machine Learning and Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import os
import subprocess
import sys

import matplotlib.pyplot as plt
from IPython import display
import numpy as np

def get_device():
    if 'COLAB_TPU_ADDR' in os.environ:
        print("TPU detected. Setting up TPU...")
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            print("TPU libraries not found. Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "cloud-tpu-client==0.10",
                                   "torch==1.13.0",
                                   "https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-1.13-cp38-cp38-linux_x86_64.whl"])
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, road_env, car_env, settings, train_mode):
        self.road_env = road_env
        self.car_env = car_env
        self.settings = settings
        self.train_mode = train_mode

        # Define the device
        self.device = get_device()

        # Define the neural network
        super(PolicyNetwork, self).__init__()

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.settings.n_sensors, self.settings.hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(self.settings.hidden_layers) - 1):
            layers.append(nn.Linear(self.settings.hidden_layers[i], self.settings.hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(self.settings.hidden_layers[-1], self.settings.action_dim))

        # Combine all layers
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        return self.model(state)


    def get_action(self, state): # epsilon greedy
        if np.random.rand() < self.settings.epsilon:
            return np.random.choice(self.settings.actions)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            state = state.to(device=self.device)
            with torch.no_grad():
                action = self.forward(state)
            action = torch.argmax(action).item()
            return action

    def train(self, num_episodes):
        all_rewards = []
        plot_training_progress(all_rewards, first=True)

        # we can add more complex trainig rate scheduling
        optimizer = optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)
        criterion = nn.MSELoss()

        for episode in range(num_episodes):
            total_reward = 0

            state = self.car_env.car_reset()
            state = torch.FloatTensor(self.car_env.get_state())

            while not self.car_env.terminal:
                action = self.get_action(state)
                next_state, reward = self.car_env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)

                total_reward += reward

                # Compute Q(s, a):
                Q_values = self.forward(state)
                Q_value = Q_values[action]  # No need for np.round().astype(int) if action is already an integer

                # Compute Q(s', a')
                with torch.no_grad():
                    next_Q_values = self.forward(next_state)
                    next_Q_value = torch.max(next_Q_values)

                # Compute the target Q value
                target_Q_value = reward + self.settings.gamma * next_Q_value * (1 - self.car_env.terminal)

                # Update the expected Q value
                expected_Q_values = Q_values.clone()
                expected_Q_values[action] = target_Q_value

                # Compute the loss
                loss = criterion(Q_values, expected_Q_values)
                # Update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state

            all_rewards.append(total_reward)
            if episode % 10 == 0:
                plot_training_progress(all_rewards, first=False)
        #save the model
        torch.save(self.model.state_dict(), 'car_policy_model.pth')
        print('Model saved successfully')

def plot_training_progress(all_rewards, first):
    if first:
        fig, ax = plt.subplots(figsize=(20, 7))
        line, = ax.plot([], [])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Progress')
    else:
        if 'line' not in locals():
            line, = ax.plot([], [])
        line.set_xdata(range(len(all_rewards)))
        line.set_ydata(all_rewards)
        ax.relim()
        ax.autoscale_view()
        display.clear_output(wait=True)
        display.display(fig)
        plt.pause(0.1)