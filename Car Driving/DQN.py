# Machine Learning and Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, road_env, carenv, settings, train_mode):
        self.actions = actions
        self.road_env = road_env
        self.carenv = carenv
        self.settings = settings
        self.train_mode = train_mode

        # Define the device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Define the neural network
        super(PolicyNetwork, self).__init__()

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.settings.state_dim, hidden_layer[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], self.settings.action_dim))

        # Combine all layers
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, state):
        return self.model(state)


    def get_action(self, state): # epsilon greedy
        if np.randome.rand() < self.settings.epsilon:
            return np.random.choice(self.actions)
        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action = self.forward(state)
            action = torch.argmax(action).item()
        return action

   def plot_training_progress( all_rewards,first):
        if first:
            fig, ax = plt.subplots(figsize=(20, 7))
            line, = ax.plot([], [])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Training Progress')
        else:
            line.set_xdata(range(len(all_rewards)))
            line.set_ydata(all_rewards)
            ax.relim()
            ax.autoscale_view()
            display.clear_output(wait=True)
            display.display(fig)
            plt.pause(0.1)


    def train(self, num_episodes):
        all_rewards = []
        plot_training_progress(self, all_rewards, first=True)

        # we can add more complex trainig rate scheduling
        optimizer = optim.Adam(self.model.parameters(), lr=self.settings.learning_rate)
        criterion = nn.MSELoss()

        for episode in range(num_episodes):
            total_reward = 0

            state = self.carenv.reset()
            state = torch.FloatTensor(self.car_env.get_state()).to(self.device)

            while not self.car_env.Terminal:
                action = self.get_action(state)
                next_state, reward = self.car_env.step(action)
                next_state = torch.FloatTensor(next_state).to(self.device)

                total_reward += reward

                # Compute Q(s, a):
                Q_values = self.forward(state)
                Q_value = Q_values[action]

                # Compute Q(s', a')
                with torch.no_grad():
                   next_Q_values = self.forward(next_state)
                   next_Q_value = torch.max(next_Q_values)

                # Compute the target Q value
                target_Q_value = reward + self.settings.gamma * next_Q_value*(1 - self.car_env.Terminal)
                expected_Q_value = Q_value.clone()
                expected_Q_value[0, action] = target_Q_value

                # Compute the loss
                loss = criterion(Q_values, expected_Q_value)

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








