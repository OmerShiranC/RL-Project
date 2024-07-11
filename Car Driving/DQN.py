# Machine Learning and Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
        def __init__(self, road_env, carenv, settings, train_mode):
        self.actions = actions
        self.road_env = road_env     