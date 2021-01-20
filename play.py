import pygame
import game
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


DISPLAY_HEIGHT=400
DISPLAY_WIDTH=800
STATE_HEIGHT = DISPLAY_HEIGHT-1
STATE_WIDTH = DISPLAY_WIDTH-1

pygame.init()

image_size=84
batch_size=32
lr=1e-6
gamma=0.99
initial_epsilon=0.1
final_epsilon=1e-4
num_iters=2000000
replay_memory_size=50000


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def train(episodes):
    model = DQN(STATE_HEIGHT,STATE_WIDTH,2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = torch.nn.MSELoss()
    current_game = game.DinoGame(800,400)
    state, reward, endgame = current_game.nextframe(0)
    state = torch.from_numpy(state)
    replay_memory = []
    episode = 0
    while episode<episodes:
        pred = model(state)[0]
        epsilon = final_epsilon+((episodes-episode)*(initial_epsilon-final_epsilon)/episodes)
        take_random_action = random.random()<=epsilon
        if take_random_action:
            action = random.randint(0,1)
        else:
            action=torch.argmax(pred)[0]
        next_state, reward, endgame = current_game.nextframe(action)
        next_state = torch.from_numpy(state)
        replay_memory.append([state, action, reward, next_state, endgame])
        if len(replay_memory) > replay_memory_size:
            del replay_memory[0]
        batch = random.sample(replay_memory, min(len(replay_memory), batch_size))

        

