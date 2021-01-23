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
import cv2
import time


DISPLAY_HEIGHT=400
DISPLAY_WIDTH=400
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

def pre_processing(image, width, height):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return image[None, :, :].astype(np.float32)

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


class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, 2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

def train(episodes):
    real_time_start = time.time()
    CPU_time_start = time.process_time()
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    criterion = torch.nn.MSELoss()
    current_game = game.DinoGame(None,400,400)
    image, reward, endgame = current_game.nextframe(0)
    
    image = pre_processing(image, 84,84)
    #image = torch.from_numpy(np.expand_dims(np.transpose(image,(2,0,1)),0))
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    #image=image.float()
    state=torch.cat(tuple(image for _ in range (4)))[None, :, :, :]
    #print(state.shape)
    replay_memory = []
    episode = 0
    
    while episode<episodes:
        pred = model(state)[0]
        epsilon = final_epsilon+((episodes-episode)*(initial_epsilon-final_epsilon)/episodes)
        take_random_action = random.random()<=epsilon
        if take_random_action:
            print('random')
            action = random.randint(0,1)
        else:
            action=torch.argmax(pred)

        next_image, reward, endgame = current_game.nextframe(action)
        #next_image = torch.from_numpy(np.expand_dims(np.transpose(next_image,(2,0,1)),0))
        #next_image = next_image.float()
        #next_state = torch.cat((state.squeeze(0)[3:,:,:], next_image))

        next_image = pre_processing(next_image, 84,84)

        #action = action.unsqueeze(0)
        #reward = torch.from_numpy(np.array([reward],dtype=np.float32)).unsqueeze(0)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        replay_memory.append([state, action, reward, next_state, endgame])
        if len(replay_memory) > replay_memory_size:
            del replay_memory[0]
        batch = random.sample(replay_memory, min(len(replay_memory), batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))
        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()
        state = next_state
        
        if episode %10 == 0:
            print("Episode: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            episode + 1, episodes, action, loss, epsilon, reward, torch.max(pred)))
        episode+=1

        if episode % 1000 == 0:
            real_time_elapsed = time.time()-real_time_start
            CPU_time_elapsed =time.process_time()-CPU_time_start
            print("Real time elapsed : {}, CPU time elapsed : {}".format(real_time_elapsed,CPU_time_elapsed))
            checkpoint_path = "checkpoint_{}.pt".format(episode)
            torch.save({
            'checkpoint_episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss,
            'real_time_elapsed' : real_time_elapsed,
            'CPU_time_elapsed' : CPU_time_elapsed
            }, checkpoint_path)


train(1000000)
