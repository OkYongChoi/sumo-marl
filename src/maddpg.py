# Implementation of MADDPG Algorithm
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def update_model(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[64 for _ in range(2)]):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [obs_dim] + hidden
        output_dims = hidden + [action_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim=1, hidden=[64 for _ in range(2)]):
        super(Critic, self).__init__()

        self.layers = nn.ModuleList()
        input_dims = [state_dim + action_dim] + hidden
        output_dims = hidden + [output_dim]

        for in_dim, out_dim in zip(input_dims[:-1], output_dims[:-1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(input_dims[-1], output_dims[-1]))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        states, actions, next_states, rewards, dones = zip(*transitions)

        states = np.vstack(states)
        actions = np.vstack(actions)
        next_states = np.vstack(next_states)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, next_states, rewards, dones


class MADDPG(nn.Module):
    def __init__(self, n_agents, obs_dim, action_dim, lr=1e-3, gamma=0.99, batch_size=64, tau=1e-3):
        super(MADDPG, self).__init__()

        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.init_actors()
        self.init_critics()

        self.mse_loss = nn.MSELoss()
        self.init_optimizers()

        self.memory = ReplayMemory(capacity=5000)

        self.eps = 0.9
        self.eps_decay = 0.99
        self.eps_threshold = 0.01
        self.eps_update_step = 100
        self.step = 0

    def init_actors(self):
        self.actors = nn.ModuleList([Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)])
        self.target_actors = nn.ModuleList([Actor(self.obs_dim, self.action_dim) for _ in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.actors[i], self.target_actors[i], tau=1.0)

    def init_critics(self):
        self.critics = nn.ModuleList(
            [Critic(self.obs_dim * self.n_agents, self.action_dim * self.n_agents) for _ in range(self.n_agents)])
        self.target_critics = nn.ModuleList(
            [Critic(self.obs_dim * self.n_agents, self.action_dim * self.n_agents) for _ in range(self.n_agents)])
        for i in range(self.n_agents):
            update_model(self.critics[i], self.target_critics[i], tau=1.0)

    def init_optimizers(self):
        self.actor_optimizers = [optim.Adam(self.actors[i].parameters(), lr=self.lr) for i in range(self.n_agents)]
        self.critic_optimizers = [optim.Adam(self.critics[i].parameters(), lr=self.lr) for i in range(self.n_agents)]

    def select_action(self, obs, i):
        obs = torch.from_numpy(obs).float()
        if random.random() < self.eps:
            action = np.random.randint(self.action_dim)
            action_prob = np.zeros(2)
            action_prob[action] = 1.0
        else:
            action_prob = self.actors[i](obs).detach().numpy()
            action = np.argmax(action_prob)
        return action, action_prob

    def push(self, transition):
        before_state, actions, state, rewards, dones = transition
        before_state = before_state.flatten()
        actions = np.array(actions).flatten()
        state = state.flatten()
        transition = (before_state, actions, state, rewards, dones)
        self.memory.push(transition)

    def train_start(self):
        return len(self.memory) >= self.batch_size

    def train_model(self, i): # only update agent i's network
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)

        # make target q values 
        current_q = self.critics[i](states, actions)
        next_actions = []
        [ next_actions.append(self.actors[i](next_states.view(self.batch_size,-1,self.obs_dim)[:,i])) for _ in range(self.n_agents) ] # Each actor considers only its own state.
        next_actions = torch.hstack(next_actions)
        next_q = self.target_critics[i](next_states, next_actions).detach()
        target_q = rewards[:,i].view(self.batch_size, 1) + self.gamma * (1.0 - dones) * next_q
        
        # update critic network with MSE loss
        value_loss = self.mse_loss(target_q, current_q)
        self.critic_optimizers[i].zero_grad()
        value_loss.backward()
        self.critic_optimizers[i].step()
        
        # update actor network
        actions = []
        [ actions.append(self.actors[i](states.view(self.batch_size,-1,self.obs_dim)[:,i])) for _ in range(self.n_agents) ]
        actions = torch.hstack(actions)

        policy_loss = -self.critics[i](states, actions).mean()
        self.actor_optimizers[i].zero_grad()
        policy_loss.backward()
        self.actor_optimizers[i].step()

        # update target network 
        update_model(self.actors[i], self.target_actors[i], self.tau)        
        update_model(self.critics[i], self.target_critics[i], self.tau)

        return policy_loss.item(), value_loss.item()

    def update_eps(self):
        self.step += 1
        if self.step % self.eps_update_step == 0:
            if self.eps > self.eps_threshold:
                self.eps *= self.eps_decay
            else:
                self.eps = self.eps_threshold

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name))
