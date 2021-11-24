import copy 
import random
from typing import Dict, List,Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def rand_action():
    return [0.5+np.random.uniform(-1,0.5), 0.5+np.random.uniform(-1,0.5), np.random.random()-0.8, 0.5+np.random.uniform(-1,0.5), 0.5+np.random.uniform(-1,0.5), np.random.random()-0.8]


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, acts_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, acts_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        """Store the transition in buffer."""
        #print(obs)
        self.obs_buf[self.ptr][:] = obs
        self.next_obs_buf[self.ptr][:] = next_obs
        self.acts_buf[self.ptr][:] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self, 
        size: int, 
        mu: float = 0.0, 
        theta: float = 0.15, 
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,16,3)

        x = torch.randn(80,80).view(1,1,80,80)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 16)
        self.fc2 = nn.Linear(26, 6)

    def convs(self, images):
        x = F.max_pool2d(F.relu(self.conv(images)), (2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        #print(x.size())
        return x
    
    def forward(self, states):
        data = states[:, :10]
        images = torch.reshape(states[:,10:], (-1, 1, 80, 80))
        x1 = self.convs(images)
        x1 = x1.view(-1,self._to_linear)
        x1 = self.fc1(x1)
        x2 = data



        x = torch.cat((x1,x2), dim=1)
        x = self.fc2(x)
        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,16,3)

        x = torch.randn(80,80).view(1,1,80,80)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 16)
        self.fc2 = nn.Linear(32, 1)



    def convs(self, images):
        x = F.max_pool2d(F.relu(self.conv(images)), (2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, states, action):
        data = states[:, :10]
        images = torch.reshape(states[:,10:], (-1, 1, 80, 80))
        x1 = self.convs(images)
        x1 = x1.view(-1,self._to_linear)
        x1 = self.fc1(x1)
        x2 = data
        x3 = action
        #print(x1.size())
        #print(x2.size())
        #print(x3.size())

        #print(x1)
        #print(x2)
        #print(x3)
        x = torch.cat((x1,x2, x3), dim=1)

        x = self.fc2(x)

        return F.relu(x)

class DDPGAgent:
    """DDPGAgent 
    
    Attribute:
        obs_dim (int): number of observations
        action_dim (int): number of actions
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        memory_size: int,
        batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        load: bool,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
    ):
        """Initialize."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory = ReplayBuffer(self.obs_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.load = load
                
        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        if self.load:
            networks = torch.load("networks.pth")
            self.actor.load_state_dict(networks["Actor_state_dict"])
            self.actor_target.load_state_dict(networks["Target_Actor_state_dict"])
            self.critic.load_state_dict(networks["Critic_state_dict"])
            self.critic_target.load_state_dict(networks["Target_Critic_state_dict"])

        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0

        #metrics
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps:
            selected_action = rand_action()
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
            selected_action = selected_action[0]
        
        # add noise for exploration during training
        noise = self.noise.sample()
        selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        self.transition = [state[0], selected_action]
        
        return selected_action
    
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        #print("updating")
        device = self.device  # for shortening the following lines
        
        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        
        # train critic
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
                
        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # target update
        self._target_soft_update()
        
        return actor_loss.data, critic_loss.data
    
    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
    
    def save_models(self):
        torch.save({
            'Actor_state_dict': self.actor.state_dict(),
            'Critic_state_dict': self.critic.state_dict(),
            'Target_Actor_state_dict': self.actor_target.state_dict(),
            'Target_Critic_state_dict': self.critic_target.state_dict(),
            }, "networks.pth")





