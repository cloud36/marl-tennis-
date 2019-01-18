import numpy as np
import random
import copy
import os
import yaml
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


class MADDPG(object):
    def __init__(self, state_size, action_size, num_agents, rand_seed, hyper):

        # Replay memory
        self.memory = ReplayBuffer(action_size, hyper['BUFFER_SIZE'],  hyper['BATCH_SIZE'], rand_seed)
        self.num_agents   = num_agents
        self.na_idx       = np.arange(2)
        self.action_size  = action_size
        self.act_size     = action_size * num_agents
        self.state_size   = state_size * num_agents
        self.l_agents     = [DDPG(state_size, action_size, rand_seed, hyper, self.num_agents, self.memory) for i in range(num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.l_agents, states, actions, rewards, next_states, dones)
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            na_filtered = self.na_idx[self.na_idx != i]
            others_states = states[na_filtered]
            others_actions = actions[na_filtered]
            others_next_states = next_states[na_filtered]
            agent.step(state, action, reward, next_state, done, others_states, others_actions, others_next_states)

    def act(self, states, add_noise=True):
        na_rtn = np.zeros([self.num_agents, self.action_size])
        for idx, agent in enumerate(self.l_agents):
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def reset(self):
        for agent in self.l_agents:
            agent.reset()

    def __getitem__(self, key):
        return self.l_agents[key]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, hyper, num_agents, memory):

        self.action_size = action_size
        self.num_agents  = num_agents
    
        # Actor Network (w/ Target Network)
        self.actor_local     = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target    = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyper['LR_ACTOR'])

        # Critic Network (w/ Target Network)
        self.critic_local     = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_target    = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyper['LR_CRITIC']) #, weight_decay=hyper['WEIGHT_DECAY'])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.t           = 0 
        self.memory      = memory

    def step(self, state, action, reward, next_state, done, others_states,others_actions, others_next_states):
        self.memory.add(state, action, reward, next_state, done, others_states, others_actions, others_next_states)
        self.t = (self.t + 1) % hyper['UPDATE_EVERY']
        if self.t == 0:
            if len(self.memory) > hyper['BATCH_SIZE']:
                experiences = self.memory.sample()
                self.learn(experiences, hyper['GAMMA'])

    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        (states, actions, rewards, next_states, dones, others_states,
         others_actions, others_next_states) = experiences 
        rewards_ = rewards
        all_states = torch.cat((states, others_states), dim=1).to(device)
        all_actions = torch.cat((actions, others_actions), dim=1).to(device)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(device)

        # --------------------------- update critic --------------------------- 
        l_all_next_actions = []
        l_all_next_actions.append(self.actor_target(states))
        l_all_next_actions.append(self.actor_target(others_states))
        all_next_actions = torch.cat(l_all_next_actions, dim=1).to(device)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions) 
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones)) 
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor --------------------------- 
        this_actions_pred = self.actor_local(states)
        others_actions_pred = self.actor_local(others_states)
        others_actions_pred = others_actions_pred.detach()
        actions_pred = torch.cat((this_actions_pred, others_actions_pred), dim=1).to(device)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ---------------------- 
        self.soft_update(self.critic_local, self.critic_target, hyper['TAU'])
        self.soft_update(self.actor_local, self.actor_target, hyper['TAU']) 
        
    def soft_update(self, local_model, target_model, tau): 
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)



class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu     = mu * np.ones(size)
        self.size   = size
        self.theta  = theta
        self.sigma  = sigma
        self.seed   = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) 
        dx += self.sigma * np.random.randn(self.size)  # normal distribution
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "other_states", "other_actions", "other_next_states"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, other_states, other_actions, other_next_states):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, other_states, other_actions, other_next_states)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        other_states = torch.from_numpy(np.vstack([e.other_states for e in experiences if e is not None])).float().to(device)
        other_actions = torch.from_numpy(np.vstack([e.other_actions for e in experiences if e is not None])).float().to(device)
        other_next_states = torch.from_numpy(np.vstack([e.other_next_states for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, other_states, other_actions, other_next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


