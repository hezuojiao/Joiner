import random
import torch
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.transitions = []
        self.position = 0
        self.discount = args.discount

    def append(self, state, action, next_state, reward):

        transition = Transition(state, action, next_state, reward)
        if self.position >= len(self.transitions):
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition
        # walk insertion point through list
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.transitions, batch_size)

        states, actions, next_states, rewards = zip(*batch)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        return states, actions, next_states, rewards

    def __len__(self):
        return len(self.transitions)

