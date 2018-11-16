import random
import torch
from torch import optim
import torch.nn as nn

from model import DQN


class Agent(object):
    def __init__(self, args, action_space):
        self.action_space = action_space
        self.batch_size = args.batch_size
        self.discount = args.discount

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)
        self.loss_func = nn.MSELoss()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return self.online_net([state]).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.05):  # High ε can reduce evaluation scores drastically
        return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

    def learn(self, mem):

        # Sample transitions
        states, actions, next_states, rewards = mem.sample(self.batch_size)

        q_eval = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_eval_next_a = self.online_net(next_states).argmax(1)
            q_next = self.target_net(next_states)
            q_target = rewards + self.discount * q_next.gather(1, q_eval_next_a.unsqueeze(1)).squeeze()

        loss = self.loss_func(q_eval, q_target)
        self.online_net.zero_grad()
        loss.backward()
        self.optimiser.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path):
        torch.save(self.online_net.state_dict(), path + '.pth')

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net([state])).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
