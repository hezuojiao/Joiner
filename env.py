from itertools import combinations
import torch


class Env(object):
    def __init__(self, args, dataset):
        self.device = args.device
        self.dataset = dataset
        self.mapping = dataset.idx_mapping
        actions = [c for c in combinations(range(args.max_join_count), 2)]
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))

    def get_rewards(self, join, label):
        rewards = []
        for a in range(self.action_space()):
            _, reward, _, _ = self.step(join, label, a)
            rewards.append(reward.item())
        return rewards

    def step(self, join, label, action):
        """

        :param join:
        :param label:
        :param action:
        :return:
            :state (tree, plan)
            :reward
            :done
            :info (join, label)
        """
        next_join = self._get_next_order(join, action)
        if next_join is None:
            # return a terminal state
            return (0, torch.zeros(1, 71, dtype=torch.float, device='cpu')), 0, True, (join, label)

        next_join, next_state, next_label, done = self.dataset[self.mapping[next_join]]
        reward = label - next_label
        next_info = (next_join, next_label)
        return next_state, reward, done, next_info

    def _get_next_order(self, order, action):
        order_prefix, order = order.split(':')[0], order.split(':')[1]
        order = order.strip().split('-')
        l_idx, r_idx = self.actions[action]
        if l_idx >= len(order) or r_idx >= len(order):
            return None
        order[l_idx], order[r_idx] = order[r_idx], order[l_idx]
        new_order = order_prefix + ":" + order[0]
        for i in range(1, len(order)):
            new_order += "-" + order[i]
        return new_order

    def action_space(self):
        return len(self.actions)
