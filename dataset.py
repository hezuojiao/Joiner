import os

from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from tree import Tree


# Dataset class for neural_network
class Dataset(data.Dataset):
    def __init__(self, path, plan_size):
        super(Dataset, self).__init__()
        self.plan_size = plan_size
        self.joins = self.read_lines(os.path.join(path, 'joins.txt'))
        self.idx_mapping = self._list2map(self.joins)
        self.plans = self.read_plans(os.path.join(path, 'plans.txt'))
        self.trees = self.read_trees(os.path.join(path, 'parents.txt'))
        self.labels = self.read_labels(os.path.join(path, 'labels.txt'))
        self.dones = self.read_lines(os.path.join(path, 'dones.txt'), covert2tensor=True)

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        join = deepcopy(self.joins[index])
        tree = deepcopy(self.trees[index])
        plan = deepcopy(self.plans[index])
        label = deepcopy(self.labels[index])
        done = deepcopy(self.dones[index])

        return join, (tree, plan), label, done

    def read_plans(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_plan(line) for line in tqdm(f.readlines())]
        return sentences

    def read_plan(self, line):
        vec = list(map(int, line.strip().split(',')))
        return torch.tensor(vec, dtype=torch.float, device='cpu').view(-1, self.plan_size)

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.strip().split(',')))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.log(torch.tensor(labels, dtype=torch.float, device='cpu'))
            # labels = torch.div(labels - labels.mean(), labels.std())
        return labels

    @staticmethod
    def read_lines(filename, covert2tensor=False):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in tqdm(f.readlines())]
        if covert2tensor:
            lines = list(map(float, lines))
            lines = torch.tensor(lines, dtype=torch.float, device='cpu')
        return lines

    @staticmethod
    def _list2map(keys):
        values = list(range(len(keys)))
        return dict(zip(keys, values))
