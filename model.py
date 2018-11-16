import torch
import torch.nn as nn
import torch.nn.functional as F


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


# module for deep-q-network
class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.device = args.device
        self.in_dim = args.in_dim
        self.mem_dim = args.mem_dim
        self.hidden_dim = args.hidden_dim
        self.action_space = action_space

        self.childsumtreelstm = ChildSumTreeLSTM(self.in_dim, self.mem_dim)

        self.fc_h_v = nn.Linear(self.mem_dim, self.hidden_dim)
        self.fc_h_a = nn.Linear(self.mem_dim, self.hidden_dim)
        self.fc_z_v = nn.Linear(self.hidden_dim, 1)
        self.fc_z_a = nn.Linear(self.hidden_dim, action_space)

    def forward(self, inputs):
        states = torch.cat([self.childsumtreelstm(tree, plan.to(self.device))[0] for tree, plan in inputs])
        states = states.view(-1, self.mem_dim)
        v = self.fc_z_v(F.relu(self.fc_h_v(states)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(states)))  # Advantage stream

        v, a = v.view(-1, 1), a.view(-1, self.action_space)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        return q
