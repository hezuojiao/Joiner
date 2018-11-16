from __future__ import division
from __future__ import print_function

import os
import random
import logging


import torch

from dataset import Dataset
from agent import Agent
from env import Env
from memory import ReplayMemory
from trainer import Trainer
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    logger.debug(args)
    torch.manual_seed(random.randint(1, 10000))
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    split_files = os.path.join(args.data, args.split_file)
    dataset_file = os.path.join(args.data, 'dataset.pth')
    if os.path.isfile(dataset_file):
        train_dataset = torch.load(dataset_file)
    else:
        train_dataset = Dataset(split_files, args.in_dim)
        torch.save(train_dataset, dataset_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    # initialize environment, agent, memory
    env = Env(args, train_dataset)
    action_space = env.action_space()
    dqn = Agent(args, action_space)
    mem = ReplayMemory(args, args.memory_capacity)

    # create trainer object for training and testing
    trainer = Trainer(args, env, dqn, mem)

    if args.evaluate:
        # Evaluate step
        dqn.eval()
        for _ in range(args.evaluation_episodes):
            trainer.evaluate_one_step(train_dataset)
    else:
        # Training step
        trainer.train(train_dataset, logger)
        logger.debug('==> Checkpointing everything now...')
        dqn.save(os.path.join(args.save, args.expname))


if __name__ == "__main__":
    main()
