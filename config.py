import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Joiner')

    # data arguments
    parser.add_argument('--data', default='dataset/',
                        help='path to dataset')
    parser.add_argument('--split-file', default='split_files',
                        help='path to split files')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='logging',
                        help='Name to identify experiment')
    parser.add_argument('--load', action='store_true',
                        help='Load dqn model')

    # model arguments
    parser.add_argument('--in-dim', default=175, type=int,
                        help='Size of input plan operation vector')
    parser.add_argument('--mem-dim', default=200, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--hidden-dim', default=200, type=int,
                        help='Size of classifier MLP')
    parser.add_argument('--max-join-count', type=int, default=5,
                        help='Size of join count')

    # training arguments
    parser.add_argument('--T-max', type=int, default=int(2e3), metavar='STEPS',
                        help='Number of training steps')
    parser.add_argument('--episode', type=int, default=int(1e3), metavar='LENGTH',
                        help='Max episode length (0 to disable)')

    parser.add_argument('--memory-capacity', type=int, default=int(1e4), metavar='CAPACITY',
                        help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(2e2), metavar='τ',
                        help='Number of steps after which to update target network')

    parser.add_argument('--lr', type=float, default=0.001, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(1e2), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=100, metavar='STEPS',
                        help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                        help='Number of evaluation episodes to average over')
    parser.add_argument('--max-T-evaluation', type=int, default=10, metavar='N',
                        help='Number of evaluation steps to average over')

    # miscellaneous options
    parser.add_argument('--seed', default=301, type=int,
                        help='random seed (default: 301)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
