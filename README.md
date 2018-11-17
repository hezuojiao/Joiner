Joiner
=======

Joiner is a research project for sql join order tuning using reinforcement learning algorithm.
It aims to find optimal join order under a certain workload such as
[TPC-H](http://www.tpc.org/tpch/), [JOB](https://github.com/gregrahn/join-order-benchmark).

### What is Joiner in general?
The overall workflow of Joiner looks like this:
![overall](https://github.com/hezuojiao/Joiner/blob/master/images/overall.jpg)

### Acknowledgements
[Tree LSTM implementation in PyTorch](https://github.com/dasguptar/treelstm.pytorch) for the [Treelstm implementation](https://github.com/stanfordnlp/treelstm), and to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the fun library.

### License
MIT
