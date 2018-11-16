import random


class Trainer(object):
    def __init__(self, args, env, agent, mem):
        self.args = args
        self.device = args.device
        self.env = env
        self.dqn = agent
        self.mem = mem

    def train(self, dataset, logger):
        self.dqn.train()

        for episode in range(self.args.episode):
            current_step = 0

            idx = random.randint(0, len(dataset) - 1)
            join, state, label, done = dataset[idx]

            while current_step < self.args.T_max and (not done.item() or current_step == 0):

                action = self.dqn.act_e_greedy(state)
                # carry out action/observe reward
                next_state, reward, done, info = self.env.step(join, label, action)

                # store experience s, a, s', r in replay memory
                self.mem.append(state, action, next_state, reward)

                current_step += 1

                # Train and test
                if current_step >= self.args.learn_start:

                    if current_step % self.args.replay_frequency == 0:
                        self.dqn.learn(self.mem)  # Train with n-step distributional double-Q learning

                    if current_step % self.args.evaluation_interval == 0:
                        self.dqn.eval()  # Set DQN (online network) to evaluation mode
                        avg_reward = self._test(dataset)  # Test
                        logger.info("Episode = " + str(episode) +
                                    ' | T = ' + str(current_step) + ' / ' + str(self.args.T_max) +
                                    ' | Avg. reward: ' + str(avg_reward))
                        self.dqn.train()  # Set DQN (online network) back to training mode

                    # Update target network
                    if current_step % self.args.target_update == 0:
                        self.dqn.update_target_net()

                # move to next state
                state = next_state
                join, label = info

    def _test(self, dataset):
        idx = random.randint(0, len(dataset) - 1)
        current_step = 0
        t_rewards = []
        join, state, label, done = dataset[idx]

        while current_step < self.args.max_T_evaluation and not done.item():
            action = self.dqn.act(state)
            current_step += 1
            next_state, reward, done, info = self.env.step(join, label, action)
            t_rewards.append(reward.item())
            # move to next state
            state = next_state
            join, label = info

        if current_step == 0:
            return 0
        return sum(t_rewards) / current_step

    def evaluate_one_step(self, dataset, logger):
        done = 1
        while done:
            idx = random.randint(0, len(dataset) - 1)
            join, state, label, done = dataset[idx]

        action = self.dqn.act(state)
        q_value = self.dqn.evaluate_q(state)
        rewards = self.env.get_rewards(join, label)
        logger.info("Q-value: %s | Reward: %s | Max Reward: %s" %
                    (str(q_value), str(rewards[action]), str(max(rewards))))
