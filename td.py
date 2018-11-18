import numpy as np

"""
    MC Policy evaluation and TD(1) for a MDP environment with a finite number of states
"""


class MCPolicyEvaluation:
    def __init__(self, policy, env, mu_0, discount):
        """
        :param policy: policy to evaluate using MC Evaluation
        :param env: object with method step(state, action) that must return state, reward, done
        :param mu_0: initial state
        :param discount: Discount rate
        """
        self.policy = policy
        self.env = env
        self.mu_0 = mu_0
        self.discount = discount
        self.gains_maps = [[] for _ in range(len(self.mu_0))]
        self.V_history = []
        self.V = None

    def evaluate(self, n_episodes, t_max):
        for ep in range(n_episodes):
            initial_state = np.random.choice(np.arange(len(self.mu_0)), replace=True, p=self.mu_0)
            state = initial_state

            states_traj = []
            actions_traj = []
            rewards_traj = []
            for _ in range(t_max):
                # Assuming deterministic policy
                action = self.policy[state]

                states_traj.append(state)
                actions_traj.append(action)
                state, reward, done = self.env.step(state, action)
                rewards_traj.append(reward)

                if done:
                    break

            # First-Visit Update
            gains = 0.0
            t = -1
            for state, associated_reward in zip(states_traj[::-1], rewards_traj[::-1]):
                gains += self.discount*associated_reward
                t -= 1
                if state not in states_traj[:t]:
                    self.gains_maps[state].append(gains)

            self.V = [np.mean(gains) for gains in self.gains_maps]
            self.V_history.append(self.V)


def ok_state_actions(state_actions, n_actions):
    """
    """
    mask_ok = np.zeros((len(state_actions), n_actions))
    for state, li in enumerate(state_actions):
        mask_ok[state, li] = 1
    return mask_ok


class DiscreteTD:
    def __init__(self, env, mu_0, discount, eps, n_states, n_actions, lr_power):
        """

        Because it is much more easier to work with Q represented as a matrix,
        we will do it but be sensible to the fact that not all actions are accessible from
        one state.
        :param env: object with method step(state, action) that must return state, reward, done
        :param mu_0: Initial state distribution chosen
        :param discount: Discount rate
        :param eps: epsilon parameter of eps-greedy policy
        :param n_states: number of states
        :param n_actions: number of actions
        :param lr_power: gamma parameter of the learning rate lr(x, a) = 1 / counts(x, a)**gamma
        """
        self.discount = discount
        self.eps = eps
        self._counts = np.ones(shape=(n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_power = lr_power
        self.env = env
        self.mu_0 = mu_0

        self.Q = np.zeros(shape=(n_states, n_actions))
        self.mask_ok = ok_state_actions(env.state_actions, n_actions).astype(bool)
        self.Q[np.where(self.mask_ok==False)] = -np.inf

        self.gains_history = []
        self.V_history = []

    def learn(self, n_episodes, t_max):
        for ep in range(n_episodes):
            initial_state = np.random.choice(np.arange(len(self.mu_0)), replace=True, p=self.mu_0)
            state = initial_state

            gains = 0.0
            for _ in range(t_max):
                old_state = state
                action = self.eps_greedy_policy(state)
                state, reward, done = self.env.step(state, action)
                gains += reward
                self.policy_update(reward_old=reward,
                                   state_old=old_state,
                                   action_old=action,
                                   state_new=state)

                if done:
                    break

            self.V_history.append(self.Q.max(axis=-1))
            self.gains_history.append(gains)

    def alpha(self, x, a):
        val = 1.0 / (self._counts[x, a])**self.lr_power
        self._counts[x, a] += 1
        return val

    def eps_greedy_policy(self, x):
        id_best = np.argmax(self.Q[x, :])

        n_possible_actions = self.mask_ok[x, :].sum()
        probas = (self.eps/n_possible_actions) * np.ones(self.n_actions)
        probas[self.mask_ok[x, :]==False] = 0.0

        probas[id_best] = 1.0 - self.eps + self.eps / n_possible_actions
        return np.random.choice(np.arange(self.n_actions), p=probas)

    def policy_update(self, reward_old, state_old, action_old, state_new):
        bellman_err = reward_old + self.discount * self.Q[state_new, :].max() - \
                      self.Q[state_old, action_old]
        self.Q[state_old, action_old] += + self.alpha(state_old, action_old) * bellman_err

