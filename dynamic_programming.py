import numpy as np

"""
    Algorithms useful in Dynamic Programming.
    In this context, we assume we know the environment mecanisms of the MDP that describes the 
    environment.
    Those are modelized by:
        - transition probabilities of the MDP
        - Associated rewards
"""

N_STATES = 3
EPS = 1e-2
DISCOUNT = 0.95

R = np.array([
    # shape (n_states, n_actions)
    [0.0, 0.0, 5.0 / 100],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, .9]
])

P = np.array([
    # Indices new_state, old_state, action
    [  # new_state=0
        [0.55, 0.3, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ],

    [  # new_state=1
        [.45, .7, 0.0],
        [0.0, .4, 1.0],
        [1.0, .6, 0.0]
    ],

    [  # new_state=2
        [0.0, 0.0, 0.0],
        [0.0, .6, 0.0],
        [.0, .4, 1.0]
    ],
])


class ValueIteration:
    """
    Performs Value Iteration for a finite MDP

    Attributes:
          - rewards: (n_states, n_actions) matrix
          - probas: (n_states (new), n_states (old), n_actions) probability matrix.
          probas[y, x, a] = P(X_{n+1}=y|X_n=y, A_n=a)
          - discount: Discount used when computing gains
          - eps: stopping criteria for value iteration
    """
    def __init__(self, rewards, probas, discount, eps):
        self.rewards = rewards
        self.probas = probas
        self.discount = discount
        self.eps = eps

        self.errors = None
        self.V = None
        self.pi = None

    def iterations(self):
        V = np.zeros(self.rewards.shape[0])
        stop_criteria = np.inf
        errors = []
        while stop_criteria > self.eps:
            V_old = V.copy()
            V_3D = np.zeros(self.probas.shape)
            for idx, val in enumerate(V):
                V_3D[idx, :] = val
            bellman_op = self.rewards + self.discount*((V_3D*self.probas).sum(axis=0))
            V = bellman_op.max(axis=-1)
            stop_criteria = np.abs(V - V_old).max()
            errors.append(stop_criteria)
        self.errors = errors
        self.V = V
        return V

    def policy_estimation(self):
        V_3D = np.zeros(self.probas.shape)
        for idx, val in enumerate(self.V):
            V_3D[idx, :] = val
        bellman_op = self.rewards + self.discount*((V_3D*self.probas).sum(axis=0))
        self.pi = bellman_op.argmax(axis=-1)
        return self.pi


class PolicyIteration:
    """
    Performs Policy Iteration for a finite MDP

    Attributes:
          - rewards: (n_states, n_actions) matrix
          - probas: (n_states (new), n_states (old), n_actions) probability matrix.
          probas[y, x, a] = P(X_{n+1}=y|X_n=y, A_n=a)
          - discount: Discount used when computing gains
          - pi0: initial policy
    """
    def __init__(self, rewards, probas, discount, pi0):
        self.rewards = rewards
        self.probas = probas
        self.discount = discount

        self.errors = None
        self.V = None
        self.pi = np.array(pi0)

        self.n_states = self.rewards.shape[0]
        self.n_actions =self.rewards.shape[1]

    def iterations(self):
        done = False
        while not done:
            r_pi = self.rewards[np.arange(self.n_states), self.pi]
            probas_pi = self.probas[:, np.arange(self.n_states), self.pi].T
            self.V = np.linalg.solve(np.eye(self.n_states) - self.discount*probas_pi, r_pi)

            V_3D = np.zeros(self.probas.shape)
            for idx, val in enumerate(self.V):
                V_3D[idx, :] = val

            bellman_op = self.rewards + self.discount * ((V_3D * self.probas).sum(axis=0))
            new_pi = bellman_op.argmax(axis=-1)
            if (new_pi == self.pi).all():
                done = True

            self.pi = new_pi
