from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np

class EpsilonGreedyStrategy(ExplorationStrategy, Serializable):

    def __init__(self, env_spec, epsilon=0.01):
        assert env_spec.action_space.is_discrete
        Serializable.quick_init(self, locals())
        self._epsilon = epsilon
        self._action_space = env_spec.action_space
        self._n = env_spec.action_space.n

    def get_action(self, t, observation, policy, **kwargs):
        if np.random.rand() > self._epsilon:
            action, _ = policy.get_action(observation)
        else:
            action = np.random.randint(self._n)
        return action
