import numpy as np
import rllab.misc.logger as logger
from rllab.misc import special2 as special

class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10, env=None):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
            )
        if env is not None and env.action_space.is_discrete:
            self._actions = np.zeros((max_pool_size,),dtype=np.int64)
            self._n = env.action_space.n
            self._is_action_discrete = True
        else:
            self._actions = np.zeros(
                (max_pool_size, action_dim),
            )
            self._is_action_discrete = False
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._initials = np.zeros(max_pool_size, dtype='uint8')
        self._observations.fill(0) # pre-allocate
        self._actions.fill(0) # pre-allocate
        self._terminals.fill(0) # pre-allocate
        self._initials.fill(0) # pre-allocate
        self._rewards.fill(0) # pre-allocate
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, initial):
        self.check_replacement()
        self._observations[self._top] = observation
        if self._is_action_discrete and not isinstance(action,
                (int, np.int64)):
            action = special.from_onehot(action)
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._initials[self._top] = initial
        self.advance()

    def check_replacement(self):
        if self._replacement_prob < 1.0:
            if self._size < self._max_pool_size or \
                not self._initials[self._top]: return
            self.advance_until_terminate()

    def get_skip_flag(self):
        if self._replacement_policy == 'full': skip = False
        elif self._replacement_policy == 'stochastic':
            skip = np.random.uniform() > self._replacement_prob
        else: raise NotImplementedError
        return skip

    def advance_until_terminate(self):
        skip = self.get_skip_flag()
        n_skips = 0
        old_top = self._top
        new_top = (old_top + 1) % self._max_pool_size
        while skip and old_top != new_top and n_skips < self._max_skip_episode:
            n_skips += 1
            self.advance()
            while not self._initials[self._top]:
                self.advance()
            skip = self.get_skip_flag()
            new_top = self._top
        logger.log("add_sample, skipped %d episodes, top=%d->%d"%(
            n_skips, old_top, new_top))

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def last_batch(self, batch_size):
        assert self._size >= batch_size
        if self._top >= batch_size:
            observations=self._observations[self._top-batch_size:self._top]
        else:
            assert self._size == self._max_pool_size
            obs1 = self._observations[self._max_pool_size+
                    self._top-batch_size:]
            obs2 = self._observations[:self._top]
            observations = np.concatenate((obs1, obs2), axis=0)
        return dict(
            observations = observations,
        )

    def random_batch(self, batch_size):
        assert self._size >= batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            # make sure that the transition is valid: discard the transition if it crosses horizon-triggered resets
            if not self._terminals[index] and self._initials[transition_index]:
                continue
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        actions = self._actions[indices]
        if self._is_action_discrete:
            actions = special.to_onehot_n(actions, self._n)
        return dict(
            observations=self._observations[indices],
            actions=actions,
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            initials=self._initials[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


