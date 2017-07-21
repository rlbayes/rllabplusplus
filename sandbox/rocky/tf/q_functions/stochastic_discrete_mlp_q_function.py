from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.distributions.categorical import Categorical

class StochasticDiscreteMLPQFunction(DiscreteMLPQFunction, StochasticPolicy):
    def init_policy(self):
        output_vec = L.get_output(self._output_vec_layer, deterministic=True) / self._c
        prob = tf.nn.softmax(output_vec)
        max_qval = tf.reduce_logsumexp(output_vec, [1])

        self._f_prob = tensor_utils.compile_function([self._obs_layer.input_var], prob)
        self._f_max_qvals = tensor_utils.compile_function([self._obs_layer.input_var], max_qval)

        self._dist = Categorical(self._n)

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        output_vec = L.get_output(self._output_vec_layer,
                {self._l_obs_layer: tf.cast(obs_var, tf.float32)}, deterministic=True) / self._c
        prob = tf.nn.softmax(output_vec)
        return dict(prob=prob)

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist

    def get_max_qval(self, observations):
        return self._f_max_qvals(observations)
