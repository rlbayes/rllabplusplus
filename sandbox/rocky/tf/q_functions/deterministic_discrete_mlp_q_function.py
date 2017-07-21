from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils

class DeterministicDiscreteMLPQFunction(DiscreteMLPQFunction, Policy):
    def init_policy(self):
        output_vec = L.get_output(self._output_vec_layer, deterministic=True)
        action = tf.to_int64(tf.argmax(output_vec, 1))
        action_vec = tf.one_hot(action, self._n)
        max_qval = tf.reduce_max(output_vec, 1)

        self._f_actions = tensor_utils.compile_function([self._obs_layer.input_var], action)
        self._f_actions_vec = tensor_utils.compile_function([self._obs_layer.input_var], action_vec)
        self._f_max_qvals = tensor_utils.compile_function([self._obs_layer.input_var], max_qval)

    def get_action(self, observation):
        return self._f_actions([observation])[0], dict()

    def get_actions(self, observations):
        return self._f_actions_vec(observations), dict()

    def get_action_sym(self, obs_var):
        output_vec = L.get_output(self._output_vec_layer, obs_var, deterministic=True)
        action = tf.to_int64(tf.argmax(output_vec, 1))
        action_vec = tf.one_hot(action, self._n)
        return action_vec

    def get_max_qval(self, observations):
        return self._f_max_qvals(observations)
