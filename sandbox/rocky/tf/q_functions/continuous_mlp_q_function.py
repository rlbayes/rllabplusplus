from sandbox.rocky.tf.q_functions.base import QFunction
import numpy as np
from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.layers import batch_norm
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils

import tensorflow as tf
import sandbox.rocky.tf.core.layers as L

class ContinuousMLPQFunction(QFunction, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name='qnet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            action_merge_layer=-2,
            output_nonlinearity=None,
            eqf_use_full_qf=False,
            eqf_sample_size=1,
            bn=False):
        Serializable.quick_init(self, locals())

        assert not env_spec.action_space.is_discrete
        self._env_spec = env_spec

        with tf.variable_scope(name):
            l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim), name="obs")
            l_action = L.InputLayer(shape=(None, env_spec.action_space.flat_dim), name="actions")

            n_layers = len(hidden_sizes) + 1

            if n_layers > 1:
                action_merge_layer = \
                    (action_merge_layer % n_layers + n_layers) % n_layers
            else:
                action_merge_layer = 1

            l_hidden = l_obs

            for idx, size in enumerate(hidden_sizes):
                if bn:
                    l_hidden = batch_norm(l_hidden)

                if idx == action_merge_layer:
                    l_hidden = L.ConcatLayer([l_hidden, l_action])

                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1)
                )

            if action_merge_layer == n_layers:
                l_hidden = L.ConcatLayer([l_hidden, l_action])

            l_output = L.DenseLayer(
                l_hidden,
                num_units=1,
                nonlinearity=output_nonlinearity,
                name="output"
            )

            output_var = L.get_output(l_output, deterministic=True)
            output_var = tf.reshape(output_var, (-1,))

            self._f_qval = tensor_utils.compile_function([l_obs.input_var, l_action.input_var], output_var)
            self._output_layer = l_output
            self._obs_layer = l_obs
            self._action_layer = l_action
            self._output_nonlinearity = output_nonlinearity

            self.eqf_use_full_qf=eqf_use_full_qf
            self.eqf_sample_size=eqf_sample_size

            LayersPowered.__init__(self, [l_output])

    def get_qval(self, observations, actions):
        return self._f_qval(observations, actions)

    def get_qval_sym(self, obs_var, action_var, **kwargs):
        qvals = L.get_output(
            self._output_layer,
            {self._obs_layer: obs_var, self._action_layer: action_var},
            **kwargs
        )
        return tf.reshape(qvals, (-1,))

    def get_e_qval(self, observations, policy):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info(observations)
            means, log_stds = agent_info['mean'], agent_info['log_std']
            if self.eqf_use_full_qf and self.eqf_sample_size > 1:
                observations = np.repeat(observations, self.eqf_sample_size, axis=0)
                means = np.repeat(means, self.eqf_sample_size, axis=0)
                stds = np.repeat(np.exp(log_stds), self.eqf_sample_size, axis=0)
                randoms = np.random.randn(*(means))
                actions = means + stds * randoms
                all_qvals = self.get_qval(observations, actions)
                qvals = np.mean(all_qvals.reshape((-1,self.eqf_sample_size)),axis=1)
            else:
                qvals = self.get_qval(observations, means)
        else:
            actions, _ = policy.get_actions(observations)
            qvals = self.get_qval(observations, actions)
        return qvals

    def _get_e_qval_sym(self, obs_var, policy, **kwargs):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info_sym(obs_var)
            mean_var, log_std_var = agent_info['mean'], agent_info['log_std']
            if self.eqf_use_full_qf:
                assert self.eqf_sample_size > 0
                if self.eqf_sample_size == 1:
                    action_var = tf.random_normal(shape=tf.shape(mean_var))*tf.exp(log_std_var) + mean_var
                    return self.get_qval_sym(obs_var, action_var, **kwargs), action_var
                else: raise NotImplementedError
            else:
                return self.get_qval_sym(obs_var, mean_var, **kwargs), mean_var
        else:
            action_var = policy.get_action_sym(obs_var)
            return self.get_qval_sym(obs_var, action_var, **kwargs), action_var

    def get_e_qval_sym(self, obs_var, policy, **kwargs):
        return self._get_e_qval_sym(obs_var, policy, **kwargs)[0]

    def get_cv_sym(self, obs_var, action_var, policy, **kwargs):
        if self.eqf_use_full_qf:
            qvals = self.get_qval_sym(obs_var, action_var, deterministic=True, **kwargs)
            e_qvals = self.get_e_qval_sym(obs_var, policy, deterministic=True, **kwargs)
            return qvals - e_qvals
        else:
            qvals, action0 = self._get_e_qval_sym(obs_var, policy, deterministic=True, **kwargs)
            # use first-order Taylor expansion
            qprimes = tf.gradients(qvals, action0)[0]
            deltas = action_var - action0
            return tf.reduce_sum(deltas * qprimes, 1)
