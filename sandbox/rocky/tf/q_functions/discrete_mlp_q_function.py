from sandbox.rocky.tf.q_functions.base import QFunction
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy

class DiscreteMLPQFunction(QFunction, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name='qnet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            action_merge_layer=-2,
            output_nonlinearity=None,
            hidden_W_init=L.XavierUniformInitializer(),
            hidden_b_init=L.ZerosInitializer(),
            output_W_init=L.XavierUniformInitializer(),
            output_b_init=L.ZerosInitializer(),
            c=1.0, # temperature variable for stochastic policy
            bn=False):
        Serializable.quick_init(self, locals())

        assert env_spec.action_space.is_discrete
        self._n = env_spec.action_space.n
        self._c = c
        self._env_spec = env_spec

        with tf.variable_scope(name):
            l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim), name="obs")
            l_action = L.InputLayer(shape=(None, env_spec.action_space.n), var_type=tf.uint8, name="actions")

            n_layers = len(hidden_sizes) + 1

            l_hidden = l_obs

            for idx, size in enumerate(hidden_sizes):
                if bn:
                    l_hidden = L.batch_norm(l_hidden)

                l_hidden = L.DenseLayer(
                    l_hidden,
                    num_units=size,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    nonlinearity=hidden_nonlinearity,
                    name="h%d" % (idx + 1)
                )

            l_output_vec = L.DenseLayer(
                l_hidden,
                num_units=env_spec.action_space.n,
                W=output_W_init,
                b=output_b_init,
                nonlinearity=output_nonlinearity,
                name="output"
            )

            output_vec_var = L.get_output(l_output_vec, deterministic=True)

            output_var = tf.reduce_sum(output_vec_var*tf.to_float(l_action.input_var), 1)

            self._f_qval = tensor_utils.compile_function([l_obs.input_var, l_action.input_var], output_var)
            self._f_qval_vec = tensor_utils.compile_function([l_obs.input_var], output_vec_var)
            self._output_vec_layer = l_output_vec
            self._obs_layer = l_obs
            self._action_layer = l_action
            self._output_nonlinearity = output_nonlinearity

            self.init_policy()

            LayersPowered.__init__(self, [l_output_vec])

    def init_policy(self):
        pass

    def get_qval(self, observations, actions):
        return self._f_qval(observations, actions)

    def _get_qval_sym(self, obs_var, action_var, **kwargs):
        output_vec = L.get_output(
            self._output_vec_layer,
            {self._obs_layer: obs_var},
            **kwargs
        )
        return tf.reduce_sum(output_vec * tf.to_float(action_var), 1), output_vec

    def get_qval_sym(self, obs_var, action_var, **kwargs):
        return self._get_qval_sym(obs_var, action_var, **kwargs)[0]

    def get_e_qval(self, observations, policy):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info(observations)
            action_vec = agent_info['prob']
        else: raise NotImplementedError
        output_vec = self._f_qval_vec(observations)
        qvals = output_vec * action_vec
        return qvals.sum(axis=1)

    def get_e_qval_sym(self, obs_var, policy, **kwargs):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info_sym(obs_var)
            action_vec = agent_info['prob']
        else: raise NotImplementedError
        output_vec = L.get_output(
            self._output_vec_layer,
            {self._obs_layer: obs_var},
            **kwargs
        )
        return tf.reduce_sum(output_vec * action_vec, 1)

    def get_cv_sym(self, obs_var, action_var, policy, **kwargs):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info_sym(obs_var)
            action_vec = agent_info['prob']
        else: raise NotImplementedError
        qvals, output_vec = self._get_qval_sym(obs_var, action_var, deterministic=True, **kwargs)
        vvals = tf.reduce_sum(output_vec * action_vec, 1)
        return qvals - vvals
