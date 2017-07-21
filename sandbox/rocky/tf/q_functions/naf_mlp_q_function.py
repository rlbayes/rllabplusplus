from sandbox.rocky.tf.q_functions.base import QFunction
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy

class NAFMLPQFunction(QFunction, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name='nafqnet',
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            action_merge_layer=0,
            output_nonlinearity=None,
            hidden_W_init=L.XavierUniformInitializer(),
            hidden_b_init=L.ZerosInitializer(),
            output_W_init=L.XavierUniformInitializer(),
            output_b_init=L.ZerosInitializer(),
            bn=False):
        Serializable.quick_init(self, locals())

        assert not env_spec.action_space.is_discrete

        action_dim = env_spec.action_space.flat_dim
        self._action_dim = action_dim
        self._env_spec = env_spec

        n_layers = len(hidden_sizes)
        action_merge_layer = \
            (action_merge_layer % n_layers + n_layers) % n_layers

        with tf.variable_scope(name):
            l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim), name="obs")
            l_action = L.InputLayer(shape=(None, env_spec.action_space.flat_dim), name="actions")
            l_policy_mu = L.InputLayer(shape=(None, action_dim), name="policy_mu")
            l_policy_sigma = L.InputLayer(shape=(None, action_dim, action_dim), name="policy_sigma")

            l_hidden = l_obs
            idx = 0

            l_hidden_kwargs = dict(
                W=hidden_W_init,
                b=hidden_b_init,
                nonlinearity=hidden_nonlinearity,
            )

            l_output_kwargs = dict(
                W=output_W_init,
                b=output_b_init,
            )

            while idx < action_merge_layer:
                if bn: l_hidden = L.batch_norm(l_hidden)
                l_hidden = L.DenseLayer(
                    l_hidden,num_units=hidden_sizes[idx],
                    name="h%d" % (idx + 1), **l_hidden_kwargs,)
                idx += 1

            _idx = idx
            _l_hidden = l_hidden

            # compute L network
            while idx < n_layers:
                if bn: l_hidden = L.batch_norm(l_hidden)
                l_hidden = L.DenseLayer(
                    l_hidden,num_units=hidden_sizes[idx],
                    name="L_h%d" % (idx + 1), **l_hidden_kwargs,)
                idx += 1
            l_L = L.DenseLayer(
                l_hidden,num_units=action_dim**2, nonlinearity=None,
                name="L_h%d" % (idx + 1), **l_output_kwargs,)

            # compute V network
            idx = _idx
            l_hidden = _l_hidden
            while idx < n_layers:
                if bn: l_hidden = L.batch_norm(l_hidden)
                l_hidden = L.DenseLayer(
                    l_hidden,num_units=hidden_sizes[idx],
                    name="V_h%d" % (idx + 1), **l_hidden_kwargs,)
                idx += 1
            l_V = L.DenseLayer(
                l_hidden,num_units=1, nonlinearity=None,
                name="V_h%d" % (idx + 1), **l_output_kwargs,)

             # compute mu network
            idx = _idx
            l_hidden = _l_hidden
            while idx < n_layers:
                if bn: l_hidden = L.batch_norm(l_hidden)
                l_hidden = L.DenseLayer(
                    l_hidden,num_units=hidden_sizes[idx],
                    name="mu_h%d" % (idx + 1), **l_hidden_kwargs,)
                idx += 1
            if bn: l_hidden = L.batch_norm(l_hidden)
            l_mu = L.DenseLayer(
                l_hidden,num_units=action_dim, nonlinearity=tf.nn.tanh,
                name="mu_h%d" % (idx + 1), **l_output_kwargs,)

            L_var, V_var, mu_var = L.get_output([l_L, l_V, l_mu], deterministic=True)
            V_var = tf.reshape(V_var, (-1,))

            # compute advantage
            L_mat_var = self.get_L_sym(L_var)
            P_var = self.get_P_sym(L_mat_var)
            A_var = self.get_A_sym(P_var, mu_var, l_action.input_var)

            # compute Q
            Q_var = A_var + V_var

            # compute expected Q under Gaussian policy
            e_A_var = self.get_e_A_sym(P_var, mu_var, l_policy_mu.input_var, l_policy_sigma.input_var)
            e_Q_var = e_A_var + V_var

            self._f_qval = tensor_utils.compile_function([l_obs.input_var, l_action.input_var], Q_var)
            self._f_e_qval = tensor_utils.compile_function([l_obs.input_var, l_policy_mu.input_var,
                l_policy_sigma.input_var], e_Q_var)
            self._L_layer = l_L
            self._V_layer = l_V
            self._mu_layer = l_mu
            self._obs_layer = l_obs
            self._action_layer = l_action
            self._policy_mu_layer = l_policy_mu
            self._policy_sigma_layer = l_policy_sigma
            self._output_nonlinearity = output_nonlinearity

            self.init_policy()

            LayersPowered.__init__(self, [l_L, l_V, l_mu])

    def init_policy(self):
        pass

    def get_L_sym(self, L_vec_var):
        L = tf.reshape(L_vec_var, (-1, self._action_dim, self._action_dim))
        return tf.matrix_band_part(L, -1, 0) - \
                tf.matrix_diag(tf.matrix_diag_part(L)) + \
                tf.matrix_diag(tf.exp(tf.matrix_diag_part(L)))

    def get_P_sym(self, L_mat_var):
        return tf.matmul(L_mat_var, tf.matrix_transpose(L_mat_var))

    def get_e_A_sym(self, P_var, mu_var, policy_mu_var, policy_sigma_var):
        e_A_var1 = self.get_A_sym(P_var, mu_var, policy_mu_var)
        e_A_var2 = - 0.5 * tf.reduce_sum(tf.matrix_diag_part(
            tf.matmul(P_var, policy_sigma_var)), 1)
        #e_A_var2 = - 0.5 * tf.trace(tf.matmul(P_var, policy_sigma_var))
        return e_A_var1 + e_A_var2

    def get_A_sym(self, P_var, mu_var, action_var):
        delta_var = action_var - mu_var
        delta_mat_var = tf.reshape(delta_var, (-1, self._action_dim, 1))
        P_delta_var = tf.squeeze(tf.matmul(P_var, delta_mat_var),[2])
        return -0.5 * tf.reduce_sum(delta_var * P_delta_var, 1)

    def get_qval(self, observations, actions):
        qvals = self._f_qval(observations, actions)
        return qvals

    def get_output_sym(self, obs_var, **kwargs):
        L_var, V_var, mu_var = L.get_output(
            [self._L_layer, self._V_layer, self._mu_layer],
            {self._obs_layer: obs_var},
            **kwargs
        )
        V_var = tf.reshape(V_var, (-1,))
        return L_var, V_var, mu_var

    def _get_qval_sym(self, obs_var, action_var, **kwargs):
        L_var, V_var, mu_var = self.get_output_sym(obs_var, **kwargs)
        L_mat_var = self.get_L_sym(L_var)
        P_var = self.get_P_sym(L_mat_var)
        A_var = self.get_A_sym(P_var, mu_var, action_var)
        Q_var = A_var + V_var
        return Q_var, A_var, V_var

    def get_qval_sym(self, obs_var, action_var, **kwargs):
        return self._get_qval_sym(obs_var, action_var, **kwargs)[0]

    def get_e_qval(self, observations, policy):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info(observations)
            mu, log_std = agent_info['mean'], agent_info["log_std"]
            std = np.array([np.diag(x) for x in np.exp(log_std)], dtype=log_std.dtype)
            qvals = self._f_e_qval(observations, mu, std)
        else:
            actions, _ = policy.get_actions(observations)
            qvals = self.get_qval(observations, actions)
        return qvals

    def get_e_qval_sym(self, obs_var, policy, **kwargs):
        if isinstance(policy, StochasticPolicy):
            agent_info = policy.dist_info_sym(obs_var)
            mu, log_std = agent_info['mean'], agent_info["log_std"]
            std = tf.matrix_diag(tf.exp(log_std))
            L_var, V_var, mu_var = self.get_output_sym(obs_var, **kwargs)
            L_mat_var = self.get_L_sym(L_var)
            P_var = self.get_P_sym(L_mat_var)
            A_var = self.get_e_A_sym(P_var, mu_var, mu, std)
            qvals = A_var + V_var
        else:
            mu = policy.get_action_sym(obs_var)
            qvals = self.get_qval_sym(obs_var, mu, **kwargs)
        return qvals

    def get_cv_sym(self, obs_var, action_var, policy, **kwargs):
        #_, avals, _  = self._get_qval_sym(obs_var, action_var, **kwargs)
        qvals = self.get_qval_sym(obs_var, action_var, **kwargs)
        e_qvals = self.get_e_qval_sym(obs_var, policy, **kwargs)
        avals = qvals - e_qvals
        return avals
