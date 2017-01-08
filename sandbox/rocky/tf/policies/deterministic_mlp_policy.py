from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class DeterministicMLPPolicy(Policy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
            mean_network=None,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        with tf.variable_scope(name):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            # create network
            if mean_network is None:
                mean_network = MLP(
                    name="mean_network",
                    input_shape=(obs_dim,),
                    output_dim=action_dim,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=output_nonlinearity,
                )
            self._mean_network = mean_network

            l_mean = mean_network.output_layer
            obs_var = mean_network.input_layer.input_var

            self._l_mean = l_mean
            action_var = L.get_output(self._l_mean, deterministic=True)

            LayersPowered.__init__(self, [l_mean])
            super(DeterministicMLPPolicy, self).__init__(env_spec)

            self._f_actions = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=action_var,
            )

    def get_action(self, observation):
        action = self._f_actions([observation])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        return L.get_output(self._l_mean, obs_var)
