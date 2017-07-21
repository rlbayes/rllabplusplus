from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.q_functions.naf_mlp_q_function import NAFMLPQFunction
from sandbox.rocky.tf.misc import tensor_utils

class DeterministicNAFMLPQFunction(NAFMLPQFunction, Policy):
    def init_policy(self):
        L_var, V_var, mu_var = self.get_output_sym(self._obs_layer.input_var, deterministic=True)
        self._f_actions = tensor_utils.compile_function([self._obs_layer.input_var], mu_var)
        self._f_max_qvals = tensor_utils.compile_function([self._obs_layer.input_var], V_var)

    def get_action(self, observation):
        return self._f_actions([observation])[0], dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        _, _, mu_var = self.get_output_sym(obs_var)
        return mu_var

    def get_max_qval(self, observations):
        return self._f_max_qvals(observations)
