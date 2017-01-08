from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger

class QfunctionBaseline(Baseline, Serializable):

    def __init__(self, env_spec, qf, policy):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.qf = qf
        self.policy = policy

    def get_qbaseline_sim(self, obs_var, scale_reward=1.0):
        info_vars = self.policy.dist_info_sym(obs_var)
        action_mu = info_vars["mean"]
        qvalue = self.qf.get_qval_sym(
            obs_var, action_mu,
            deterministic=True,
        )
        qvalue /= scale_reward
        info_vars["qvalue"] = qvalue
        info_vars["qprime"] = tf.gradients(qvalue, action_mu)[0]
        info_vars["action_mu"] = action_mu

        f_baseline = tensor_utils.compile_function(
            inputs = [obs_var],
            outputs = qvalue,
        )

        self.opt_info = {
            "f_baseline": f_baseline,
        }

        return info_vars

    @overrides
    def fit(self, paths):
        logger.log("Using qf_baseline.")

    @overrides
    def predict(self, path):
        f_baseline = self.opt_info["f_baseline"]
        return f_baseline(path["observations"])

