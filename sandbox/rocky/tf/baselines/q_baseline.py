from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger

class QfunctionBaseline(Baseline, Serializable):

    def __init__(self, env_spec, qf, policy):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.qf = qf
        self.policy = policy

    @overrides
    def fit(self, paths):
        logger.log("Using qf_baseline.")

    @overrides
    def predict(self, path):
        return self.qf.get_e_qval(path["observations"])

