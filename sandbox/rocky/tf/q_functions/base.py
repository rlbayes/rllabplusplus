from sandbox.rocky.tf.core.parameterized import Parameterized

class QFunction(Parameterized):

    def __init__(self, env_spec):
        Parameterized.__init__(self)
        self._env_spec = env_spec

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def env_spec(self):
        return self._env_spec
