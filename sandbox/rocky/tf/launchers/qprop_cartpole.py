from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.tf.baselines.q_baseline import QfunctionBaseline

stub(globals())

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

qf = ContinuousMLPQFunction(env_spec=env.spec)

baseline = LinearFeatureBaseline(env_spec=env.spec)

qf_baseline = QfunctionBaseline(env_spec=env.spec,
    policy=policy, qf=qf)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    qf_baseline=qf_baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    qf=qf,
)
run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)
