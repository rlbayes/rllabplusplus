from rllab.misc.instrument import stub
import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.ddpg import DDPG
from sandbox.rocky.tf.algos.vpg import VPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.baselines.q_baseline import QfunctionBaseline

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.tf.exploration_strategies.ou_strategy import OUStrategy

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy

from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

def get_env(env_name, record_video=True, record_log=True, normalize_obs=False, **kwargs):
    env = TfEnv(normalize(GymEnv(env_name, record_video=record_video,
        record_log=record_log), normalize_obs=normalize_obs))
    return env

def get_policy(env, algo_name, **kwargs):
    policy = None
    if algo_name in ['trpo', 'qprop', 'vpg', 'qvpg']:
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            # hidden_sizes=(hid_size, hid_size)
            hidden_sizes=(100, 50, 25),
            hidden_nonlinearity=tf.nn.tanh,
        )
    elif algo_name in ['ddpg']:
        policy = DeterministicMLPPolicy(
            name="policy",
            env_spec=env.spec,
            #hidden_sizes=(hid_size, hid_size),
            hidden_sizes=(100, 50, 25),
            #hidden_sizes=(100,100),
            hidden_nonlinearity=tf.nn.relu,
        )
    return policy

def get_qf(env, algo_name, qf_hid_size, qf_hidden_nonlinearity, **kwargs):
    qf = None
    if algo_name in ['ddpg', 'qprop', 'qvpg']:
        if qf_hidden_nonlinearity == 'relu':
            hidden_nonlinearity = tf.nn.relu
        elif qf_hidden_nonlinearity == 'tanh':
            hidden_nonlinearity = tf.nn.tanh
        else: raise NotImplementedError(qf_hidden_nonlinearity)
        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            #hidden_sizes=(100,100),
            hidden_sizes=(qf_hid_size, qf_hid_size),
            hidden_nonlinearity=hidden_nonlinearity,
        )
    return qf

def get_es(env, algo_name, **kwargs):
    es = None
    if algo_name in ['ddpg']:
        es = OUStrategy(env_spec=env.spec)
    return es

def get_baseline(env, algo_name, **kwargs):
    baseline = None
    if algo_name in ['trpo', 'qprop', 'vpg', 'qvpg']:
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    return baseline

def get_algo(env, policy, es, qf, baseline, max_path_length,
        batch_size, replay_pool_size, discount,
        scale_reward, learning_rate, replacement_prob,
        policy_updates_ratio,
        step_size, gae_lambda,
        sample_backups,
        qprop_min_itr,
        qf_updates_ratio,
        qprop_use_qf_baseline,
        qprop_eta_option,
        algo_name,
        qf_learning_rate,
        n_itr,
        **kwargs):
    algo = None
    min_pool_size = 1000
    qf_batch_size = 64
    qf_baseline = None

    print('Creating algo=%s with n_itr=%d, max_path_length=%d...'%(
        algo_name, n_itr, max_path_length))

    if algo_name in ['ddpg']:
        algo = DDPG(
            env=env,
            policy=policy,
            es=es,
            qf=qf,
            batch_size=qf_batch_size,
            max_path_length=max_path_length,
            epoch_length=batch_size, # make comparable to batchopt methods
            min_pool_size=min_pool_size,
            replay_pool_size=replay_pool_size,
            n_epochs=n_itr,
            discount=discount,
            scale_reward=scale_reward,
            qf_learning_rate=qf_learning_rate,
            policy_learning_rate=learning_rate,
            replacement_prob=replacement_prob,
            policy_updates_ratio=policy_updates_ratio,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
    elif algo_name in ['trpo', 'qprop']:
        if qf is not None:
            qf_baseline = QfunctionBaseline(env_spec=env.spec,
                policy=policy, qf=qf)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=discount,
            step_size=step_size,
            gae_lambda=gae_lambda,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
            sample_backups=sample_backups,
            qf=qf,
            qf_batch_size=qf_batch_size,
            min_pool_size=min_pool_size,
            scale_reward=scale_reward,
            qprop_min_itr=qprop_min_itr,
            qf_updates_ratio=qf_updates_ratio,
            qprop_eta_option=qprop_eta_option,
            replay_pool_size=replay_pool_size,
            replacement_prob=replacement_prob,
            qf_baseline=qf_baseline,
            qf_learning_rate=qf_learning_rate,
            qprop_use_qf_baseline=qprop_use_qf_baseline,
        )
    elif algo_name in ['vpg', 'qvpg']:
        if qf is not None:
            qf_baseline = QfunctionBaseline(env_spec=env.spec,
                policy=policy, qf=qf)
        algo = VPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=discount,
            gae_lambda=gae_lambda,
            optimizer_args=dict(
                tf_optimizer_args=dict(
                    learning_rate=learning_rate,
                )
            ),
            qf=qf,
            qf_batch_size=qf_batch_size,
            min_pool_size=min_pool_size,
            scale_reward=scale_reward,
            qprop_min_itr=qprop_min_itr,
            qf_updates_ratio=qf_updates_ratio,
            qprop_eta_option=qprop_eta_option,
            replay_pool_size=replay_pool_size,
            qf_baseline=qf_baseline,
            qf_learning_rate=qf_learning_rate,
            qprop_use_qf_baseline=qprop_use_qf_baseline,
        )
    return algo

stub(globals())
