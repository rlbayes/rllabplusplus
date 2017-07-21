from rllab.misc.instrument import stub
import tensorflow as tf

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.ddpg import DDPG
from sandbox.rocky.tf.algos.vpg import VPG

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.baselines.q_baseline import QfunctionBaseline
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.rocky.tf.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.categorical_lstm_policy import CategoricalLSTMPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy

import sandbox.rocky.tf.core.layers as L

from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.tf.q_functions.naf_mlp_q_function import NAFMLPQFunction
from sandbox.rocky.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from sandbox.rocky.tf.q_functions.deterministic_discrete_mlp_q_function import DeterministicDiscreteMLPQFunction
from sandbox.rocky.tf.q_functions.deterministic_naf_mlp_q_function import DeterministicNAFMLPQFunction
from sandbox.rocky.tf.q_functions.stochastic_discrete_mlp_q_function import StochasticDiscreteMLPQFunction

from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

def get_nonlinearity(name):
    if name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.nn.tanh
    elif name =="None" or name is None:
        return None
    else: raise NotImplementedError(name)

def get_hidden_sizes(sizes_string):
    return [int(x) for x in sizes_string.split('x')]

def get_env(env_name, record_video=True, record_log=True, normalize_obs=False, **kwargs):
    env = TfEnv(normalize(GymEnv(env_name, record_video=record_video,
        record_log=record_log), normalize_obs=normalize_obs))
    return env

def get_policy(env, algo_name, info, policy_hidden_sizes,
        policy_hidden_nonlinearity,
        policy_output_nonlinearity,
        recurrent, **kwargs):
    policy = None
    policy_class = None
    hidden_sizes = get_hidden_sizes(policy_hidden_sizes)
    hidden_nonlinearity = get_nonlinearity(policy_hidden_nonlinearity)
    output_nonlinearity = get_nonlinearity(policy_output_nonlinearity)
    if algo_name in [
            'trpo',
            'actrpo',
            'acqftrpo',
            'qprop',
            'qfqprop',
            'trpg',
            'trpgoff',
            'nuqprop',
            'nuqfqprop',
            'nafqprop',
            'vpg',
            'qvpg',
            'dspg',
            'dspgoff',
            ]:
        if not info['is_action_discrete']:
            if recurrent:
                policy = GaussianLSTMPolicy(
                    name="gauss_lstm_policy",
                    env_spec=env.spec,
                    lstm_layer_cls=L.TfBasicLSTMLayer,
                    # gru_layer_cls=L.GRULayer,
                    output_nonlinearity=output_nonlinearity, # None
                )
                policy_class = 'GaussianLSTMPolicy'
            else:
                policy = GaussianMLPPolicy(
                    name="gauss_policy",
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity, # tf.nn.tanh
                    output_nonlinearity=output_nonlinearity, # None
                )
                policy_class = 'GaussianMLPPolicy'
        else:
            if recurrent:
                policy = CategoricalLSTMPolicy(
                    name="cat_lstm_policy",
                    env_spec=env.spec,
                    lstm_layer_cls=L.TfBasicLSTMLayer,
                    # gru_layer_cls=L.GRULayer,
                )
                policy_class = 'CategoricalLSTMPolicy'
            else:
                policy = CategoricalMLPPolicy(
                    name="cat_policy",
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity, # tf.nn.tanh
                )
                policy_class = 'CategoricalMLPPolicy'
    elif algo_name in [
            'ddpg',
            ]:
        assert not info['is_action_discrete']
        policy = DeterministicMLPPolicy(
            name="det_policy",
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity, # tf.nn.relu
            output_nonlinearity=output_nonlinearity, # tf.nn.tanh
        )
        policy_class = 'DeterministicMLPPolicy'
    print('[get_policy] Instantiating %s, with sizes=%s, hidden_nonlinearity=%s.'%(
        policy_class, str(hidden_sizes), policy_hidden_nonlinearity))
    print('[get_policy] output_nonlinearity=%s.'%(
        policy_output_nonlinearity))
    return policy

def get_qf(env, info, algo_name, qf_hidden_sizes, qf_hidden_nonlinearity, **kwargs):
    qf = None
    qf_class = None
    hidden_sizes = get_hidden_sizes(qf_hidden_sizes)
    hidden_nonlinearity = get_nonlinearity(qf_hidden_nonlinearity)
    extra_kwargs = dict()
    if algo_name in [
            'ddpg',
            'trpg',
            'trpgoff',
            'qprop',
            'nuqprop',
            'nuqfqprop',
            'qfqprop',
            'actrpo',
            'acqftrpo',
            'qvpg',
            'dspg',
            'dspgoff',
            ]:
        if info['is_action_discrete']:
            qf = DiscreteMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
            )
            qf_class = 'DiscreteMLPQFunction'
        else:
            if algo_name in [
                    'trpg',
                    'trpgoff',
                    'dspg',
                    'dspgoff',
                    'acqftrpo',
                    'qfqprop',
                    'nuqfqprop',
                    ]:
                extra_kwargs['eqf_use_full_qf'] = True
            qf = ContinuousMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                **extra_kwargs,
            )
            qf_class = 'ContinuousMLPQFunction'
    elif algo_name in [
            'nafqprop',
            ]:
        assert not info['is_action_discrete']
        qf = NAFMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
        )
        qf_class = 'NAFMLPQFunction'
    elif algo_name in [
            'dqn',
            ]:
        if info['is_action_discrete']:
            qf = DeterministicDiscreteMLPQFunction(
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                )
            qf_class = 'DeterministicDiscreteMLPQFunction'
        else:
            qf = DeterministicNAFMLPQFunction(
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                )
            qf_class = 'DeterministicNAFMLPQFunction'
    elif algo_name in ['dsqn']:
        assert info['is_action_discrete']
        qf = StochasticDiscreteMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
            )
        qf_class = 'StochasticDiscreteMLPQFunction'
    print('[get_qf] Instantiating %s, with sizes=%s, hidden_nonlinearity=%s.'%(
        qf_class, str(hidden_sizes), qf_hidden_nonlinearity))
    return qf

def get_es(env, algo_name, info, **kwargs):
    es = None
    es_class = None
    if algo_name in [
            'ddpg',
            'dspgoff',
            'trpgoff',
            ]:
        es = OUStrategy(env_spec=env.spec)
        es_class = 'OUStrategy'
    elif algo_name in [
            'dqn',
            ]:
        if info['is_action_discrete']:
            es = EpsilonGreedyStrategy(env_spec=env.spec)
            es_class = 'EpsilonGreedyStrategy'
        else:
            es = OUStrategy(env_spec=env.spec)
            es_class = 'OUStrategy'
    print('[get_es] Instantiating %s.'%es_class)
    return es

def get_baseline(env, algo_name, baseline_cls, baseline_hidden_sizes, **kwargs):
    baseline = None
    baseline_class = None
    if algo_name in [
            'trpo',
            'qprop',
            'qfqprop',
            'nuqprop',
            'nuqfqprop',
            'actrpo',
            'acqftrpo',
            'vpg',
            'qvpg',
            'nafqprop',
            ]:
        if baseline_cls == 'linear':
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            baseline_class = 'LinearFeatureBaseline'
        elif baseline_cls == 'mlp':
            baseline = GaussianMLPBaseline(env_spec=env.spec,
                    regressor_args=dict(
                            hidden_sizes=get_hidden_sizes(baseline_hidden_sizes),
                        ),
                    )
            baseline_class = 'GaussianMLPBaseline'
        else: raise NotImplementedError(baseline_cls)
    print('[get_baseline] Instantiating %s.'%baseline_class)
    return baseline

def get_algo(env, policy, es, qf, baseline, max_path_length,
        batch_size, replay_pool_size, discount,
        scale_reward, learning_rate, replacement_prob,
        policy_updates_ratio,
        step_size, gae_lambda,
        sample_backups,
        kl_sample_backups,
        qprop_eta_option,
        qprop_nu,
        algo_name,
        n_itr,
        recurrent,
        updates_ratio,
        policy_use_target,
        policy_batch_size,
        policy_sample_last,
        ac_delta,
        ac_sample_backups,
        save_freq, restore_auto,
        qf_learning_rate,
        qf_use_target,
        qf_mc_ratio,
        qf_batch_size,
        qf_residual_phi,
        **kwargs):
    algo = None
    algo_class = None
    min_pool_size = 1000
    qf_baseline = None
    extra_kwargs = dict()

    print('Creating algo=%s with n_itr=%d, max_path_length=%d...'%(
        algo_name, n_itr, max_path_length))
    if algo_name in [
            'ddpg',
            'dspg',
            'dspgoff',
            'dqn',
            'dsqn',
            'trpg',
            'trpgoff',
            ]:
        if algo_name in [
                'trpg',
                ]:
            extra_kwargs['policy_update_method'] = 'cg'
        algo = DDPG(
            env=env,
            policy=policy,
            policy_use_target=policy_use_target,
            es=es,
            qf=qf,
            qf_use_target=qf_use_target,
            policy_batch_size=policy_batch_size,
            qf_batch_size=qf_batch_size,
            qf_mc_ratio=qf_mc_ratio,
            qf_residual_phi=qf_residual_phi,
            max_path_length=max_path_length,
            epoch_length=batch_size, # make comparable to batchopt methods
            min_pool_size=min_pool_size,
            replay_pool_size=replay_pool_size,
            n_epochs=n_itr,
            discount=discount,
            scale_reward=scale_reward,
            qf_learning_rate=qf_learning_rate,
            policy_learning_rate=learning_rate,
            policy_step_size=step_size,
            policy_sample_last=policy_sample_last,
            replacement_prob=replacement_prob,
            policy_updates_ratio=policy_updates_ratio,
            updates_ratio=updates_ratio,
            save_freq=save_freq,
            restore_auto=restore_auto,
            **extra_kwargs,
        )
        algo_class = 'DDPG'
    elif algo_name in [
            'trpo',
            'nuqprop',
            'nuqfqprop',
            'actrpo',
            'acqftrpo',
            'qprop',
            'qfqprop',
            'nafqprop',
            ]:
        if recurrent:
            extra_kwargs['optimizer'] = \
                ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        if algo_name in [
                'actrpo',
                'acqftrpo',
                ]:
            extra_kwargs['ac_delta'] = ac_delta
            extra_kwargs['qprop'] = False # disable qprop
            if ac_delta == 0: qf = None
        if algo_name in [
                'nuqprop',
                'nuqfqprop',
                ]:
            extra_kwargs['qprop_nu'] = qprop_nu
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
            sample_backups=sample_backups,
            kl_sample_backups=kl_sample_backups,
            qf=qf,
            qf_use_target=qf_use_target,
            qf_batch_size=qf_batch_size,
            qf_mc_ratio=qf_mc_ratio,
            qf_residual_phi=qf_residual_phi,
            min_pool_size=min_pool_size,
            scale_reward=scale_reward,
            qf_updates_ratio=updates_ratio,
            qprop_eta_option=qprop_eta_option,
            replay_pool_size=replay_pool_size,
            replacement_prob=replacement_prob,
            qf_baseline=qf_baseline,
            qf_learning_rate=qf_learning_rate,
            ac_sample_backups=ac_sample_backups,
            policy_sample_last=policy_sample_last,
            save_freq=save_freq,
            restore_auto=restore_auto,
            **extra_kwargs
        )
        algo_class = 'TRPO'
    elif algo_name in [
            'vpg',
            'qvpg',
            ]:
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
            qf_use_target=qf_use_target,
            qf_batch_size=qf_batch_size,
            qf_mc_ratio=qf_mc_ratio,
            qf_residual_phi=qf_residual_phi,
            min_pool_size=min_pool_size,
            scale_reward=scale_reward,
            qf_updates_ratio=updates_ratio,
            qprop_eta_option=qprop_eta_option,
            replay_pool_size=replay_pool_size,
            qf_baseline=qf_baseline,
            qf_learning_rate=qf_learning_rate,
            save_freq=save_freq,
            restore_auto=restore_auto,
        )
        algo_class = 'VPG'
    print('[get_algo] Instantiating %s.'%algo_class)
    return algo

stub(globals())
