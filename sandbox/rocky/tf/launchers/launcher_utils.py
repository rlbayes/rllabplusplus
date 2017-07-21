import tensorflow as tf
import pprint
import os.path as osp
from rllab import config
from rllab.misc.ext import set_seed
import rllab.misc.logger as logger
import datetime
import dateutil.tz

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

flags = tf.app.flags
FLAGS = flags.FLAGS

# misc
flags.DEFINE_string('ec2_settings', 'experiments/python/example.py', 'Settings file for launching EC2 experiments.')
flags.DEFINE_string('exp', 'default', 'Experiment name.')
flags.DEFINE_boolean('overwrite', False, 'Overwrite logs by default.')
flags.DEFINE_boolean('force_start', False, 'Force start all.')
flags.DEFINE_integer('save_freq', 0, 'Save checkpoint frequency.')
flags.DEFINE_boolean('restore_auto', True, 'Restore params if checkpoint is available.')

# environment params
flags.DEFINE_string('env_name', 'HalfCheetah-v1', 'Environment.')
flags.DEFINE_float('discount', 0.99, 'Discount.')

# learning params
flags.DEFINE_float('learning_rate', 0.001, 'Base learning rate.')
flags.DEFINE_integer('batch_size', 5000, 'Batch size.')
flags.DEFINE_string('algo_name', 'trpo', 'RLAlgorithm.')
flags.DEFINE_integer('seed', 1, 'Seed.')
flags.DEFINE_integer('max_episode', 100000, 'Max episodes.')
flags.DEFINE_boolean('normalize_obs', False, 'Normalize observations.')
flags.DEFINE_boolean('recurrent', False, 'Recurrent policy.')
flags.DEFINE_string('policy_hidden_sizes', '100x50', 'Sizes of policy hidden layers.')
flags.DEFINE_string('qf_hidden_sizes', '100x100', 'Sizes of qf hidden layers.')
flags.DEFINE_string('policy_hidden_nonlinearity', 'tanh', 'hidden nonlinearity for policy.')
flags.DEFINE_string('policy_output_nonlinearity', None, 'output nonlinearity for policy.')
flags.DEFINE_string('qf_hidden_nonlinearity', 'relu', 'Hidden nonlinearity for qf.')
flags.DEFINE_boolean('policy_use_target', True, 'Use target policy.')
flags.DEFINE_boolean('qf_use_target', True, 'Use target qf')

# batchopt params
flags.DEFINE_float('gae_lambda', 0.97, 'Generalized advantage estimation lambda.')
flags.DEFINE_string('baseline_cls', 'linear', 'Baseline class.')
flags.DEFINE_string('baseline_hidden_sizes', '32x32', 'Baseline network hidden sizes.')

# trpo params
flags.DEFINE_float('step_size', 0.01, 'Step size for TRPO.')
flags.DEFINE_integer('sample_backups', 0, 'Backup off-policy samples for Q-prop est.')
flags.DEFINE_integer('kl_sample_backups', 0, 'Backup off-policy samples for KL est.')

# ddpg params
flags.DEFINE_float('scale_reward', 1.0, 'Scale reward for Q-learning.')
flags.DEFINE_float('policy_updates_ratio', 1.0, 'Policy updates per critic update for DDPG.')
flags.DEFINE_integer('replay_pool_size', 1000000, 'Batch size during Q-prop.')
flags.DEFINE_float('replacement_prob', 1.0, 'Replacement probability.')
flags.DEFINE_float('qf_learning_rate', 1e-3, 'Learning rate for Qfunction.')
flags.DEFINE_float('updates_ratio', 1.0, 'Updates per actor experience.')
flags.DEFINE_integer('policy_batch_size', 64, 'Batch size for policy update.')
flags.DEFINE_boolean('policy_sample_last', True, 'Sample most recent batch for policy update.')
flags.DEFINE_integer('qf_batch_size', 64, 'Qf batch size.')
flags.DEFINE_float('qf_mc_ratio', 0, 'Ratio of MC regression objective for fitting Q function.')
flags.DEFINE_float('qf_residual_phi', 0, 'Phi interpolating direct method and residual gradient method.')

# qprop params
flags.DEFINE_string('qprop_eta_option', 'ones', 'Eta multiplier for adaptive control variate.')
flags.DEFINE_float('qprop_nu', 0, 'Nu in interpolated policy gradient with control variate.')

# pgac params
flags.DEFINE_float('ac_delta', 0, 'PGAC delta.')
flags.DEFINE_integer('ac_sample_backups', 0, 'PGAC sample size.')

def shortkeys_map(key):
    return ''.join([s[0] for s in key.split('_')])

policy_keys = [
        'seed',
        'batch_size',
        #'normalize_obs',
        #'recurrent',
        'policy_hidden_sizes',
        'policy_hidden_nonlinearity',
        'policy_output_nonlinearity',
]

qf_keys = [
        'seed',
        'batch_size',
        'normalize_obs',
        'qf_hidden_sizes',
        'qf_hidden_nonlinearity',
        'qf_learning_rate',
        'scale_reward',
        #'replay_pool_size',
        'updates_ratio',
        'qf_use_target',
        'qf_batch_size',
        'qf_mc_ratio',
        'qf_residual_phi',
]

qprop_keys = [
        'qprop_eta_option',
        'sample_backups',
        'policy_sample_last',
]

pg_keys = [
        'gae_lambda',
        'baseline_cls',
        'baseline_hidden_sizes',
]

actor_keys = [
        'policy_sample_last',
        'policy_use_target',
        'policy_batch_size',
        'policy_updates_ratio',
]

pgac_keys = [
        'ac_delta',
        'ac_sample_backups',
        'policy_sample_last',
]

tr_keys = [
        'step_size',
        'kl_sample_backups',
]

keys_by_algo_map = dict(
    # TRPO
    trpo=list(set(policy_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # IPG
    actrpo=list(set(policy_keys) |
        set(qf_keys) |
        set(pgac_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # IPG with reparam critic gradient
    acqftrpo=list(set(policy_keys) |
        set(qf_keys) |
        set(pgac_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # Q-Prop
    qprop=list(set(policy_keys) |
        set(qf_keys) |
        set(qprop_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # IPG with control variate
    nuqprop=list(set(policy_keys) |
        set({'qprop_nu'}) |
        set(qf_keys) |
        set(qprop_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # IPG with control variate & reparam critic gradient
    nuqfqprop=list(set(policy_keys) |
        set({'qprop_nu'}) |
        set(qf_keys) |
        set(qprop_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # Q-Prop with reparam critic gradient
    qfqprop=list(set(policy_keys) |
        set(qf_keys) |
        set(qprop_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # Q-Prop with NAF control variate
    nafqprop=list(set(policy_keys) |
        set(qf_keys) |
        set(qprop_keys) |
        set(tr_keys)|
        set(pg_keys)),
    # vanilla policy gradient
    vpg=list(set(policy_keys) |
        {'learning_rate'}|
        set(pg_keys)),
    # Q-Prop with vanilla policy gradient
    qvpg=list(set(policy_keys) |
        set(qf_keys) |
        set(qprop_keys) |
        {'learning_rate'}|
        set(pg_keys)),
    # DDPG
    ddpg=list(set(qf_keys) |
        {'learning_rate', 'qf_learning_rate'}|
        set(actor_keys) |
        set(policy_keys)),
    # Trust-region Actor-critic
    trpg=list(set(qf_keys) |
        {'step_size', 'qf_learning_rate'}|
        set(actor_keys) |
        set(policy_keys)),
    # Trust-region Actor-critic with Off-policy exploration
    trpgoff=list(set(qf_keys) |
        {'step_size', 'qf_learning_rate'}|
        set(actor_keys) |
        set(policy_keys)),
    # SVG(0)
    dspg=list(set(qf_keys) |
        {'learning_rate', 'qf_learning_rate'}|
        set(actor_keys) |
        set(policy_keys)),
    # SVG(0) with Off-policy exploration
    dspgoff=list(set(qf_keys) |
        {'learning_rate', 'qf_learning_rate'}|
        set(actor_keys) |
        set(policy_keys)),
    # DQN/NAF
    dqn=list(set(qf_keys) |
        {'learning_rate'}),
)

blacklist_keys = [
    'env_name', 'seed', 'batch_size', 'normalize_obs',
]

def get_annotations_string(**kwargs):
    algo_name = kwargs["algo_name"]
    keys = keys_by_algo_map[algo_name]
    keys = list(set(keys) - set(blacklist_keys))
    keys = list(set(keys) | set(['algo_name']))

    annotations = {}
    for key in keys:
        annotations[shortkeys_map(key)] = kwargs[key]
    annotations_str = pprint.pformat(annotations, indent=0)
    translator = str.maketrans({
        " ":None,"'":None,"}":None,"{":None,".":"-",
        ",":"--",":":"-","\n":"",
    })
    annotations_str = annotations_str.translate(translator)
    return annotations_str

def get_env(record_video=True, record_log=True, env_name=None, normalize_obs=False, **kwargs):
    env = TfEnv(normalize(GymEnv(env_name, record_video=record_video,
        record_log=record_log), normalize_obs=normalize_obs))
    return env

def get_env_info(env=None, env_name=None, **kwargs):
    if env is None:
        env = get_env(env_name=env_name, **kwargs)
    info = {
            'name':env_name,
            'horizon':env.horizon,
            'obs_dim':env.spec.observation_space.flat_dim,
            'action_dim':env.spec.action_space.flat_dim,
            'obs_space':str(env.observation_space),
            'action_space':str(env.action_space),
            'is_obs_discrete': env.spec.observation_space.is_discrete,
            'is_action_discrete': env.spec.action_space.is_discrete,
    }
    return info, env

def run_experiment(algo, n_parallel=0, seed=0,
        plot=False, log_dir=None, exp_name=None,
        snapshot_mode='last', snapshot_gap=1,
        exp_prefix='experiment',
        log_tabular_only=False):
    default_log_dir = config.LOG_DIR + "/local/" + exp_prefix
    set_seed(seed)
    if exp_name is None:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')
        exp_name = 'experiment_%s' % (timestamp)
    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        parallel_sampler.set_seed(seed)
    if plot:
        from rllab.plotter import plotter
        plotter.init_worker()
    if log_dir is None:
        log_dir = osp.join(default_log_dir, exp_name)
    tabular_log_file = osp.join(log_dir, 'progress.csv')
    text_log_file = osp.join(log_dir, 'debug.log')
    #params_log_file = osp.join(log_dir, 'params.json')

    #logger.log_parameters_lite(params_log_file, args)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    prev_snapshot_dir = logger.get_snapshot_dir()
    prev_mode = logger.get_snapshot_mode()
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    logger.push_prefix("[%s] " % exp_name)

    algo.train()

    logger.set_snapshot_mode(prev_mode)
    logger.set_snapshot_dir(prev_snapshot_dir)
    logger.remove_tabular_output(tabular_log_file)
    logger.remove_text_output(text_log_file)
    logger.pop_prefix()


