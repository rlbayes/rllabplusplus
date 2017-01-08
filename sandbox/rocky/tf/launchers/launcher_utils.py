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

# common params
flags.DEFINE_float('learning_rate', 0.001, 'Base learning rate.')
flags.DEFINE_integer('batch_size', 5000, 'Batch size.')
flags.DEFINE_float('discount', 0.99, 'Discount.')
flags.DEFINE_integer('hid_size', 100, 'Size of hidden layers.')
flags.DEFINE_string('env_name', 'HalfCheetah-v1', 'Environment.')
flags.DEFINE_string('exp', 'default', 'Experiment name.')
flags.DEFINE_string('algo_name', 'trpo', 'RLAlgorithm.')
flags.DEFINE_boolean('overwrite', False, 'Overwrite logs by default.')
flags.DEFINE_boolean('force_start', False, 'Force start all.')
flags.DEFINE_integer('seed', 1, 'Seed.')
flags.DEFINE_integer('max_episode', 50000, 'Max episodes.')
flags.DEFINE_boolean('normalize_obs', False, 'Normalize observations.')

# batchopt params
flags.DEFINE_float('gae_lambda', 0.97, 'Generalized advantage estimation lambda.')

# trpo params
flags.DEFINE_float('step_size', 0.01, 'Step size for TRPO.')
flags.DEFINE_integer('sample_backups', 0, 'Backup off-policy samples for KL est.')

# ddpg params
flags.DEFINE_float('scale_reward', 1.0, 'Scale reward for Q-learning.')
flags.DEFINE_float('policy_updates_ratio', 1.0, 'Policy updates per critic update for DDPG.')
flags.DEFINE_integer('replay_pool_size', 1000000, 'Batch size during Q-prop.')
flags.DEFINE_float('replacement_prob', 1.0, 'Replacement probability.')
flags.DEFINE_integer('qf_hid_size', 100, 'Hidden layer size for Qfunction.')
flags.DEFINE_float('qf_learning_rate', 1e-3, 'Learning rate for Qfunction.')
flags.DEFINE_string('qf_hidden_nonlinearity', 'relu', 'Hidden nonlinearity for Qfunction.')

# qprop params
flags.DEFINE_integer('qprop_min_itr', 0, 'Min itr before Q-prop begins.')
flags.DEFINE_float('qf_updates_ratio', 1.0, 'Critic updates per actor experience.')
flags.DEFINE_boolean('qprop_use_qf_baseline', False, 'Use Qfunction as baseline.')
flags.DEFINE_string('qprop_eta_option', 'ones', 'Eta multiplier for control variate.')

shortkeys_map = {
        'algo_name': 'al',
        'learning_rate': 'lr',
        'step_size': 'ss',
        'hid_size': 'hs',
        'gae_lambda': 'gl',
        'scale_reward': 'sr',
        'qprop_min_itr': 'mi',
        'qf_updates_ratio': 'qur',
        'replay_pool_size': 'rps',
        'qprop_use_qf_baseline': 'quqb',
        'policy_updates_ratio': 'pur',
        'qprop_eta_option': 'qeo',
        'qf_hid_size': 'qhs',
        'qf_learning_rate': 'qlr',
        'qf_hidden_nonlinearity': 'qhn',
        }

keys_by_algo_map = dict(
    trpo=[
        'seed',
        'batch_size',
        'normalize_obs',
        'step_size',
        'gae_lambda',
    ],
    qprop=[
        'seed',
        'batch_size',
        'normalize_obs',
        'step_size',
        'qf_updates_ratio',
        'qprop_eta_option',
        'scale_reward',
        'qf_hid_size',
        'qf_learning_rate',
        'qf_hidden_nonlinearity',
        'replay_pool_size',
        'gae_lambda',
    ],
    qvpg=[
        'seed',
        'batch_size',
        'normalize_obs',
        'learning_rate',
        'qf_updates_ratio',
        'qprop_eta_option',
        'scale_reward',
        'qf_hid_size',
        'qf_learning_rate',
        'qf_hidden_nonlinearity',
        'replay_pool_size',
        'gae_lambda',
    ],
    vpg=[
        'seed',
        'batch_size',
        'normalize_obs',
        'learning_rate',
        'gae_lambda',
    ],
    ddpg=[
        'seed',
        'batch_size',
        'learning_rate',
        'normalize_obs',
        'scale_reward',
        'qf_hid_size',
        'qf_learning_rate',
        'qf_hidden_nonlinearity',
        'replay_pool_size',
    ],
)

blacklist_keys = [
    'seed', 'batch_size', 'normalize_obs',
]

def get_annotations_string(keys=None, **kwargs):
    algo_name = kwargs["algo_name"]
    if keys is None: keys = keys_by_algo_map[algo_name]
    keys = list(set(keys) - set(blacklist_keys))
    keys = list(set(keys) | set(['algo_name']))

    annotations = {}
    for key in keys:
        annotations[shortkeys_map[key]] = kwargs[key]
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


