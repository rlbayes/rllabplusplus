from sandbox.rocky.tf.launchers.launcher_utils import FLAGS, get_env_info, get_annotations_string
from rllab.misc.instrument import run_experiment_lite
from rllab import config
import os.path as osp
import sys
from sandbox.rocky.tf.launchers.launcher_stub_utils import get_env, get_policy, get_baseline, get_qf, get_es, get_algo
import tensorflow as tf
from copy import deepcopy
import numpy as np


def run_experiment(mode="local", keys=None, params=dict()):
    flags = FLAGS.__flags
    flags = deepcopy(flags)

    for k, v in params.items():
        print('Modifying flags.%s from %r to %r'%(
            k, flags[k], v))
        flags[k] = v

    n_episodes = flags["max_episode"] # max episodes before termination
    env = get_env(record_video=False, record_log=False, **flags)
    policy = get_policy(env=env, **flags)
    baseline = get_baseline(env=env, **flags)
    qf = get_qf(env=env, **flags)
    es = get_es(env=env, **flags)

    info, _ = get_env_info(**flags)
    max_path_length = info['horizon']
    n_itr = int(np.ceil(float(n_episodes*max_path_length)/flags['batch_size']))

    algo = get_algo(n_itr=n_itr, env=env, policy=policy, baseline=baseline,
            qf=qf, es=es, max_path_length=max_path_length, **flags)

    #exp_prefix='%s-%s-%d'%(flags["exp"], flags["env_name"], flags["batch_size"])
    exp_prefix='%s'%(flags["exp"])
    exp_name=get_annotations_string(keys=keys, **flags)
    if flags["normalize_obs"]: flags["env_name"] += 'norm'
    exp_name = '%s-%d--'%(flags["env_name"], flags["batch_size"]) + exp_name
    log_dir = config.LOG_DIR + "/local/" + exp_prefix.replace("_", "-") + "/" + exp_name
    if flags["seed"] is not None:
        log_dir += '--s-%d'%flags["seed"]
    if not flags["overwrite"] and osp.exists(log_dir):
        ans = input("Overwrite %s?: (yes/no)"%log_dir)
        if ans != 'yes': sys.exit(0)

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        # Number of parallel workers for sampling
        n_parallel=1,
        snapshot_mode="last_best",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=flags["seed"],
        # plot=True,
        terminate_machine=True,
        sync_s3_pkl=True,
        periodic_sync_interval=1200,
        # terminate_machine=False,
        # fast_code_sync=False,
        mode=mode,
    )

def main(argv=None):
    run_experiment(mode="local")

if __name__ == '__main__':
    tf.app.run()
