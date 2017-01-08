from sandbox.rocky.tf.launchers.algo_gym_stub import run_experiment
from sandbox.rocky.tf.launchers.launcher_utils import keys_by_algo_map, FLAGS
from itertools import product
from functools import reduce
import tensorflow as tf
from copy import deepcopy
flags = tf.app.flags
flags.DEFINE_string('settings', 'example-settings.py', 'Settings file.')

def execute(params, fix_params, algo_name, mode):
    keys = keys_by_algo_map[algo_name]
    values = [params[k] for k in keys]
    N = reduce(lambda x, y: x*y, [len(v) for v in values])
    if not FLAGS.force_start:
        ans = input("Launch %d %s jobs?: (yes/no)"%(N, algo_name))
        if ans != 'yes': print("Skipping %s..."%algo_name); return 0
    for params_combo in product(*values):
        params_dict = {k: p for k, p in zip(keys, params_combo)}
        params_dict['algo_name'] = algo_name
        params_dict['overwrite'] = True
        for k, v in fix_params.items(): params_dict[k] = v
        run_experiment(mode=mode, keys=keys, params=params_dict)
    return N

def run_execute(params, fix_params, mode="ec2"):
    N = 0
    if FLAGS.algo_name == 'all':
        algos = ['trpo', 'qprop', 'qvpg', 'vpg', 'ddpg']
        for algo_name in algos:
            N += execute(params=params, fix_params=fix_params, algo_name=algo_name, mode=mode)
    elif FLAGS.algo_name == 'all_less_ddpg':
        algos = ['trpo', 'qprop', 'qvpg', 'vpg']
        for algo_name in algos:
            N += execute(params=params, fix_params=fix_params, algo_name=algo_name, mode=mode)
    elif FLAGS.algo_name == 'all_qprop':
        algos = ['qprop', 'qvpg']
        for algo_name in algos:
            N += execute(params=params, fix_params=fix_params, algo_name=algo_name, mode=mode)
    else:
        N += execute(params=params, fix_params=fix_params, algo_name=FLAGS.algo_name, mode=mode)
    print('Launched %d jobs.'%N)

def main(argv=None):
    import importlib
    import os.path as osp
    module_name = 'sandbox.rocky.tf.launchers.experiments.python.%s'%(
        osp.splitext(osp.basename(FLAGS.settings))[0])
    mod = importlib.import_module(module_name)
    default_module_name = 'sandbox.rocky.tf.launchers.experiments.python.%s'%(
        'example-settings')
    default_mod = importlib.import_module(default_module_name)
    default_mod.fix_params.update(mod.fix_params)
    if type(mod.params) == list:
        for params in mod.params:
            default_params = deepcopy(default_mod.params)
            default_params.update(params)
            run_execute(params=default_params, fix_params=default_mod.fix_params)
    else:
        default_mod.params.update(mod.params)
        run_execute(params=default_mod.params, fix_params=default_mod.fix_params)

if __name__ == '__main__':
    tf.app.run()
