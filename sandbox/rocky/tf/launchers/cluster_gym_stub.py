from sandbox.rocky.tf.launchers.algo_gym_stub import run_experiment
from sandbox.rocky.tf.launchers.launcher_utils import FLAGS
from itertools import product
from functools import reduce
import tensorflow as tf
import sys

def execute(params, mode):
    keys = params.keys()
    values = [params[k] for k in keys]
    N = reduce(lambda x, y: x*y, [len(v) for v in values])
    if not FLAGS.force_start:
        ans = input("Launch %d for algo=%s jobs?: (yes/no)"%(N, str(params["algo_name"])))
        if ans != 'yes': print("Skipping..."); return 0
    for params_combo in product(*values):
        params_dict = {k: p for k, p in zip(keys, params_combo)}
        params_dict['overwrite'] = True
        run_experiment(mode=mode, keys=keys, params=params_dict)
    return N

def main(argv=None):
    if len(sys.argv) <= 2:
        print('Usage ./%s --exp=<exp_name> --ec2_settings=<relative_path_to_ec2_settings_file>'%sys.argv[0])
        sys.exit(0)
    import importlib
    import os.path as osp
    import shutil
    module_name = 'sandbox.rocky.tf.launchers.%s'%(
        osp.splitext(FLAGS.ec2_settings)[0].replace('/','.'))
    mod = importlib.import_module(module_name)
    dst_py = osp.join(osp.dirname(FLAGS.ec2_settings),FLAGS.exp+'.py')
    try:
        shutil.copy(FLAGS.ec2_settings, dst_py)
    except shutil.SameFileError as e:
        print(e)
    if type(mod.params) != list: mod.params = [mod.params]
    if hasattr(mod, 'base_params'): base_params = mod.base_params
    else: base_params = dict()
    N = 0
    for params in mod.params:
        ps = base_params.copy()
        ps.update(params)
        N += execute(params=ps, mode="ec2")
    print('Launched %d jobs.'%N)

if __name__ == '__main__':
    tf.app.run()
