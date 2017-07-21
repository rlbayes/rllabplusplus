import tensorflow as tf
from sandbox.rocky.tf.launchers.algo_gym_stub import set_experiment
from sandbox.rocky.tf.launchers.launcher_utils import keys_by_algo_map

def main(argv=None):
    for i, key in enumerate(keys_by_algo_map):
        print('[Testing %d/%d] algo_name=%s'%(i, len(keys_by_algo_map), key))
        algo, _ = set_experiment(params=dict(algo_name=key, overwrite=True))

if __name__ == '__main__':
    tf.app.run()
