import argparse

import joblib
import tensorflow as tf
#from sandbox.rocky.tf.core.parameterized import suppress_params_loading

from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--record_gym', type=bool, default=True,
                        help='Record video and log for gym environment.')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    while True:
        with tf.Session() as sess:
            #with tf.variable_scope("load_policy"):
            #with tf.variable_scope("load_policy", reuse=True):
                data = joblib.load(args.file)
                policy = data['policy']
                env = data['env']
                if args.record_gym:
                    from sandbox.rocky.tf.launchers.launcher_utils import get_env
                    import rllab.misc.logger as logger
                    import os.path as osp
                    logger.set_snapshot_dir(osp.dirname(args.file))
                    env = get_env(env_name = env._wrapped_env._wrapped_env.env_id,
                        record_video=True, record_log=True)
                while True:
                    path = rollout(env, policy, max_path_length=args.max_path_length,
                                   animated=True, always_return_paths=True, speedup=args.speedup)
                    print("Path length=%d, reward=%f"%(len(path["rewards"]),path["rewards"].sum()))
