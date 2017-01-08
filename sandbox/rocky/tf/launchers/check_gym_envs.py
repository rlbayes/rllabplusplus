
from sandbox.rocky.tf.launchers.launcher_utils import FLAGS, get_env_info
from pprint import pprint
import tensorflow as tf

def main(argv=None):
    info, _ = get_env_info(**FLAGS.__flags)
    pprint(info)

if __name__ == '__main__':
    tf.app.run()
