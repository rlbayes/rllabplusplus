import shutil
import os
import time
import numpy as np
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.pool.simple_pool import SimpleReplayPool
from sandbox.rocky.tf.misc.common_utils import memory_usage_resource
from sandbox.rocky.tf.misc.common_utils import pickle_load, pickle_dump
import gc
import joblib
from sandbox.rocky.tf.algos.poleval import Poleval
from rllab.sampler.utils import rollout

class BatchPolopt(RLAlgorithm, Poleval):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            # qprop params
            qf=None,
            qf_updates_ratio=1,
            qprop_eta_option='ones',
            qprop_nu=0,
            save_freq=0,
            restore_auto=True,
            policy_sample_last=True,
            qprop=True,
            ac_sample_backups=0,
            ac_delta=0,
            save_format='pickle',
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :param qf: q function for q-prop.
        :return:
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.qf = qf
        if self.qf is not None:
            self.qprop = qprop
            self.qf_updates_ratio = qf_updates_ratio
            self.qprop_nu = qprop_nu
            self.qprop_eta_option = qprop_eta_option
            self.policy_sample_last = policy_sample_last

            self.ac_delta = ac_delta
            if self.ac_delta > 0:
                self.ac_sample_backups = ac_sample_backups

            self.init_critic(**kwargs)

            self.qf_dqn = False
            assert self.qprop or self.ac_delta > 0, "Error: no use for qf."
        else:
            self.ac_delta = 0
            self.qprop = False
            self.qf_mc_ratio = 0
            self.qf_residual_phi = 0

        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()

        self.sampler = sampler_cls(self, **sampler_args)

        self.save_freq = save_freq
        self.save_format = save_format
        self.restore_auto = restore_auto

    def start_worker(self):
        self.sampler.start_worker()

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def save(self, checkpoint_dir=None):
        if checkpoint_dir is None: checkpoint_dir = logger.get_snapshot_dir()

        if self.qf is not None:
            pool_file = os.path.join(checkpoint_dir, 'pool.chk')
            if self.save_format == 'pickle':
                pickle_dump(pool_file + '.tmp', self.pool)
            elif self.save_format == 'joblib':
                joblib.dump(self.pool, pool_file + '.tmp', compress=1, cache_size=1e9)
            else: raise NotImplementedError
            shutil.move(pool_file + '.tmp', pool_file)

        checkpoint_file = os.path.join(checkpoint_dir, 'params.chk')
        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.save(sess, checkpoint_file)

        tabular_file = os.path.join(checkpoint_dir, 'progress.csv')
        if os.path.isfile(tabular_file):
            tabular_chk_file = os.path.join(checkpoint_dir, 'progress.csv.chk')
            shutil.copy(tabular_file, tabular_chk_file)

        logger.log('Saved to checkpoint %s'%checkpoint_file)

    def restore(self, checkpoint_dir=None):
        if checkpoint_dir is None: checkpoint_dir = logger.get_snapshot_dir()
        checkpoint_file = os.path.join(checkpoint_dir, 'params.chk')
        if os.path.isfile(checkpoint_file + '.meta'):
            sess = tf.get_default_session()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)

            tabular_chk_file = os.path.join(checkpoint_dir, 'progress.csv.chk')
            if os.path.isfile(tabular_chk_file):
                tabular_file = os.path.join(checkpoint_dir, 'progress.csv')
                logger.remove_tabular_output(tabular_file)
                shutil.copy(tabular_chk_file, tabular_file)
                logger.add_tabular_output(tabular_file)

            if self.qf is not None:
                pool_file = os.path.join(checkpoint_dir, 'pool.chk')
                if self.save_format == 'pickle':
                    pickle_load(pool_file)
                elif self.save_format == 'joblib':
                    self.pool = joblib.load(pool_file)
                else: raise NotImplementedError

            logger.log('Restored from checkpoint %s'%checkpoint_file)
        else:
            logger.log('No checkpoint %s'%checkpoint_file)

    def train(self, sess=None):
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        increment_global_step_op = tf.assign(global_step, global_step+1)
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
        sess.run(tf.global_variables_initializer())
        if self.qf is not None:
            self.pool = SimpleReplayPool(
                max_pool_size=self.replay_pool_size,
                observation_dim=self.env.observation_space.flat_dim,
                action_dim=self.env.action_space.flat_dim,
                replacement_prob=self.replacement_prob,
                env=self.env,
            )
        self.start_worker()
        self.init_opt()
        # This initializes the optimizer parameters
        sess.run(tf.global_variables_initializer())
        if self.restore_auto: self.restore()
        itr = sess.run(global_step)
        start_time = time.time()
        t0 = time.time()
        while itr < self.n_itr:
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Mem: %f"%memory_usage_resource())
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                if self.qf is not None:
                    logger.log("Adding samples to replay pool...")
                    self.add_pool(itr, paths, self.pool)
                    logger.log("Optimizing critic before policy...")
                    self.optimize_critic(itr, self.pool, samples_data)
                logger.log("Optimizing policy...")
                self.optimize_policy(itr, samples_data)
                self.log_critic_training()
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
                if time.time() - t0 > 10: gc.collect(); t0 = time.time()
                itr = sess.run(increment_global_step_op)
                if self.save_freq > 0 and (itr-1) % self.save_freq == 0: self.save()

        self.shutdown_worker()
        if created_session:
            sess.close()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_control_variate(self, observations, actions):
        control_variate = self.opt_info_qprop["f_control_variate"](observations, actions)
        return control_variate

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def add_pool(self, itr, paths, pool):
        # Add samples to replay pool
        path_lens = []
        for path in paths:
            path_len = path["observations"].shape[0]
            for i in range(path_len):
                observation = path["observations"][i]
                action = path["actions"][i]
                reward = path["rewards"][i]
                terminal = path["terminals"][i]
                initial = i == 0
                pool.add_sample(observation, action, reward, terminal, initial)
            path_lens.append(path_len)
        path_lens = np.array(path_lens)
        logger.log("PathsInfo epsN=%d, meanL=%.2f, maxL=%d, minL=%d"%(
            len(paths), path_lens.mean(), path_lens.max(), path_lens.min()))
        logger.log("Put %d transitions to replay, size=%d"%(path_lens.sum(), pool.size))

    def optimize_critic(self, itr, pool, samples_data):
        # Train the critic
        if pool.size >= self.min_pool_size:
            qf_itrs = float(self.batch_size)*self.qf_updates_ratio
            qf_itrs = int(np.ceil(qf_itrs))
            logger.log("Fitting critic for %d iterations, batch size=%d"%(
                qf_itrs, self.qf_batch_size))
            for i in range(qf_itrs):
                # Train policy
                batch = pool.random_batch(self.qf_batch_size)
                self.do_critic_training(itr, batch, samples_data)

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
