import time
import numpy as np
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from rllab.pool.simple_pool import SimpleReplayPool
from rllab.misc import ext
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

class BatchPolopt(RLAlgorithm):
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
            min_pool_size=10000,
            replay_pool_size=1000000,
            replacement_prob=1.0,
            qf_updates_ratio=1,
            qprop_use_mean_action=True,
            qprop_min_itr=0,
            qprop_batch_size=None,
            qprop_use_advantage=True,
            qprop_use_qf_baseline=False,
            qprop_eta_option='ones',
            qf_weight_decay=0.,
            qf_update_method='adam',
            qf_learning_rate=1e-3,
            qf_batch_size=32,
            qf_baseline=None,
            soft_target=True,
            soft_target_tau=0.001,
            scale_reward=1.0,
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
            self.qprop = True
            self.qprop_optimizer = Serializable.clone(self.optimizer)
            self.min_pool_size = min_pool_size
            self.replay_pool_size = replay_pool_size
            self.replacement_prob = replacement_prob
            self.qf_updates_ratio = qf_updates_ratio
            self.qprop_use_mean_action = qprop_use_mean_action
            self.qprop_min_itr = qprop_min_itr
            self.qprop_use_qf_baseline = qprop_use_qf_baseline
            self.qf_weight_decay = qf_weight_decay
            self.qf_update_method = \
                FirstOrderOptimizer(
                    update_method=qf_update_method,
                    learning_rate=qf_learning_rate,
                )
            self.qf_learning_rate = qf_learning_rate
            self.qf_batch_size = qf_batch_size
            self.qf_baseline = qf_baseline
            if qprop_batch_size is None:
                self.qprop_batch_size = self.batch_size
            else:
                self.qprop_batch_size = qprop_batch_size
            self.qprop_use_advantage = qprop_use_advantage
            self.qprop_eta_option = qprop_eta_option
            self.soft_target_tau = soft_target_tau
            self.scale_reward = scale_reward

            self.qf_loss_averages = []
            self.q_averages = []
            self.y_averages = []
            if self.start_itr >= self.qprop_min_itr:
                self.batch_size = self.qprop_batch_size
                if self.qprop_use_qf_baseline:
                    self.baseline = self.qf_baseline
                self.qprop_enable = True
            else:
                self.qprop_enable = False
        else:
            self.qprop = False
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()

        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr)

    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            if self.qprop:
                pool = SimpleReplayPool(
                    max_pool_size=self.replay_pool_size,
                    observation_dim=self.env.observation_space.flat_dim,
                    action_dim=self.env.action_space.flat_dim,
                    replacement_prob=self.replacement_prob,
                )
            self.start_worker()
            self.init_opt()
            # This initializes the optimizer parameters
            sess.run(tf.initialize_all_variables())
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    if self.qprop and not self.qprop_enable and \
                            itr >= self.qprop_min_itr:
                        logger.log("Restarting workers with batch size %d->%d..."%(
                            self.batch_size, self.qprop_batch_size))
                        self.shutdown_worker()
                        self.batch_size = self.qprop_batch_size
                        self.start_worker()
                        if self.qprop_use_qf_baseline:
                            self.baseline = self.qf_baseline
                        self.qprop_enable = True
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    if self.qprop:
                        logger.log("Adding samples to replay pool...")
                        self.add_pool(itr, paths, pool)
                        logger.log("Optimizing critic before policy...")
                        self.optimize_critic(itr, pool)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()

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

    def init_opt_critic(self, vars_info, qbaseline_info):
        assert(not self.policy.recurrent)

        # Compute Taylor expansion Q function
        delta = vars_info["action_var"] - qbaseline_info["action_mu"]
        control_variate = tf.reduce_sum(delta * qbaseline_info["qprime"], 1)
        if not self.qprop_use_advantage:
            control_variate += qbaseline_info["qvalue"]
            logger.log("Qprop, using Q-value over A-value")
        f_control_variate = tensor_utils.compile_function(
            inputs=[vars_info["obs_var"], vars_info["action_var"]],
            outputs=[control_variate, qbaseline_info["qprime"]],
        )

        target_qf = Serializable.clone(self.qf, name="target_qf")

        # y need to be computed first
        obs = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )

        # The yi values are computed separately as above and then passed to
        # the training functions below
        action = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        yvar = tf.placeholder(dtype=tf.float32, shape=[None], name='ys')

        qf_weight_decay_term = 0.5 * self.qf_weight_decay * \
                               sum([tf.reduce_sum(tf.square(param)) for param in
                                    self.qf.get_params(regularizable=True)])

        qval = self.qf.get_qval_sym(obs, action)

        qf_loss = tf.reduce_mean(tf.square(yvar - qval))
        qf_reg_loss = qf_loss + qf_weight_decay_term

        qf_input_list = [yvar, obs, action]

        self.qf_update_method.update_opt(
            loss=qf_reg_loss, target=self.qf, inputs=qf_input_list)

        f_train_qf = tensor_utils.compile_function(
            inputs=qf_input_list,
            outputs=[qf_loss, qval, self.qf_update_method._train_op],
        )

        self.opt_info_critic = dict(
            f_train_qf=f_train_qf,
            target_qf=target_qf,
            f_control_variate=f_control_variate,
        )

    def get_control_variate(self, observations, actions):
        control_variate, qprime = self.opt_info_critic["f_control_variate"](observations, actions)
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
                pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
            path_lens.append(path_len)
        path_lens = np.array(path_lens)
        logger.log("PathsInfo epsN=%d, meanL=%.2f, maxL=%d, minL=%d"%(
            len(paths), path_lens.mean(), path_lens.max(), path_lens.min()))
        logger.log("Put %d transitions to replay, size=%d"%(path_lens.sum(), pool.size))

    def optimize_critic(self, itr, pool):
        # Train the critic
        if pool.size >= self.min_pool_size:
            #qf_itrs = float(self.batch_size)/self.qf_batch_size*self.qf_updates_ratio
            qf_itrs = float(self.batch_size)*self.qf_updates_ratio
            qf_itrs = int(np.ceil(qf_itrs))
            logger.log("Fitting critic for %d iterations, batch size=%d"%(
                qf_itrs, self.qf_batch_size))
            for i in range(qf_itrs):
                # Train policy
                batch = pool.random_batch(self.qf_batch_size)
                self.do_training(itr, batch)

    def do_training(self, itr, batch):

        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )

        # compute the on-policy y values
        target_qf = self.opt_info_critic["target_qf"]

        next_actions, next_actions_dict = self.policy.get_actions(next_obs)
        if self.qprop_use_mean_action:
            next_actions = next_actions_dict["mean"]
        next_qvals = target_qf.get_qval(next_obs, next_actions)

        ys = rewards + (1. - terminals) * self.discount * next_qvals

        f_train_qf = self.opt_info_critic["f_train_qf"]

        qf_loss, qval, _ = f_train_qf(ys, obs, actions)

        target_qf.set_param_values(
            target_qf.get_param_values() * (1.0 - self.soft_target_tau) +
            self.qf.get_param_values() * self.soft_target_tau)

        self.qf_loss_averages.append(qf_loss)
        self.q_averages.append(qval)
        self.y_averages.append(ys)

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
