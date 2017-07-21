import os, shutil
import joblib
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import special2 as special
from sandbox.rocky.tf.misc import tensor_utils
from rllab.sampler import parallel_sampler
from rllab.misc import ext
import rllab.misc.logger as logger
import numpy as np
import pyprind
import tensorflow as tf
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.pool.simple_pool import SimpleReplayPool
from sandbox.rocky.tf.misc.common_utils import memory_usage_resource
from sandbox.rocky.tf.misc.common_utils import pickle_load, pickle_dump
from rllab.exploration_strategies.base import ExplorationStrategy
import gc
from time import time
from sandbox.rocky.tf.algos.poleval import Poleval

class DDPG(RLAlgorithm, Poleval):

    def __init__(
            self,
            env,
            qf,
            es,
            policy=None,
            policy_batch_size=32,
            n_epochs=200,
            epoch_length=1000,
            discount=0.99,
            max_path_length=250,
            policy_weight_decay=0,
            policy_update_method='adam',
            policy_learning_rate=1e-3,
            policy_step_size=0.01,
            policy_optimizer_args=dict(),
            policy_updates_ratio=1.0,
            policy_use_target=True,
            policy_sample_last=False,
            eval_samples=10000,
            updates_ratio=1.0, # #updates/#samples
            scale_reward=1.0,
            include_horizon_terminal_transitions=False,
            save_freq=0,
            save_format='pickle',
            restore_auto=True,
            **kwargs):

        self.env = env
        self.policy = policy
        if self.policy is None: self.qf_dqn = True
        else: self.qf_dqn = False
        self.qf = qf
        self.es = es
        if self.es is None: self.es = ExplorationStrategy()
        self.n_epochs = n_epochs
        self.epoch_length = epoch_length
        self.discount = discount
        self.max_path_length = max_path_length

        self.init_critic(**kwargs)

        if not self.qf_dqn:
            self.policy_weight_decay = policy_weight_decay
            if policy_update_method == 'adam':
                self.policy_update_method = \
                    FirstOrderOptimizer(
                        update_method=policy_update_method,
                        learning_rate=policy_learning_rate,
                        **policy_optimizer_args,
                    )
                self.policy_learning_rate = policy_learning_rate
            elif policy_update_method == 'cg':
                self.policy_update_method = \
                    ConjugateGradientOptimizer(
                        **policy_optimizer_args,
                    )
                self.policy_step_size = policy_step_size
            self.policy_optimizer_args = policy_optimizer_args
            self.policy_updates_ratio = policy_updates_ratio
            self.policy_use_target = policy_use_target
            self.policy_batch_size = policy_batch_size
            self.policy_sample_last = policy_sample_last
            self.policy_surr_averages = []
            self.exec_policy = self.policy
        else:
            self.policy_batch_size = 0
            self.exec_policy = self.qf

        self.eval_samples = eval_samples
        self.updates_ratio = updates_ratio
        self.include_horizon_terminal_transitions = include_horizon_terminal_transitions

        self.paths = []
        self.es_path_returns = []
        self.paths_samples_cnt = 0

        self.scale_reward = scale_reward

        self.train_policy_itr = 0

        self.save_freq = save_freq
        self.save_format = save_format
        self.restore_auto = restore_auto

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.exec_policy)

    def save(self, checkpoint_dir=None):
        if checkpoint_dir is None: checkpoint_dir = logger.get_snapshot_dir()

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

            pool_file = os.path.join(checkpoint_dir, 'pool.chk')
            if self.save_format == 'pickle':
                pickle_load(pool_file)
            elif self.save_format == 'joblib':
                self.pool = joblib.load(pool_file)
            else: raise NotImplementedError

            logger.log('Restored from checkpoint %s'%checkpoint_file)
        else:
            logger.log('No checkpoint %s'%checkpoint_file)

    @overrides
    def train(self):
        global_itr = tf.Variable(0, name='global_itr', trainable=False, dtype=tf.int32)
        increment_global_itr_op = tf.assign(global_itr, global_itr+1)
        global_epoch = tf.Variable(0, name='global_epoch', trainable=False, dtype=tf.int32)
        increment_global_epoch_op = tf.assign(global_epoch, global_epoch+1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # This seems like a rather sequential method
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
            path_length = 0
            path_return = 0
            terminal = False
            initial = False
            n_updates = 0
            observation = self.env.reset()

            sample_policy = Serializable.clone(self.exec_policy, name="sample_policy")

            if self.restore_auto: self.restore()
            itr = sess.run(global_itr)
            epoch = sess.run(global_epoch)
            t0 = time()
            logger.log("Critic batch size=%d, Actor batch size=%d"%(self.qf_batch_size, self.policy_batch_size))
            while epoch < self.n_epochs:
                logger.push_prefix('epoch #%d | ' % epoch)
                logger.log("Mem: %f"%memory_usage_resource())
                logger.log("Training started")
                train_qf_itr, train_policy_itr = 0, 0
                for epoch_itr in pyprind.prog_bar(range(self.epoch_length)):
                    # Execute policy
                    if terminal:  # or path_length > self.max_path_length:
                        # Note that if the last time step ends an episode, the very
                        # last state and observation will be ignored and not added
                        # to the replay pool
                        observation = self.env.reset()
                        self.es.reset()
                        sample_policy.reset()
                        self.es_path_returns.append(path_return)
                        path_length = 0
                        path_return = 0
                        initial = True
                    else:
                        initial = False
                    action = self.es.get_action(itr, observation, policy=sample_policy)  # qf=qf)

                    next_observation, reward, terminal, _ = self.env.step(action)
                    path_length += 1
                    path_return += reward

                    if not terminal and path_length >= self.max_path_length:
                        terminal = True
                        # only include the terminal transition in this case if the flag was set
                        if self.include_horizon_terminal_transitions:
                            self.pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)
                    else:
                        self.pool.add_sample(observation, action, reward * self.scale_reward, terminal, initial)

                    observation = next_observation

                    if self.pool.size > max(self.min_pool_size, self.qf_batch_size):
                        n_updates += self.updates_ratio
                        while n_updates > 0:
                            # Train policy
                            itrs = self.do_training(itr)
                            train_qf_itr += itrs[0]
                            train_policy_itr += itrs[1]
                            n_updates -= 1
                        sample_policy.set_param_values(self.exec_policy.get_param_values())

                    itr = sess.run(increment_global_itr_op)
                    if time() - t0 > 100: gc.collect(); t0 = time()

                logger.log("Training finished")
                logger.log("Trained qf %d steps, policy %d steps"%(train_qf_itr, train_policy_itr))
                if self.pool.size >= self.min_pool_size:
                    self.evaluate(epoch, self.pool)
                    params = self.get_epoch_snapshot(epoch)
                    logger.save_itr_params(epoch, params)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()
                epoch = sess.run(increment_global_epoch_op)
                if self.save_freq > 0 and (epoch-1) % self.save_freq == 0: self.save()
            self.env.terminate()
            self.exec_policy.terminate()

    def init_opt(self):
        self.init_opt_critic()
        self.init_opt_policy()

    def init_opt_policy(self):
        if not self.qf_dqn:
            obs = self.policy.env_spec.observation_space.new_tensor_variable(
                'pol_obs',
                extra_dims=1,
            )

            if self.policy_use_target:
            	logger.log("[init_opt] using target policy.")
            	target_policy = Serializable.clone(self.policy, name="target_policy")
            else:
            	logger.log("[init_opt] no target policy.")
            	target_policy = self.policy

            policy_weight_decay_term = 0.5 * self.policy_weight_decay * \
                                   sum([tf.reduce_sum(tf.square(param))
                                        for param in self.policy.get_params(regularizable=True)])
            policy_qval = self.qf.get_e_qval_sym(
                obs, self.policy,
                deterministic=True
            )
            policy_surr = -tf.reduce_mean(policy_qval)

            policy_reg_surr = policy_surr + policy_weight_decay_term


            policy_input_list = [obs]

            if isinstance(self.policy_update_method, FirstOrderOptimizer):
                self.policy_update_method.update_opt(
                    loss=policy_reg_surr, target=self.policy, inputs=policy_input_list)

                f_train_policy = tensor_utils.compile_function(
                    inputs=policy_input_list,
                    outputs=[policy_surr, self.policy_update_method._train_op],
                )
            else:
                f_train_policy = self.policy_update_method.update_opt_trust_region(
                        loss=policy_reg_surr,
                        input_list=policy_input_list,
                        obs_var=obs,
                        target=self.policy,
                        policy=self.policy,
                        step_size=self.policy_step_size,
                )

            self.opt_info = dict(
                f_train_policy=f_train_policy,
                target_policy=target_policy,
            )

    def do_training(self, itr):
        batch = self.pool.random_batch(self.qf_batch_size)
        self.do_critic_training(itr, batch=batch)

        train_policy_itr = 0

        if not self.qf_dqn and self.pool.size > max(
                self.min_pool_size, self.policy_batch_size):
            self.train_policy_itr += self.policy_updates_ratio
            while self.train_policy_itr > 0:
                if self.policy_sample_last:
                    pol_batch = self.pool.last_batch(self.policy_batch_size)
                else:
                    pol_batch = self.pool.random_batch(self.policy_batch_size)
                self.do_policy_training(itr, batch=pol_batch)
                self.train_policy_itr -= 1
                train_policy_itr += 1

        return 1, train_policy_itr # number of itrs qf, policy are trained

    def do_policy_training(self, itr, batch):
        target_policy = self.opt_info["target_policy"]
        obs, = ext.extract(batch, "observations")
        f_train_policy = self.opt_info["f_train_policy"]
        if isinstance(self.policy_update_method, FirstOrderOptimizer):
            policy_surr, _ = f_train_policy(obs)
        else:
            agent_infos = self.policy.dist_info(obs)
            state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
            dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            all_input_values = (obs, obs, ) + tuple(state_info_list) + tuple(dist_info_list)
            policy_results = f_train_policy(all_input_values)
            policy_surr = policy_results["loss_after"]
        if self.policy_use_target:
            target_policy.set_param_values(
                target_policy.get_param_values() * (1.0 - self.soft_target_tau) +
                self.policy.get_param_values() * self.soft_target_tau)
        self.policy_surr_averages.append(policy_surr)

    def evaluate(self, epoch, pool):
        logger.log("Collecting samples for evaluation")
        paths = parallel_sampler.sample_paths(
            policy_params=self.exec_policy.get_param_values(),
            max_samples=self.eval_samples,
            max_path_length=self.max_path_length,
        )

        average_discounted_return = np.mean(
            [special.discount_return(path["rewards"], self.discount) for path in paths]
        )

        returns = [sum(path["rewards"]) for path in paths]

        average_action = np.mean(np.square(np.concatenate(
            [path["actions"] for path in paths]
        )))

        qfun_reg_param_norm = np.linalg.norm(
            self.qf.get_param_values(regularizable=True)
        )

        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Iteration', epoch)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn',
                              np.std(returns))
        logger.record_tabular('MaxReturn',
                              np.max(returns))
        logger.record_tabular('MinReturn',
                              np.min(returns))
        if len(self.es_path_returns) > 0:
            logger.record_tabular('AverageEsReturn',
                                  np.mean(self.es_path_returns))
            logger.record_tabular('StdEsReturn',
                                  np.std(self.es_path_returns))
            logger.record_tabular('MaxEsReturn',
                                  np.max(self.es_path_returns))
            logger.record_tabular('MinEsReturn',
                                  np.min(self.es_path_returns))
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageAction', average_action)

        logger.record_tabular('QFunRegParamNorm',
                              qfun_reg_param_norm)
        self.env.log_diagnostics(paths)
        self.log_critic_training()

        self.es_path_returns = []

        if not self.qf_dqn:
            average_policy_surr = np.mean(self.policy_surr_averages)
            policy_reg_param_norm = np.linalg.norm(
                self.policy.get_param_values(regularizable=True)
            )
            logger.record_tabular('AveragePolicySurr', average_policy_surr)
            logger.record_tabular('PolicyRegParamNorm',
                              policy_reg_param_norm)
            self.policy.log_diagnostics(paths)
            self.policy_surr_averages = []

    def get_epoch_snapshot(self, epoch):
        snapshot = dict(
            env=self.env,
            epoch=epoch,
            qf=self.qf,
            target_qf=self.opt_info_critic["target_qf"],
            es=self.es,
        )
        if not self.qf_dqn:
            snapshot.update(dict(
                policy=self.policy,
                target_policy=self.opt_info["target_policy"],
            ))
        return snapshot

