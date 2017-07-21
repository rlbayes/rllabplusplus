from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import gc

class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            sample_backups=0,
            kl_sample_backups=0,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.sample_backups = sample_backups
        self.kl_sample_backups = kl_sample_backups
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, name=''):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            name + 'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            name + 'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            name + 'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=name+'old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=name+k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name=name+"valid")
        else:
            valid_var = None

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        if self.kl_sample_backups > 0:
            kl_obs_var = self.env.observation_space.new_tensor_variable(
                name + 'kl_obs',
                extra_dims=1 + is_recurrent,
            )
            kl_old_dist_info_vars = {
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=name+'kl_old_%s' % k)
                for k, shape in dist.dist_info_specs
                }
            kl_old_dist_info_vars_list = [kl_old_dist_info_vars[k] for k in dist.dist_info_keys]

            kl_state_info_vars = {
                k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=name+'kl_%s'%k)
                for k, shape in self.policy.state_info_specs
                }
            kl_state_info_vars_list = [kl_state_info_vars[k] for k in self.policy.state_info_keys]
            kl_dist_info_vars = self.policy.dist_info_sym(kl_obs_var, kl_state_info_vars)
            kl = dist.kl_sym(kl_old_dist_info_vars, kl_dist_info_vars)

            input_list += [kl_obs_var] + kl_state_info_vars_list + kl_old_dist_info_vars_list

            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        else:
            dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)

        if not self.qprop:
            if is_recurrent:
                mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
                surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
            else:
                mean_kl = tf.reduce_mean(kl)
                surr_loss = - tf.reduce_mean(lr * advantage_var)
        else:
            if is_recurrent: raise NotImplementedError
            eta_var = tensor_utils.new_tensor(
                'eta',
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            )
            surr_loss = -tf.reduce_mean(lr * advantage_var)
            if self.qprop_nu > 0: surr_loss *= 1-self.qprop_nu
            if self.sample_backups > 0 or not self.policy_sample_last:
                off_obs_var = self.env.observation_space.new_tensor_variable(
                    name + 'off_obs',
                    extra_dims=1 + is_recurrent,
                )
                off_e_qval = self.qf.get_e_qval_sym(off_obs_var, self.policy, deterministic=True)
                input_list += [off_obs_var]
                surr_loss -= tf.reduce_mean(off_e_qval)# * eta_var)
            else:
                e_qval = self.qf.get_e_qval_sym(obs_var, self.policy, deterministic=True)
                surr_loss -= tf.reduce_mean(e_qval * eta_var)
            mean_kl = tf.reduce_mean(kl)
            input_list += [eta_var]
            control_variate = self.qf.get_cv_sym(obs_var,
                    action_var, self.policy)
            f_control_variate = tensor_utils.compile_function(
                inputs=[obs_var, action_var],
                outputs=control_variate,
            )
            self.opt_info_qprop = dict(
                f_control_variate=f_control_variate,
            )
        if self.ac_delta > 0:
            ac_obs_var = self.env.observation_space.new_tensor_variable(
                name + 'ac_obs',
                extra_dims=1 + is_recurrent,
            )
            e_qval = self.qf.get_e_qval_sym(ac_obs_var, self.policy, deterministic=True)
            input_list += [ac_obs_var]
            surr_loss *= (1.0 - self.ac_delta)
            surr_loss -= self.ac_delta * tf.reduce_mean(e_qval)
        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        self.opt_info = dict(
                target_policy=self.policy,
        )
        self.init_opt_critic()
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        base_batch_size = len(all_input_values[0])
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        if self.kl_sample_backups > 0:
            if not self.qprop: raise NotImplementedError
            batch_size = (self.kl_sample_backups+1) * base_batch_size
            if self.pool.size < batch_size:
                logger.log("Using on-policy samples to estimate KL.")
                kl_obs = all_input_values[0]
                kl_agent_infos = agent_infos
            else:
                logger.log("Using last %d samples to estimate KL."%batch_size)
                batch_data = self.pool.last_batch(batch_size=batch_size)
                kl_obs = batch_data["observations"]
                kl_agent_infos = self.policy.dist_info(kl_obs)
            kl_state_info_list = [kl_agent_infos[k] for k in self.policy.state_info_keys]
            kl_dist_info_list = [kl_agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            all_input_values += (kl_obs, )
            all_input_values += tuple(kl_state_info_list) + tuple(kl_dist_info_list)
        if self.qprop:
            if self.sample_backups > 0 or not self.policy_sample_last:
                batch_size = (self.sample_backups+1) * base_batch_size
                if self.pool.size < batch_size:
                    logger.log("Using on-policy samples to estimate Q-Prop AC.")
                    off_obs = all_input_values[0]
                else:
                    if self.policy_sample_last or self.ac_delta > 0:
                        logger.log("Using last %d off-policy samples to estimate Q-Prop AC."%(
                            batch_size))
                        batch_data = self.pool.last_batch(batch_size=batch_size)
                    else:
                        logger.log("Using random %d off-policy samples to estimate Q-Prop AC."%(
                            batch_size))
                        batch_data = self.pool.random_batch(batch_size=batch_size)
                    off_obs = batch_data["observations"]
                all_input_values += (off_obs, )
            all_input_values += (samples_data["etas"], )
        if self.ac_delta > 0:
            batch_size = (self.ac_sample_backups+1) * base_batch_size
            if self.pool.size < batch_size or \
                    (self.ac_sample_backups==0 and self.policy_sample_last):
                logger.log("Using on-policy samples to estimate AC.")
                ac_obs = all_input_values[0]
            else:
                if self.policy_sample_last:
                    logger.log("Using last %d off-policy samples to estimate AC."%(
                        batch_size))
                    batch_data = self.pool.last_batch(batch_size=batch_size)
                else:
                    logger.log("Using random %d off-policy samples to estimate AC."%(
                        batch_size))
                    batch_data = self.pool.random_batch(batch_size=batch_size)
                ac_obs = batch_data["observations"]
            all_input_values += (ac_obs, )
        optimizer = self.optimizer
        logger.log("Computing loss before")
        loss_before = optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        gc.collect()
        optimizer.optimize(all_input_values)
        gc.collect()
        logger.log("Computing KL after")
        mean_kl = optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
