


from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np

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
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.sample_backups = sample_backups
        super(NPO, self).__init__(**kwargs)

    def init_opt_vars(self, name=''):
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

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = - tf.reduce_sum(lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)
        return {
            "mean_kl": mean_kl,
            "input_list": input_list,
            "obs_var": obs_var,
            "action_var": action_var,
            "advantage_var": advantage_var,
            "surr_loss": surr_loss,
            "dist_info_vars": dist_info_vars,
            "lr": lr,
        }

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        vars_info = self.init_opt_vars()

        if self.sample_backups > 0:
            assert(not self.policy.recurrent)
            vars_info_kl = self.init_opt_vars(name='kl_')
            vars_info["input_list"] += vars_info_kl["input_list"]
            vars_info["mean_kl"] = vars_info_kl["mean_kl"]
            self.input_values_backups = None

        self.optimizer.update_opt(
            loss=vars_info["surr_loss"],
            target=self.policy,
            leq_constraint=(vars_info["mean_kl"], self.step_size),
            inputs=vars_info["input_list"],
            constraint_name="mean_kl"
        )

        if self.qprop:
            eta_var = tensor_utils.new_tensor(
                'eta',
                ndim=1 + is_recurrent,
                dtype=tf.float32,
            )
            qbaseline_info = self.qf_baseline.get_qbaseline_sim(
                vars_info["obs_var"], scale_reward=self.scale_reward)
            qprop_surr_loss = - tf.reduce_mean(vars_info["lr"] *
                vars_info["advantage_var"]) - tf.reduce_mean(
                qbaseline_info["qvalue"] * eta_var)
            self.qprop_optimizer.update_opt(
                loss=qprop_surr_loss,
                target=self.policy,
                leq_constraint=(vars_info["mean_kl"], self.step_size),
                inputs=vars_info["input_list"] + [eta_var],
                constraint_name="mean_kl"
            )
            self.init_opt_critic(vars_info=vars_info, qbaseline_info=qbaseline_info)
        return dict()

    def merge_input_values(self, all_input_values):
        assert(self.sample_backups > 0)
        if self.input_values_backups is None:
            self.input_values_backups = [ v
                for v in all_input_values]
        else:
            cutoff = self.batch_size * (1+self.sample_backups)
            backups = [np.concatenate((v1, v2), axis=0)
                for v1, v2 in zip(self.input_values_backups,
                all_input_values)]
            self.input_values_backups = [ v[-cutoff:]
                for v in backups ]
        n_samples = self.input_values_backups[0].shape[0]
        logger.log("Using %d sample backups for KL"% n_samples)
        return all_input_values + tuple(self.input_values_backups)

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        if self.sample_backups > 0:
            all_input_values = self.merge_input_values(all_input_values)
        if self.qprop and self.qprop_enable:
            optimizer = self.qprop_optimizer
            all_input_values += (samples_data["etas"], )
            logger.log("Using Qprop optimizer")
        else:
            optimizer = self.optimizer
        logger.log("Computing loss before")
        loss_before = optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        optimizer.optimize(all_input_values)
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
