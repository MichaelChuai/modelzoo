import numpy as np
import tensorflow as tf
import collections
from ..model import average_grad
from ..misc import delete_and_make_dir, create_op_graph
from ..tftrain import load_ckpt, load_ckpt_path, create_init_op
from ..metrics import get_metric_name
from tqdm import trange

__all__ = ['ENASModelTensors', 'ENASModel']

ENASModelTensors = collections.namedtuple('ENASModelTensors',
                                          ('inputs',
                                           'labels',
                                           'is_training',
                                           'is_fixed_arc',
                                           'fixed_arc',
                                           'child',
                                           'controller',
                                           'train_input_func',
                                           'reward_input_func',
                                           'reward_metric_func',
                                           'reward_batches',
                                           'child_loss_func',
                                           'child_optimizer',
                                           'controller_optimizer'))


class ENASModel:
    def __init__(self, enas_tensors: ENASModelTensors, n_gpu, ckpt_dir=None, training=True, one_shot_reward=False):
        self.inputs = enas_tensors.inputs
        self.labels = enas_tensors.labels
        self.is_training = enas_tensors.is_training
        self.is_fixed_arc = enas_tensors.is_fixed_arc
        self.fixed_arc = enas_tensors.fixed_arc
        self.child = enas_tensors.child
        self.controller = enas_tensors.controller
        self.train_input_func = enas_tensors.train_input_func
        self.reward_input_func = enas_tensors.reward_input_func
        self.reward_metric_func = enas_tensors.reward_metric_func
        self.reward_batches = enas_tensors.reward_batches
        self.one_shot_reward = one_shot_reward
        self.child_loss_func = enas_tensors.child_loss_func
        self.child_optimizer = enas_tensors.child_optimizer
        self.controller_optimizer = enas_tensors.controller_optimizer

        self.n_gpu = n_gpu
        self.ckpt_dir = ckpt_dir
        self.sess = None
        self.model_loaded = False
        self.graph_def = None

        self.child.connect_controller(self.controller)
        with tf.name_scope('main'):
            self.controller()
            self.child_logits = self.child(self.inputs, self.is_training, self.is_fixed_arc, self.fixed_arc)
        self.progress_range = trange
        if training:
            with tf.name_scope('training'):
                self._build_training_graph()

    def load_weights(self, sess, ckpt_path=None):
        self.sess = sess
        if ckpt_path is None:
            load_ckpt(sess, self.ckpt_dir)
        else:
            load_ckpt_path(sess, ckpt_path)
        self.model_loaded = True
        self.graph_def = sess.graph_def

    def _build_training_graph(self):
        with tf.device('/cpu:0'):
            tower_grads = []
            child_losses = []
            for i in range(self.n_gpu):
                with tf.device(f'/gpu:{i}'):
                    with tf.name_scope(f'GPU_{i}'):
                        (x, y), _ = self.train_input_func()
                        child_logits = self.child(x, tf.constant(True), tf.constant(False), tf.constant([0]))
                        child_loss = self.child_loss_func(y, child_logits)
                        grads = self.child_optimizer.compute_gradients(child_loss)
                        grads = list(filter(lambda g: g[0] is not None, grads))
                        tower_grads.append(grads)
                        child_losses.append(child_loss)
            self.child_loss = tf.reduce_mean(child_losses)
            grads = average_grad(tower_grads)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.child_optimizer.apply_gradients(grads, global_step=self.child.train_step)
        self.controller_loss = self.controller.get_loss(self.fixed_arc, self.reward_input_func, self.reward_metric_func,
                                                        self.reward_batches, self.one_shot_reward)
        self.controller_train_op = self.controller_optimizer.minimize(self.controller_loss,
                                                                      global_step=self.controller.train_step)
        self.graph_def = tf.get_default_graph().as_graph_def()

    def train(self,
              num_steps,
              summ_steps,
              ckpt_steps,
              metric_opdefs,
              num_sampled_childs,
              num_ckpts_training_controller,
              num_ckpts_before_sampling_childs,
              controller_steps=None,
              controller_summ_steps=None,
              train_controller=True,
              extra_summ_ops=None,
              listeners=None,
              max_ckpt_to_keep=10,
              sess_config_proto=None,
              from_scratch=True):
        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        metric_summ_names = ['train/child_{0}'.format(m.name.split('/')[-2]) for m in metric_ops]
        metric_summ_ops = [tf.summary.scalar(*tup) for tup in list(zip(metric_summ_names, metric_ops))]
        summ_ops = metric_summ_ops + list(extra_summ_ops) if extra_summ_ops else metric_summ_ops
        summ_op = tf.summary.merge(summ_ops)
        summ_loss_op = tf.summary.scalar('train/child_loss', self.child_loss)
        if from_scratch:
            delete_and_make_dir(self.ckpt_dir)
        child_global_step = self.child.train_step
        data_gntr, _ = self.train_input_func()
        num_ckpts = num_steps // ckpt_steps
        saver = tf.train.Saver(max_to_keep=max_ckpt_to_keep)
        summ_writer = tf.summary.FileWriter(f'{self.ckpt_dir}/train')
        if listeners:
            for l in listeners:
                l.begin(self.ckpt_dir, self.inputs, self.labels, self.is_training, self.is_fixed_arc, self.fixed_arc,
                        self.child_logits, ckpt_steps)
        summ_writer.add_graph(create_op_graph(self.graph_def))
        ctrl_global_step = self.controller.train_step
        ctrl_summ_writer = tf.summary.FileWriter(f'{self.ckpt_dir}/controller')
        ctrl_summ_sample_entropy = tf.summary.scalar('controller/sample_entropy', self.controller.sample_entropy)
        ctrl_summ_baseline = tf.summary.scalar('controller/baseline', self.controller.baseline)
        ctrl_summ_skip_rate = tf.summary.scalar('controller/skip_rate', self.controller.skip_rate)
        ctrl_summ_nofd_op = tf.summary.merge([ctrl_summ_sample_entropy, ctrl_summ_baseline, ctrl_summ_skip_rate])

        ctrl_summ_loss = tf.summary.scalar('controller/loss', self.controller_loss)
        ctrl_summ_reward = tf.summary.scalar('controller/reward', self.controller.raw_reward)
        ctrl_summ_op = tf.summary.merge([ctrl_summ_loss, ctrl_summ_reward])

        if sess_config_proto is None:
            sess_config_proto = tf.ConfigProto(allow_soft_placement=True)
            sess_config_proto.gpu_options.allow_growth = True
        else:
            sess_config_proto = sess_config_proto
        with tf.Session(config=sess_config_proto) as sess:
            sess.run(create_init_op())
            if not from_scratch:
                load_ckpt(sess, model_dir=self.ckpt_dir)
            child_global_step_val = sess.run(child_global_step)
            id_ckpt = child_global_step_val // ckpt_steps
            for _ in range(num_ckpts):
                progress_desc = f'Checkpoint {id_ckpt+1}'
                for _ in self.progress_range(ckpt_steps, desc=progress_desc):
                    xb, yb = sess.run(data_gntr)
                    fd = {self.inputs: xb, self.labels: yb, self.is_training: True,
                          self.is_fixed_arc: False, self.fixed_arc: [0]}
                    if child_global_step_val == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=child_global_step_val)
                        sess.run(reset_ops)
                    summ_loss, _ = sess.run([summ_loss_op, self.train_op])
                    child_global_step_val = sess.run(child_global_step)
                    if child_global_step_val % summ_steps == 0:
                        sess.run(update_ops, feed_dict=fd)
                        summ = sess.run(summ_op, feed_dict=fd)
                        summ_writer.add_summary(summ, global_step=child_global_step_val)
                        summ_writer.add_summary(summ_loss, global_step=child_global_step_val)
                        sess.run(reset_ops)
                id_ckpt = child_global_step_val // ckpt_steps
                summ_writer.flush()

                if train_controller:
                    if id_ckpt > num_ckpts_training_controller:
                        ctr_progress_desc = f'Checkpoint {id_ckpt} Controller'
                        ctrl_global_step_val = sess.run(ctrl_global_step)
                        for _ in self.progress_range(controller_steps, desc=ctr_progress_desc):
                            if ctrl_global_step_val == 0:
                                ctrl_summ_nofd, sampled_arc = sess.run([ctrl_summ_nofd_op, self.controller.sample_arc])
                                ctrl_summ_writer.add_summary(ctrl_summ_nofd, global_step=ctrl_global_step_val)
                                ctrl_summ = sess.run(ctrl_summ_op, feed_dict={self.fixed_arc: sampled_arc})
                                ctrl_summ_writer.add_summary(ctrl_summ, global_step=ctrl_global_step_val)

                            sampled_arc = sess.run(self.controller.sample_arc)
                            sess.run(self.controller_train_op, feed_dict={self.fixed_arc: sampled_arc})
                            ctrl_global_step_val = sess.run(ctrl_global_step)
                            if ctrl_global_step_val % controller_summ_steps == 0:
                                ctrl_summ_nofd, sampled_arc = sess.run([ctrl_summ_nofd_op, self.controller.sample_arc])
                                ctrl_summ_writer.add_summary(ctrl_summ_nofd, global_step=ctrl_global_step_val)
                                ctrl_summ = sess.run(ctrl_summ_op, feed_dict={self.fixed_arc: sampled_arc})
                                ctrl_summ_writer.add_summary(ctrl_summ, global_step=ctrl_global_step_val)
                        ctrl_summ_writer.flush()
                if id_ckpt > num_ckpts_before_sampling_childs:
                    if listeners:
                        sampled_arcs = []
                        for _ in range(num_sampled_childs):
                            sampled_arc = sess.run(self.controller.sample_arc)
                            sampled_arcs.append(sampled_arc)
                        print(f'{num_sampled_childs} architectures presented:')
                        for i, sampled_arc in enumerate(sampled_arcs):
                            print(f'Arch #{i+1}/{num_sampled_childs}:')
                            start = 0
                            for layer_id in range(self.child.num_layers):
                                layer_skip_count = layer_id
                                end = start + 1 + layer_skip_count
                                print(np.reshape(sampled_arc[start:end], [-1]))
                                start = end
                        for l in listeners:
                            l.run(sampled_arcs, sess, child_global_step_val)
                saver.save(sess, f'{self.ckpt_dir}/model.ckpt', global_step=child_global_step_val,
                           write_meta_graph=False)
            if listeners:
                for l in listeners:
                    l.end()
            summ_writer.close()
            ctrl_summ_writer.close()

    def eval(self, metric_opdefs, gnte, provided_arc=None):
        data_gn, steps_per_epoch = gnte
        metric_ops, update_ops, reset_ops, _, _ = list(zip(*metric_opdefs))
        metric_names = [get_metric_name(m) for m in metric_ops]
        logits_lst = []
        if not self.model_loaded:
            raise RuntimeError('Load model first!')
        sess = self.sess
        if provided_arc is None:
            sampled_arc = sess.run(self.controller.sample_arc)
        else:
            sampled_arc = provided_arc
        sess.run(reset_ops)
        progress_desc = 'Test evaluation'
        for _ in self.progress_range(steps_per_epoch, desc=progress_desc):
            xb, yb = sess.run(data_gn)
            fd = {self.inputs: xb, self.labels: yb, self.is_training: False, self.is_fixed_arc: True,
                  self.fixed_arc: sampled_arc}
            logits_val, _ = sess.run([self.child_logits, update_ops], feed_dict=fd)
            logits_lst.append(logits_val)
        logits = np.concatenate(logits_lst, axis=0)
        metrics = sess.run(metric_ops)
        num_child_layers = self.child.num_layers
        print('Sampled architecture:')
        start = 0
        for layer_id in range(num_child_layers):
            layer_skip_count = layer_id
            end = start + 1 + layer_skip_count
            print(np.reshape(sampled_arc[start:end], [-1]))
            start = end
        for name, metric in zip(metric_names, metrics):
            print(f'{name}: {metric}')
        return logits, metrics
