import tensorflow as tf
from tensorflow.contrib import slim
import sonnet as snt
from .layer_collection import LayerCollection

__all__ = ['MacroChildNetwork']


def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[1, 2], keepdims=True)


class MacroChildNetwork(snt.AbstractModule):
    def __init__(self,
                 layer_setups,
                 use_default_layer_func,
                 stem_network,
                 num_classes,
                 num_layers,
                 out_filters,
                 reduction_stride,
                 keep_prob,
                 name='child'):
        super(MacroChildNetwork, self).__init__(name=name)

        self.layer_setups = layer_setups
        self.use_default_layer_func = use_default_layer_func
        self.stem_network = stem_network
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.out_filters = out_filters
        self.reduction_stride = reduction_stride
        self.keep_prob = keep_prob
        self.name = name

        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

        self.controller = None

    def enas_layer(self, prev_layers, layer_id, start_idx, sample_arc, is_training):
        layer = LayerCollection(self.layer_setups, self.out_filters, self.use_default_layer_func, name='layercol')
        inputs = prev_layers[-1]
        layer_type_id = sample_arc[start_idx]
        out = layer(inputs, is_training, layer_type_id)
        if layer_id > 0:
            skip_start = start_idx + 1
            layer_skip_count = layer_id
            skip = sample_arc[skip_start: skip_start + layer_skip_count]
            with tf.variable_scope('skip'):
                res_layers = []
                for i in range(layer_skip_count):
                    res_layers.append(tf.cond(tf.equal(skip[i], 1), lambda: prev_layers[i + 1],
                                              lambda: tf.zeros_like(prev_layers[i + 1])))
                res_layers.append(out)
                out = tf.add_n(res_layers)
                out = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(out, is_training=is_training)
        return out

    def connect_controller(self, controller):
        self.controller = controller
        self.controller.child_network = self
        with tf.variable_scope(self.name):
            self.train_step = tf.get_variable(initializer=tf.zeros_initializer, shape=[], dtype=tf.int32,
                                              trainable=False,
                                              name='train_step')

    def _build(self, inputs, is_training, is_fixed_arc, fixed_arc):
        assert self.controller is not None, 'controller must be provided'
        sample_arc = tf.cond(is_fixed_arc, lambda: fixed_arc, lambda: self.controller.sample_arc)
        layers = []
        net = self.stem_network(inputs, is_training)
        layers.append(net)
        start_idx = 0
        for layer_id in range(self.num_layers):
            with tf.variable_scope(f'layer_{layer_id}'):
                net = self.enas_layer(layers, layer_id, start_idx, sample_arc, is_training)
                layers.append(net)
                if layer_id in self.pool_layers:
                    with tf.variable_scope(f'pool_at_{layer_id}'):
                        pooled_layers = [layers[0]]
                        for i, layer in enumerate(layers[1:]):
                            with tf.variable_scope(f'from_{i}'):
                                net = LayerCollection.factorized_reduction(layer, is_training,
                                                                           out_filters=self.out_filters,
                                                                           stride=self.reduction_stride)
                                pooled_layers.append(net)
                        layers = pooled_layers
            layer_skip_count = layer_id
            start_idx += 1 + layer_skip_count
        net = global_avg_pool(net)
        net = snt.BatchFlatten()(net)
        net = slim.dropout(net, self.keep_prob, is_training=is_training)
        if self.num_classes != 0:
            net = snt.Linear(self.num_classes, name='fc')(net)
        else:
            net = snt.Linear(1, name='fc')(net)[:, 0]
        return net
