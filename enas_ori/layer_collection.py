import tensorflow as tf
from tensorflow.contrib import slim
import sonnet as snt
import collections

__all__ = ['LayerSetup', 'LayerFunc', 'LayerCollection', 'SimpleStem', 'example_setups']

LayerSetup = collections.namedtuple('LayerSetup',
                                    ('layer_func', 'layer_kwargs'))


class LayerFunc:
    def __init__(self, layer_setup: LayerSetup):
        self.layer_setup = layer_setup

    def __call__(self, inputs, is_training, **kwargs):
        for k in kwargs:
            self.layer_setup.layer_kwargs[k] = kwargs[k]
        return self.layer_setup.layer_func(inputs, is_training, **self.layer_setup.layer_kwargs)


class FuncWrapper:
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x


class LayerCollection(snt.AbstractModule):
    def __init__(self, layer_setups, out_filters=None, use_default_layer_func=True, name='layercol'):
        super(LayerCollection, self).__init__(name=name)
        self.layer_setups = layer_setups
        self.out_filters = out_filters
        if not use_default_layer_func:
            self.out_filters = None

    @classmethod
    def conv_layer(cls, inputs, is_training, out_filters, filter_size, channel_multiplier, separable=False):
        with tf.variable_scope('inp_conv_1'):
            net = snt.Conv2D(out_filters, kernel_shape=1)(inputs)
            net = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(net, is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope(f'out_conv_{out_filters}'):
            if separable:
                net = snt.SeparableConv2D(out_filters, channel_multiplier, filter_size)(net)
            else:
                net = snt.Conv2D(out_filters, kernel_shape=filter_size)(net)
            net = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(net, is_training=is_training)
        return net

    @classmethod
    def pool_layer(cls, inputs, is_training, out_filters, kernel_size, avg_or_max):
        with tf.variable_scope('inp_conv_1'):
            net = snt.Conv2D(out_filters, kernel_shape=1)(inputs)
            net = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(net, is_training=is_training)
            net = tf.nn.relu(net)

        with tf.variable_scope('pool'):
            if avg_or_max == 'avg':
                net = slim.avg_pool2d(net, kernel_size=kernel_size, stride=[1, 1], padding='SAME')
            elif avg_or_max == 'max':
                net = slim.max_pool2d(net, kernel_size=kernel_size, stride=[1, 1], padding='SAME')
            else:
                raise ValueError(f'Unknown pool {avg_or_max}')
            return net

    @classmethod
    def factorized_reduction(cls, inputs, is_training, out_filters, stride):
        assert out_filters % 2 == 0, ('Need even number of filters when using this factorized reduction')
        if stride == 1:
            with tf.variable_scope('path_conv'):
                net = snt.Conv2D(out_filters, kernel_shape=1)(inputs)
                net = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(net, is_training=is_training)
                return net

        # Skip path 1
        path1 = slim.avg_pool2d(inputs, kernel_size=1, stride=stride, padding='VALID')
        with tf.variable_scope('path1_conv'):
            path1 = snt.Conv2D(out_filters // 2, kernel_shape=1)(path1)

        # Skip path 2
        pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
        path2 = tf.pad(inputs, pad_arr)[:, 1:, 1:, :]
        path2 = slim.avg_pool2d(path2, kernel_size=1, stride=stride, padding='VALID')
        with tf.variable_scope('path2_conv'):
            path2 = snt.Conv2D(out_filters // 2, kernel_shape=1)(path2)

        final_path = tf.concat([path1, path2], axis=-1)
        final_path = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(final_path, is_training=is_training)
        return final_path

    def _build(self, inputs, is_training, layer_type_id):
        layer_types = {}
        for i, layer_setup in enumerate(self.layer_setups):
            with tf.variable_scope(f'layer_type_{i}'):
                if self.out_filters:
                    layer = LayerFunc(layer_setup)(inputs,
                                                   is_training,
                                                   out_filters=self.out_filters)
                else:
                    layer = LayerFunc(layer_setup)(inputs, is_training)
                layer_types[tf.equal(layer_type_id, i)] = FuncWrapper(layer)
        out = tf.case(layer_types, default=None, exclusive=True)
        return out


class SimpleStem(snt.AbstractModule):
    def __init__(self, out_filters, stem_kernel_shape, name='child_stem'):
        super(SimpleStem, self).__init__(name=name)
        self.out_filters = out_filters
        self.stem_kernel_shape = stem_kernel_shape

    def _build(self, inputs, is_training):
        net = snt.Conv2D(self.out_filters, kernel_shape=self.stem_kernel_shape)(inputs)
        net = snt.BatchNormV2(scale=True, decay_rate=0.9, eps=1e-5)(net, is_training=is_training)
        return net


example_setups = [
    LayerSetup(layer_func=LayerCollection.conv_layer,
               layer_kwargs={'filter_size': [3, 3],
                             'channel_multiplier': 1,
                             'separable': False}),
    LayerSetup(layer_func=LayerCollection.conv_layer,
               layer_kwargs={'filter_size': [3, 3],
                             'channel_multiplier': 1,
                             'separable': True}),
    LayerSetup(layer_func=LayerCollection.conv_layer,
               layer_kwargs={'filter_size': [5, 5],
                             'channel_multiplier': 1,
                             'separable': False}),
    LayerSetup(layer_func=LayerCollection.conv_layer,
               layer_kwargs={'filter_size': [5, 5],
                             'channel_multiplier': 1,
                             'separable': True}),
    LayerSetup(layer_func=LayerCollection.pool_layer,
               layer_kwargs={'kernel_size': [3, 3],
                             'avg_or_max': 'avg'}),
    LayerSetup(layer_func=LayerCollection.pool_layer,
               layer_kwargs={'kernel_size': [3, 3],
                             'avg_or_max': 'max'})
]
