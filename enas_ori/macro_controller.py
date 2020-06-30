import tensorflow as tf
import sonnet as snt

__all__ = ['SamplerNetwork', 'MacroController']


class SamplerNetwork(snt.RNNCore):
    def __init__(self, lstm_size, lstm_num_layers, name='sampler'):
        super(SamplerNetwork, self).__init__(name=name)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers

    def _build(self, inputs, prev_state):
        lstm_cells = [snt.LSTM(self.lstm_size) for _ in range(self.lstm_num_layers)]
        cell = snt.DeepRNN(lstm_cells, skip_connections=False)
        output, next_state = cell(inputs, prev_state)
        return output, next_state

    def zero_state(self, batch_size, dtype=tf.float32):
        h0 = tf.zeros([batch_size, self.lstm_size], dtype=dtype)
        c0 = tf.zeros([batch_size, self.lstm_size], dtype=dtype)
        return [(h0, c0) for _ in range(self.lstm_num_layers)]

    @property
    def state_size(self):
        return tf.TensorShape(self.lstm_size)

    @property
    def output_size(self):
        return tf.TensorShape(self.lstm_size)


class MacroController(snt.AbstractModule):
    def __init__(self,
                 lstm_size,
                 lstm_num_layers,
                 num_layers,
                 num_layer_types,
                 skip_target,
                 skip_weight,
                 entropy_weight,
                 bl_dec,
                 name='controller'):
        super(MacroController, self).__init__(name=name)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.num_layers = num_layers
        self.num_layer_types = num_layer_types
        self.skip_target = skip_target
        self.skip_weight = skip_weight
        self.entropy_weight = entropy_weight
        self.bl_dec = bl_dec
        self.name = name

        self.child_network = None
        self.raw_reward = None
        self.reward = None

    def _create_params(self):
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        g_emb = tf.get_variable('g_emb', [1, self.lstm_size], initializer=initializer)
        with tf.variable_scope('emb'):
            w_emb = tf.get_variable('w', [self.num_layer_types, self.lstm_size])
        with tf.variable_scope('softmax'):
            w_soft = tf.get_variable('w', [self.lstm_size, self.num_layer_types])
        with tf.variable_scope('attention'):
            w_attn_1 = tf.get_variable('w_1', [self.lstm_size, self.lstm_size])
            w_attn_2 = tf.get_variable('w_2', [self.lstm_size, self.lstm_size])
            v_attn = tf.get_variable('v', [self.lstm_size, 1])
        return g_emb, w_emb, w_soft, w_attn_1, w_attn_2, v_attn

    def _build(self):
        arc_seq = []
        entropys = []
        log_probs = []
        skip_counts = []
        skip_penaltys = []

        anchors = []
        anchors_w_1 = []

        g_emb, w_emb, w_soft, w_attn_1, w_attn_2, v_attn = self._create_params()
        self.ws = [g_emb, w_emb, w_soft, w_attn_1, w_attn_2, v_attn]
        inputs = g_emb
        skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target], dtype=tf.float32)
        sampler = SamplerNetwork(self.lstm_size, self.lstm_num_layers)
        prev_state = sampler.zero_state(batch_size=1)
        for layer_i in range(self.num_layers):
            next_h, next_state = sampler(inputs, prev_state)
            prev_state = next_state
            logit = tf.matmul(next_h, w_soft)
            layer_type_id = tf.reshape(tf.to_int32(tf.multinomial(logit, 1)), [-1])
            arc_seq.append(layer_type_id)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=layer_type_id)
            log_probs.append(log_prob)
            entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
            entropys.append(entropy)
            inputs = tf.nn.embedding_lookup(w_emb, layer_type_id)

            next_h, next_state = sampler(inputs, prev_state)
            prev_state = next_state
            if layer_i == 0:
                inputs = g_emb
            elif layer_i > 0:
                layer_skip_count = layer_i
                query = tf.concat(anchors_w_1, axis=0)
                query = tf.tanh(query + tf.matmul(next_h, w_attn_2))
                query = tf.matmul(query, v_attn)
                logit = tf.concat([-query, query], axis=1)
                skip = tf.reshape(tf.to_int32(tf.multinomial(logit, 1)), [layer_skip_count])
                arc_seq.append(skip)

                skip_prob = tf.sigmoid(logit)
                kl = tf.reduce_sum(skip_prob * tf.log(skip_prob / skip_targets))
                skip_penaltys.append(kl)

                log_prob = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=skip),
                                         keepdims=True)
                log_probs.append(log_prob)

                entropy = tf.stop_gradient(tf.reduce_sum(log_prob * tf.exp(-log_prob), keepdims=True))
                entropys.append(entropy)

                skip = tf.reshape(tf.to_float(skip), [1, layer_skip_count])
                skip_counts.append(tf.reduce_sum(skip))
                inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                inputs /= (1.0 + tf.reduce_sum(skip))
            else:
                raise ValueError('layer_i cannot be smaller than 0')
            anchors.append(next_h)
            anchors_w_1.append(tf.matmul(next_h, w_attn_1))

        self.sample_arc = tf.reshape(tf.concat(arc_seq, axis=0), [-1])
        self.sample_entropy = tf.reduce_sum(entropys)
        self.sample_log_prob = tf.reduce_sum(log_probs)
        self.sample_skip_count = tf.reduce_sum(skip_counts)
        self.sample_skip_penaltys = tf.reduce_mean(skip_penaltys)

        all_skip_count = self.num_layers * (self.num_layers - 1) / 2
        self.skip_rate = self.sample_skip_count / all_skip_count
        self.baseline = tf.get_variable(initializer=tf.zeros_initializer, shape=[], dtype=tf.float32, trainable=False,
                                        name='baseline')
        self.train_step = tf.get_variable(initializer=tf.zeros_initializer, shape=[], dtype=tf.int32, trainable=False,
                                          name='train_step')

    def get_reward(self, sampled_arc, reward_input_func, reward_metric_func, reward_batches, one_shot_reward=False):
        assert self.child_network is not None, 'no child no reward!'
        with tf.name_scope('reward'):
            if one_shot_reward:
                (reward_inputs, reward_labels), reward_batches_raw = reward_input_func()
                dtype = reward_labels.dtype
                if reward_batches is None:
                    reward_batches = reward_batches_raw
                while_cond = lambda i, labels, predictions: tf.less(i, reward_batches)

                def while_body(i, labels, predictions):
                    (reward_inputs, reward_labels), _ = reward_input_func()
                    child_logit = self.child_network(reward_inputs, False, tf.constant(True), sampled_arc)
                    labels = tf.concat([labels, reward_labels], axis=0)
                    predictions = tf.concat([predictions, child_logit], axis=0)
                    i = i + 1
                    return i, labels, predictions

                _, labels, predictions = tf.while_loop(while_cond, while_body,
                                                       loop_vars=[tf.constant(0), tf.constant(([]), dtype=dtype),
                                                                  tf.constant(([]), dtype=dtype)],
                                                       shape_invariants=[tf.TensorShape([]), tf.TensorShape([None]),
                                                                         tf.TensorShape([None])],
                                                       parallel_iterations=1, back_prop=False)
                metric_op = reward_metric_func(labels, predictions)
                child_reward_val = tf.identity(metric_op)
                return child_reward_val
            if reward_batches is None:
                _, reward_batches = reward_input_func()
            while_cond = lambda i, _: tf.less(i, reward_batches)

            def while_body(i, _):
                (reward_inputs, reward_labels), _ = reward_input_func()
                child_logit = self.child_network(reward_inputs, False, tf.constant(True), sampled_arc)
                _, update_op, reset_op, _, _ = reward_metric_func(reward_labels, child_logit)
                cond_op = tf.cond(tf.equal(i, 0), lambda: reset_op, lambda: tf.no_op())
                with tf.control_dependencies([cond_op]):
                    val = update_op
                    i = i + 1
                    return i, val

            _, child_reward_val = tf.while_loop(while_cond, while_body,
                                                loop_vars=[tf.constant(0), tf.constant(0.0, dtype=tf.float32)],
                                                shape_invariants=[tf.TensorShape([]), tf.TensorShape([])],
                                                parallel_iterations=1,
                                                back_prop=False)
            return child_reward_val

    def get_loss(self, sampled_arc, reward_input_func, reward_metric_func, reward_batches=None, one_shot_reward=False):
        reward = self.get_reward(sampled_arc, reward_input_func, reward_metric_func, reward_batches, one_shot_reward)
        self.raw_reward = reward
        if self.entropy_weight is not None:
            reward += self.entropy_weight * self.sample_entropy

        baseline_update_op = tf.assign_sub(self.baseline, (1 - self.bl_dec) * (self.baseline - reward))

        with tf.control_dependencies([baseline_update_op]):
            reward = tf.identity(reward)
        reward = tf.stop_gradient(reward)
        loss = self.sample_log_prob * (reward - self.baseline)
        if self.skip_weight is not None:
            loss += self.skip_weight * self.sample_skip_penaltys
        self.reward = reward
        return loss
