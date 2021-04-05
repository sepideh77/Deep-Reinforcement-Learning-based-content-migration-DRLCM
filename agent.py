import numpy as np
import tensorflow as tf




def vector_embedding(inputBatch):


    state_size_sequence = inputBatch.Length_plc_chain
    state_size_embeddings = inputBatch.Length_plc_chain

    state = np.zeros((inputBatch.batchSize, state_size_sequence, state_size_embeddings), dtype='int32')

    for batch in range(inputBatch.batchSize):
        for i in range(inputBatch.serviceLength[batch]):
            embedding = inputBatch.state[batch][i]
            state[batch][i][embedding] = 1

    return state

class DynamicMultiRNN(object):

    def __init__(self, action_size, batch_size, input_, input_len_, num_activations, num_layers):

        self.action_size = action_size
        self.batch_size = batch_size
        self.num_activations = num_activations
        self.num_layers = num_layers

        self.positions = []
        self.outputs = []

        self.input_ = input_
        self.input_len_ = input_len_


        initializer = tf.contrib.layers.xavier_initializer()
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(self.num_activations, state_is_tuple=True) for _ in range(self.num_layers)], state_is_tuple=True)

        c_initial_states = []
        h_initial_states = []


        for i in range(self.num_layers):
            first_state = tf.get_variable("var{}".format(i), [1, self.num_activations], initializer=initializer)
            c_initial_state = tf.tile(first_state, [self.batch_size, 1])
            h_initial_state = tf.tile(first_state, [self.batch_size, 1])

            c_initial_states.append(c_initial_state)
            h_initial_states.append(h_initial_state)

        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(c_initial_states[idx], h_initial_states[idx])
             for idx in range(self.num_layers)]
        )

        states_series, current_state = tf.nn.dynamic_rnn(cells, input_, initial_state=rnn_tuple_state, sequence_length=input_len_)
        states_series = tf.Print(states_series, ["states_series", states_series, tf.shape(states_series)], summarize=10)

        self.outputs = tf.layers.dense(states_series, self.action_size, activation=tf.nn.softmax)       # [Batch, seq_length, action_size]
        self.outputs = tf.Print(self.outputs, ["outputs", self.outputs, tf.shape(self.outputs)],summarize=10)


        prob = tf.contrib.distributions.Categorical(probs=self.outputs)

        self.positions = prob.sample()        # [Batch, seq_length]
        self.positions = tf.cast(self.positions, tf.int32)
        self.positions = tf.Print(self.positions, ["position", self.positions, tf.shape(self.positions)], summarize=10)


class Agent:

    def __init__(self, state_size_embeddings, state_maxServiceLength, action_size, batch_size, learning_rate, hidden_dim,  num_stacks, num_Total_cache_nodes, num_non_safety_contents):

        self.num_Total_cache_nodes = num_Total_cache_nodes
        self.num_non_safety_contents = num_non_safety_contents


        self.learning_rate = learning_rate
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_size_embeddings = state_size_embeddings
        self.state_maxServiceLength = state_maxServiceLength
        self.hidden_dim = hidden_dim
        self.num_stacks = num_stacks


        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.state_maxServiceLength, self.state_size_embeddings], name="input")
        self.input_len_ = tf.placeholder(tf.float32, [self.batch_size], name="input_len")

        self._build_model()
        self._build_optimization()

        self.merged = tf.summary.merge_all()

    def _build_model(self):

        with tf.variable_scope('multi_lstm'):
            self.ptr = DynamicMultiRNN(self.action_size, self.batch_size, self.input_, self.input_len_, self.hidden_dim, self.num_stacks)

    def _build_optimization(self):

        with tf.name_scope('reinforce'):

            self.reward_holder = tf.placeholder(tf.float32, [self.batch_size], name="reward_holder")
            self.positions_holder = tf.placeholder(tf.float32, [self.batch_size, self.state_maxServiceLength], name="positions_holder")


            opt = tf.train.GradientDescentOptimizar(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99, epsilon=0.0001)


            probs = tf.contrib.distributions.Categorical(probs=self.ptr.outputs)
            log_softmax = probs.log_prob(self.positions_holder)         # [Batch, seq_length]
            log_softmax = tf.Print(log_softmax, ["log_softmax", log_softmax, tf.shape(log_softmax)])


            log_softmax_mean = tf.reduce_sum(log_softmax,1)                  # [Batch]
            log_softmax_mean = tf.Print(log_softmax_mean, ["log_softmax_mean",log_softmax_mean, tf.shape(log_softmax_mean)])
            variable_summaries('log_softmax_mean', log_softmax_mean, with_max_min=True)

            reward = tf.divide(1000.0, self.reward_holder, name="div")      # [Batch]


            reward = tf.stop_gradient(reward)

            # Compute Loss
            loss = tf.reduce_mean(reward * log_softmax_mean, 0)     # Scalar

            tf.summary.scalar('loss', loss)

            # Minimize step
            gvs = opt.compute_gradients(loss)

            #Clipping
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip

            self.train_step = opt.apply_gradients(capped_gvs)


if __name__ == "__main__":

    pass
