"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import analysis
import pickle
import time
from parameters import *
import os, sys

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load the initial hidden state activity to be used at the start of each trial

        #self.hidden_init = tf.constant(par['h_init'])

        with tf.variable_scope('initial_activity'):
            self.hidden_init = tf.get_variable('hidden_init', initializer = par['h_init'], trainable=True)

        #self.hidden_init = tf.random_uniform([par['n_hidden'], 1], 0, 0.5)


        # Load the initial synaptic depression and facilitation to be used at the start of each trial
        self.synapse_x_init = tf.constant(par['syn_x_init'])
        self.synapse_u_init = tf.constant(par['syn_u_init'])

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        """
        Run the reccurent network
        History of hidden state activity stored in self.hidden_state_hist
        """



        self.rnn_cell_loop(self.input_data, self.hidden_init, self.synapse_x_init, self.synapse_u_init)

        with tf.variable_scope('output'):
            W_out = tf.get_variable('W_out', initializer = par['w_out0'], trainable=True)
            b_out = tf.get_variable('b_out', initializer = par['b_out0'], trainable=True)

        """
        Network output
        Only use excitatory projections from the RNN to the output layer
        """
        self.y_hat = [tf.matmul(tf.nn.relu(W_out),h)+b_out for h in self.hidden_state_hist]


    def rnn_cell_loop(self, x_unstacked, h, syn_x, syn_u):

        """
        Initialize weights and biases
        """
        with tf.variable_scope('rnn_cell'):
            W_in = tf.get_variable('W_in', initializer = par['w_in0'], trainable=True)
            W_rnn = tf.get_variable('W_rnn', initializer = par['w_rnn0'], trainable=True)
            b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn0'], trainable=True)
        self.W_ei = tf.constant(par['EI_matrix'])

        self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []

        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input in x_unstacked:
            h, syn_x, syn_u = self.rnn_cell(rnn_input, h, syn_x, syn_u)
            self.hidden_state_hist.append(h)
            self.syn_x_hist.append(syn_x)
            self.syn_u_hist.append(syn_u)


    def rnn_cell(self, rnn_input, h, syn_x, syn_u):

        """
        Main computation of the recurrent network
        """
        with tf.variable_scope('rnn_cell', reuse=True):
            W_in = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')
            b_rnn = tf.get_variable('b_rnn')

        if par['EI']:
            # ensure excitatory neurons only have postive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            W_rnn_effective = tf.matmul(tf.nn.relu(W_rnn), self.W_ei)
        else:
            W_rnn_effective = W_rnn_drop

        """
        Update the synaptic plasticity paramaters
        """
        if par['synapse_config'] is not None:
            # implement both synaptic short term facilitation and depression
            syn_x += (par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h)*par['dynamic_synapse']
            syn_u += (par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h)*par['dynamic_synapse']
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h

        else:
            # no synaptic plasticity
            h_post = h

        """
        Update the hidden state
        Only use excitatory projections from input layer to RNN
        All input and RNN activity will be non-negative
        """
        h = tf.nn.relu(h*(1-par['alpha_neuron'])
                       + par['alpha_neuron']*(tf.matmul(tf.nn.relu(W_in), tf.nn.relu(rnn_input))
                       + tf.matmul(W_rnn_effective, h_post) + b_rnn)
                       + tf.random_normal([par['n_hidden'], par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

        return h, syn_x, syn_u


    def optimize(self):

        """
        Calculate the loss functions and optimize the weights

        perf_loss = [mask*tf.reduce_mean(tf.square(y_hat-desired_output),axis=0)
                     for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]
        """
        """
        cross_entropy
        """
        self.perf_loss = tf.reduce_mean(tf.stack([mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                for (y_hat, desired_output, mask) in zip(self.y_hat, self.target_data, self.mask)]))


        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        #spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.hidden_state_hist]
        if par['spike_regularization'] == 'L1':
            self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([tf.reduce_mean(h) for h in self.hidden_state_hist]))
        elif par['spike_regularization'] == 'L2':
            self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([tf.reduce_mean(tf.square(h)) for h in self.hidden_state_hist]))
        else:
            error('Unrecognized spike regularization')


        with tf.variable_scope('rnn_cell', reuse = True):
            W_rnn = tf.get_variable('W_rnn')

        self.weight_loss = par['weight_cost']*tf.reduce_mean(tf.square(tf.nn.relu(W_rnn)))

        self.loss = self.perf_loss + self.spike_loss + self.weight_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        grads_and_vars = opt.compute_gradients(self.loss)

        """
        Apply any applicable weights masks to the gradient and clip
        """
        capped_gvs = []
        for grad, var in grads_and_vars:
            if var.name == "rnn_cell/W_rnn:0":
                grad *= par['w_rnn_mask']
                print('Applied weight mask to w_rnn.')
            elif var.name == "output/W_out:0":
                grad *= par['w_out_mask']
                print('Applied weight mask to w_out.')
            elif var.name == "rnn_cell/W_in:0":
                grad *= par['w_in_mask']
                print('Applied weight mask to w_in.')
            if not str(type(grad)) == "<class 'NoneType'>":
                capped_gvs.append((tf.clip_by_norm(grad, par['clip_max_grad_val']), var))

        self.train_op = opt.apply_gradients(capped_gvs)


def main(gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Print key parameters
    """
    print_important_params()

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.Stimulus()

    n_input, n_hidden, n_output = par['shape']
    N = par['batch_train_size'] # trials per iteration, calculate gradients after batch_train_size

    """
    Define all placeholder
    """
    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[n_input, par['num_time_steps'], par['batch_train_size']])  # input data
    y = tf.placeholder(tf.float32, shape=[n_output, par['num_time_steps'], par['batch_train_size']]) # target data

    config = tf.ConfigProto()

    # enter "config=tf.ConfigProto(log_device_placement=True)" inside Session to check whether CPU/GPU in use
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, mask)

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], 'weight_loss': [], 'trial': []}

        task_list=['DMC','OIC']
        for task in task_list:

            for i in range(par['num_iterations']):

                # generate batch of batch_train_size
                save_fn = task + 'seq' + '.pkl'
                updates = {'trial_type': task, 'save_fn': save_fn}
                update_parameters(updates)
                trial_info = stim.generate_trial(set_rule = None)

                """
                Run the model
                """
                _, loss, perf_loss, spike_loss, weight_loss, y_hat, state_hist, syn_x_hist, syn_u_hist = \
                    sess.run([model.train_op, model.loss, model.perf_loss, model.spike_loss, model.weight_loss, model.y_hat, \
                    model.hidden_state_hist, model.syn_x_hist, model.syn_u_hist], {x: trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask']})

                accuracy, accuracy_dmc, accuracy_oic = analysis.get_perf_oicdmc(trial_info['desired_output'], y_hat, trial_info['train_mask'],trial_info['task'])

                updates = {'trial_type': task, 'save_fn': save_fn}
                update_parameters(updates)
                model_performance = append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, (i+1)*N)

                """
                Save the network model and output model performance to screen
                """
                if i%par['iters_between_outputs']==0:
                    print_results(i, N, perf_loss, spike_loss, weight_loss, state_hist, accuracy,accuracy_dmc,accuracy_oic)


            """
            Save model, analyze the network model and save the results - KM added 11/19/18 - remove once you can access data from analysis
            """
            h_stacked = np.stack(state_hist, axis=1)
            trial_time = np.arange(0,h_stacked.shape[1]*par['dt'], par['dt'])
            weights = eval_weights()
            results = {}
            results = {
                'model_performance': model_performance,
                'parameters': par,
                'weights': weights,
                'trial_time': trial_time}
            results['h'] = state_hist
            results['y_hat'] = np.array(y_hat)
            results['trial_info'] = trial_info
            results['syn_x'] = syn_x_hist
            results['syn_u'] = syn_u_hist
            save_fn = par['save_dir'] + par['save_fn']
            pickle.dump(results, open(save_fn, 'wb'))
            #save_results(model_performance)



def save_results(model_performance):

    weights = eval_weights()
    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    fn = par['save_dir'] + par['save_fn']
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ',fn)


def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, trial_num):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['trial'].append(trial_num)

    return model_performance

def eval_weights():

    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    with tf.variable_scope('initial_activity', reuse=True):
        hidden_init = tf.get_variable('hidden_init')


    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval(),
        'hidden_init': hidden_init.eval()}

    return weights

def print_results(iter_num, trials_per_iter, perf_loss, spike_loss, weight_loss, state_hist, accuracy,accuracy_dmc,accuracy_oic):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(np.mean(state_hist)) +
      ' | DMC Accuracy {:0.4f}'.format(accuracy_dmc) + ' | OIC Accuracy {:0.4f}'.format(accuracy_oic))

def print_important_params():

    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd','spike_cost',\
        'spike_regularization', 'weight_cost','test_cost_multiplier', 'trial_type','balance_EI', 'dt',\
        'delay_time','weight_multiplier', 'connection_prob','synapse_config','tau_slow']
    for k in important_params:
        print(k, ': ', par[k])
