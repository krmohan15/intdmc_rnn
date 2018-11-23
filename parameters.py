import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : 'C:/Users/FREEDMAN_LAB/Documents/RNNs/intdmc_rnn-master/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : False,

    # Network configuration
    'synapse_config'        : 'full',      # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,

    # Network shape
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 10,
    'num_rule_tuned'        : 12,
    'n_hidden'              : 100,
    'n_output'              : 4,

    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 2e-2,
    'membrane_time_constant': 100,
    'connection_prob'       : 1,         # Usually 1
    'test_cost_multiplier'  : 1.,
    'rule_cue_multiplier'   : 1.,
    'balance_EI'            : False,
    'weight_multiplier'     : 1.,

    # Variance values
    'clip_max_grad_val'     : 0.1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.1,
    'noise_rnn_sd'          : 0.5,

    # Tuning function data
    'num_motion_dirs'       : 10,
    'tuning_height'         : 5,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_regularization'  : 'L2', # 'L1' or 'L2'
    'spike_cost'            : 2e-2,
    'weight_cost'           : 0.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 256,
    'num_iterations'        : 2000,
    'iters_between_outputs' : 10,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 0,
    'fix_time'              : 500,
    'sample_time'           : 650,
    'delay_time'            : 1000,
    'test_time'             : 650,
    'variable_delay_max'    : 300,
    'mask_duration'         : 50,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 2, # this will be two for the DMS+DMRS task

    # Save paths
    'save_fn'               : 'model_results.pkl',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 100,
    'simulation_reps'       : 100,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : False,
    'analyze_tuning'        : True,
    'decode_stability'      : False,
    'save_trial_data'       : False,

}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    #print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    par['num_rules'] = 1
    par['num_receptive_fields'] = 1
    #par['num_rule_tuned'] = 0
    par['ABBA_delay' ] = 0
    par['rule_onset_time'] = [par['dead_time']]
    par['rule_offset_time'] = [par['dead_time']]

    if par['trial_type'] == 'DMS':
        par['rotation_match'] = 0

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 6
        #par['num_rule_tuned'] = 2
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True
        par['num_motion_tuned'] = 24*2
        par['rule_onset_time'] = []
        par['rule_offset_time'] = []
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time']/2)
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + par['delay_time'] + par['test_time'])
        par['rule_onset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 3*par['delay_time']/2 + par['test_time'])
        par['rule_offset_time'].append(par['dead_time'] + par['fix_time'] + par['sample_time'] + 2*par['delay_time'] + 2*par['test_time'])


    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['ABBA_delay'] = 400
        par['delay_time'] = 6*par['ABBA_delay']
        par['repeat_pct'] = 0
        par['analyze_test'] = False
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 6
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 500]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + 750]
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = [par['dead_time']]
            par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']]

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]

    elif par['trial_type'] == 'DMS+DMRS+DMC':
        par['num_rules'] = 3
        par['num_rule_tuned'] = 18
        par['rotation_match'] = [0, 90, 0]
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]

    elif par['trial_type'] == 'location_DMS':
        par['num_receptive_fields'] = 3
        par['rotation_match'] = 0
        par['num_motion_tuned'] = 24*3

    elif par['trial_type'] == 'distractor':
        # this task will not use the create_tuning_functions in stimulus.py
        # instead, it will used a simplified neural input
        par['n_output'] = par['num_motion_dirs'] + 1
        par['sample_time'] = 300
        par['distractor_time'] = 300
        par['delay_time'] = 800
        par['test_time'] = 500
        par['num_fix_tuned'] = 4
        par['simulation_reps'] = 0
        par['analyze_tuning'] = False
        par['num_receptive_fields'] = 1

    elif par['trial_type'] == 'DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['sample_time'] = 650
        par['delay_time'] = 1000
        par['test_time'] = 650
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]

    elif par['trial_type'] == 'OIC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]
        par['sample_time'] = 500
        par['delay_time'] = 0
        par['test_time'] = 300
        par['pad_time'] =1500

    elif par['trial_type'] == 'OICDelay':
        par['num_rules'] = 1
        par['num_rule_tuned'] = 12
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]
        par['sample_time'] = 500
        par['delay_time'] = 1000
        par['test_time'] = 300
        par['pad_time'] =1500-par['delay_time']

    elif par['trial_type'] == 'OICDMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['sample_time'] = 650
        par['delay_time'] = 1000
        par['test_time'] = 650
        par['rule_onset_time'] = [par['dead_time']]
        par['rule_offset_time'] = [par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']]


    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    elif par['trial_type'] == 'OIC' or par['trial_type']=='OICDelay':
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']+par['pad_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    par['dead_time_rng'] = range(par['dead_time']//par['dt'])
    par['sample_time_rng'] = range((par['dead_time']+par['fix_time'])//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt'])
    par['rule_time_rng'] = [range(int(par['rule_onset_time'][n]/par['dt']), int(par['rule_offset_time'][n]/par['dt'])) for n in range(len(par['rule_onset_time']))]


    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['drop_mask'] = np.ones((par['n_hidden'],par['n_hidden']), dtype=np.float32)
    par['ind_inh'] = np.where(par['EI_list']==-1)[0]

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']



    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################
    par['h_init'] = 0.1*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)


    # Initialize input weights
    par['w_in0'] = initialize([par['n_hidden'], par['n_input']], par['connection_prob']/par['num_receptive_fields'], shape=0.2, scale=1.)

    # Initialize starting recurrent weights
    # If excitatory/inhibitory neurons desired, initializes with random matrix with
    #   zeroes on the diagonal
    # If not, initializes with a diagonal matrix
    if par['EI']:
        par['w_rnn0'] = par['weight_multiplier']*initialize([par['n_hidden'], par['n_hidden']], par['connection_prob'])
        if par['balance_EI']:
            par['w_rnn0'][:, par['ind_inh']] = par['weight_multiplier']*initialize([par['n_hidden'], par['num_inh_units']], par['connection_prob'], shape=0.2, scale=1.)
            par['w_rnn0'][par['ind_inh'], :] = par['weight_multiplier']*initialize([ par['num_inh_units'], par['n_hidden']], par['connection_prob'], shape=0.2, scale=1.)

        for i in range(par['n_hidden']):
            par['w_rnn0'][i,i] = 0
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
    else:
        par['w_rnn0'] = 0.54*np.eye(par['n_hidden'])
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    par['b_rnn0'] = np.zeros((par['n_hidden'], 1), dtype=np.float32)

    # Effective synaptic weights are stronger when no short-term synaptic plasticity
    # is used, so the strength of the recurrent weights is reduced to compensate
    if par['synapse_config'] is None:
        par['w_rnn0'] = par['w_rnn0']/(spectral_radius(par['w_rnn0']))

    # Initialize output weights and biases
    par['w_out0'] = par['weight_multiplier']*initialize([par['n_output'], par['n_hidden']], par['connection_prob'])
    par['b_out0'] = np.zeros((par['n_output'], 1), dtype=np.float32)
    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)

    if par['EI']:
        par['w_out0'][:, par['ind_inh']] = 0
        par['w_out_mask'][:, par['ind_inh']] = 0

    par['w_in_mask'] = np.ones_like(par['w_in0'])


    # for the location_DMS task, inputs from the 3 receptive fields project onto non-overlapping
    # units in the RNN. This tries to replicates what liekly happesn in areas MST, which are retinotopic
    if par['trial_type'] == 'location_DMS':
        par['w_in_mask'] *= 0
        target_ind = [range(0, par['n_hidden'],3), range(1, par['n_hidden'],3), range(2, par['n_hidden'],3)]
        for n in range(par['n_input']):
            u = int(n//(par['n_input']/3))
            par['w_in_mask'][target_ind[u], n] = 1
        par['w_in0'] = par['w_in0']*par['w_in_mask']

    # initialize synaptic values
    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['dynamic_synapse'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    if par['synapse_config'] == 'full' or par['synapse_config'] == 'std_stf':
        par['synapse_type'] = ['facilitating' if i%2==0 else 'depressing' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'excitatory_facilitating':
        par['synapse_type'] = ['facilitating' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'facilitating':
        par['synapse_type'] = ['facilitating' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'inhibitory_facilitating':
        par['synapse_type'] = ['facilitating' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'excitatory_half_facilitating':
        par['synapse_type'] = ['facilitating' if par['EI_list'][i]==1 and i%2==0 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'excitatory_depressing':
        par['synapse_type'] = ['depressing' if par['EI_list'][i]==1 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'excitatory_depressing_inhibitory_facilitating':
        par['synapse_type'] = ['depressing' if par['EI_list'][i]==1 else 'facilitating' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'inhibitory_depressing':
        par['synapse_type'] = ['depressing' if par['EI_list'][i]==-1 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'excitatory_half_depressing':
        par['synapse_type'] = ['depressing' if par['EI_list'][i]==1  and i%2==0 else 'static' for i in range(par['n_hidden'])]
    elif par['synapse_config'] == 'depressing':
        par['synapse_type'] = ['depressing' for i in range(par['n_hidden'])]

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 'facilitating':
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 'depressing':
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 'static':
            par['dynamic_synapse'][i,0] = 0
            par['U'][i,0] = 1.
            par['syn_x_init'][i,0] = 1.
            par['syn_u_init'][i,0] = 1.

def initialize(dims, connection_prob, shape=0.1, scale=1.0 ):
    w = np.random.gamma(shape, scale, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)

    return np.float32(w)


def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
