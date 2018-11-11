import numpy as np
import matplotlib.pyplot as plt
import itertools
from parameters import *


class Stimulus:

    def __init__(self):

        # generate tuning functions
        if par['trial_type'] == 'OIC':
            self.motion_tuning = self.create_tuning_functions()
        elif par['trial_type'] == 'DMC':
            self.motion_tuning = self.create_tuning_functions()
        elif par['trial_type'] == 'OICDMC':
            self.motion_tuning = self.create_tuning_functions()
        elif par['trial_type'] == 'OICMATCH':
            self.motion_tuning = self.create_tuning_functions()
        else:
            self.motion_tuning, self.fix_tuning, self.rule_tuning, self.target_tuning, self.color_tuning = self.create_tuning_functions()

        self.num_trials = par['batch_train_size']


    def generate_trial(self,trial_type):


        if trial_type in ['OIC']:
            trial_info = self.generate_oic_trial()
        elif trial_type in ['DMC']:
            trial_info = self.generate_dmc_trial()
        elif trial_type in ['OICDMC']:
            trial_info = self.generate_oicdmc_trial()
        elif par['trial_type'] == 'OICMATCH':
            trial_info = self.generate_oicmatch_trial()

        return trial_info

    def generate_oicdmc_trial(self):
        """
        Generate a batch of trials with half the trials from OIC and the other half from DMC
        one-interval categorization task
        One motion stimulus is shown for 500ms,
        then two colored targets are shown in two locations.
        If the stimulus is category 1, then go to the red target
        else if the stimulus is category 2, then go to the green target.

        Delayed match to categorization task
        One motion stimulus is shown for 650ms - sample stimulus,
        it is followed by a delay period - 1000 ms,
        this is followed by a test stimulus.
        If the sample and test stimulus match in category, then
        release lever.
        If the sample and test stimulus do not match in category, then
        hold lever.
        """
        trial_length = par['num_time_steps']
        mask_duration = par['mask_duration']//par['dt']

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], trial_length, self.num_trials),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, self.num_trials),dtype=np.float32),
                      'sample'          :  np.zeros((self.num_trials),dtype=np.int8),
                      'test'            :  -1*np.ones((self.num_trials),dtype=np.float32),
                      'match'           :  np.zeros((self.num_trials),dtype=np.int8),
                      'task'            :  np.zeros((self.num_trials),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], trial_length, self.num_trials))}

        for t in range(self.num_trials):
            tr_type = np.random.randint(2)
            if tr_type == 0: #OIC trial
                updates = {'trial_type': 'OIC'}
                update_parameters(updates)
                eodead = par['dead_time']//par['dt']
                eof = (par['dead_time']+par['fix_time'])//par['dt']
                eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
                eog= (par['dead_time']+par['fix_time']+par['sample_time']+par['go_time'])//par['dt']

                # end of neuron indices
                emt = par['num_motion_tuned']
                eft = par['num_fix_tuned']+par['num_motion_tuned']
                ert = par['num_fix_tuned']+par['num_motion_tuned'] + par['num_rule_tuned']
                nrule=par['num_rule_tuned']//2;

                # set to mask equal to zero during the dead time
                trial_info['train_mask'][:eodead, t] = 0
                # set to mask equal to zero during the test stimulus
                trial_info['train_mask'][eos:eos+mask_duration,t]=0
                #set mask equal to zero at any time after which the trial has "ended".
                trial_info['train_mask'][eog:,t]=0
                # set fixation equal to 1 for all times; will then change
                trial_info['desired_output'][0, :eos, t] = 1 #only for first 500 ms

                #eft+1:esct+nrule esct+1+nrule:ersct
                trial_info['neural_input'][eft+1+nrule:ert,eodead:eos,t] = par['tuning_height']; #rule
                sample_dir = np.random.randint(par['num_motion_dirs'])
                sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
                trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[0][:,sample_dir],(-1,1))

                if sample_cat==0:
                    trial_info['desired_output'][2,eos:,t]=1
                elif sample_cat==1:
                    trial_info['desired_output'][3,eos:,t]=1

                trial_info['sample'][t]=sample_dir
                trial_info['task'][t]=tr_type #OIC

                #plt.imshow(trial_info['desired_output'][:, :, t])
                #plt.colorbar()
                #plt.show()

                #plt.imshow(trial_info['neural_input'][:, :, t])
                #plt.colorbar()
                #plt.show()

                #plt.imshow(trial_info['train_mask'][:,:])
                #plt.colorbar()
                #plt.show()

            elif tr_type==1:
                updates = {'trial_type': 'DMC'}
                update_parameters(updates)
                eodead = par['dead_time']//par['dt']
                eof = (par['dead_time']+par['fix_time'])//par['dt']
                eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
                eod= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
                eot= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt']

                # end of neuron indices
                emt = par['num_motion_tuned']
                eft = par['num_fix_tuned']+par['num_motion_tuned']
                ert = par['num_fix_tuned']+par['num_motion_tuned'] + par['num_rule_tuned']
                nrule=par['num_rule_tuned']//2;

                # set to mask equal to zero during the dead time
                trial_info['train_mask'][:eodead, t] = 0
                # set to mask equal to zero during the test stimulus
                trial_info['train_mask'][eod:eod+mask_duration,t]=0
                # set fixation equal to 1 for all times; will then change
                trial_info['desired_output'][0, :eod, t] = 1 #only for first 500 ms

                trial_info['neural_input'][eft+1:eft+nrule,eodead:eot,t] = par['tuning_height']; #rule
                sample_dir = np.random.randint(par['num_motion_dirs'])
                sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
                match = np.random.randint(2)

                trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[0][:,sample_dir],(-1,1))

                if match==1:
                    dir0 = int(sample_cat*par['num_motion_dirs']//2)
                    dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                    possible_dirs = np.setdiff1d(list(range(dir0, dir1)), sample_dir)
                    test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                else:
                    test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                    test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])

                trial_info['neural_input'][:emt, eod:eot, t] += np.reshape(self.motion_tuning[0][:,test_dir],(-1,1))

                if match==1:
                    trial_info['desired_output'][1,eod:,t]=1
                else:
                    trial_info['desired_output'][0,eod:,t]=1


                trial_info['sample'][t]=sample_dir
                trial_info['match'][t]=match
                trial_info['test'][t]=test_dir
                trial_info['task'][t]=tr_type #DMC

                #plt.imshow(trial_info['desired_output'][:, :, t])
                #plt.colorbar()
                #plt.show()

                #plt.imshow(trial_info['neural_input'][:, :, t])
                #plt.colorbar()
                #plt.show()

                #plt.imshow(trial_info['train_mask'][:,:])
                #plt.colorbar()
                #plt.show()


        return trial_info

    def generate_oic_trial(self):

        """
        Generate a one-interval categorization task
        One motion stimulus is shown for 500ms,
        then two colored targets are shown in two locations.
        If the stimulus is category 1, then go to the red target
        else if the stimulus is category 2, then go to the green target.
        """
        trial_length = par['num_time_steps']
        mask_duration = par['mask_duration']//par['dt']
        eodead = par['dead_time']//par['dt']
        eof = (par['dead_time']+par['fix_time'])//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        eod= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        eog= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['go_time'])//par['dt']
        # end of neuron indices
        emt = par['num_motion_tuned']
        eft = par['num_fix_tuned']+par['num_motion_tuned']
        ert = par['num_fix_tuned']+par['num_motion_tuned'] + par['num_rule_tuned']
        nrule=par['num_rule_tuned']//2;

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], trial_length, self.num_trials),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, self.num_trials),dtype=np.float32),
                      'sample'          :  np.zeros((self.num_trials),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], trial_length, self.num_trials))}
        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0
        # set to mask equal to zero during the test stimulus
        #trial_info['train_mask'][eos:eos+mask_duration,:]=0
        trial_info['train_mask'][eod:eod+mask_duration,:]=0
        #set mask equal to zero at any time after which the trial has "ended".
        trial_info['train_mask'][eog:,:]=0
        # set fixation equal to 1 for all times; will then change
        trial_info['desired_output'][0, :eod, :] = 1 #only for first 500 ms

        for t in range(self.num_trials):#eft+1:esct+nrule esct+1+nrule:ersct
            trial_info['neural_input'][eft+1+nrule:ert,eodead:eog,t] = par['tuning_height']; #rule
            sample_dir = np.random.randint(par['num_motion_dirs'])
            sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
            trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[0][:,sample_dir],(-1,1))

            """
            if sample_cat==0:
                trial_info['desired_output'][1,eos:,t]=1
            elif sample_cat==1:
                trial_info['desired_output'][2,eos:,t]=1
            """
            if sample_cat==0:
                trial_info['desired_output'][1,eod:,t]=1
            elif sample_cat==1:
                trial_info['desired_output'][2,eod:,t]=1

            trial_info['sample'][t]=sample_dir

            #plt.imshow(trial_info['desired_output'][:, :, t])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(trial_info['neural_input'][:, :, t])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(trial_info['train_mask'][:,:])
            #plt.colorbar()
            #plt.show()


        return trial_info

    def generate_dmc_trial(self):

        """
        Generate a delayed match to categorization task
        One motion stimulus is shown for 650ms - sample stimulus,
        it is followed by a delay period - 1000 ms,
        this is followed by a test stimulusself.
        If the sample and test stimulus match in category, then
        release lever.
        If the sample and test stimulus do not match in category, then
        hold lever.
        """
        trial_length = par['num_time_steps']
        mask_duration = par['mask_duration']//par['dt']
        eodead = par['dead_time']//par['dt']
        eof = (par['dead_time']+par['fix_time'])//par['dt']
        eos = (par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        eod= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']
        eot= (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt']

        # end of neuron indices
        emt = par['num_motion_tuned']
        eft = par['num_fix_tuned']+par['num_motion_tuned']
        ert = par['num_fix_tuned']+par['num_motion_tuned'] + par['num_rule_tuned']
        nrule=par['num_rule_tuned']//2;

        trial_info = {'desired_output'  :  np.zeros((par['n_output'], trial_length, self.num_trials),dtype=np.float32),
                      'train_mask'      :  np.ones((trial_length, self.num_trials),dtype=np.float32),
                      'sample'          :  np.zeros((self.num_trials),dtype=np.int8),
                      'test'            :  -1*np.ones((self.num_trials),dtype=np.float32),
                      'match'           :  np.zeros((self.num_trials),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['n_input'], trial_length, self.num_trials))}

        # set to mask equal to zero during the dead time
        trial_info['train_mask'][:eodead, :] = 0
        # set to mask equal to zero during the test stimulus
        trial_info['train_mask'][eod:eod+mask_duration,:]=0
        # set fixation equal to 1 for all times; will then change
        trial_info['desired_output'][0, :eod, :] = 1 #only for first 500 ms

        for t in range(self.num_trials):#eft+1:esct+nrule esct+1+nrule:ersct

            trial_info['neural_input'][eft+1:eft+nrule,eodead:eot,t] = par['tuning_height']; #rule
            sample_dir = np.random.randint(par['num_motion_dirs'])
            sample_cat = np.floor(sample_dir/(par['num_motion_dirs']/2))
            match = np.random.randint(2)

            trial_info['neural_input'][:emt, eof:eos, t] += np.reshape(self.motion_tuning[0][:,sample_dir],(-1,1))

            if match==1:
                dir0 = int(sample_cat*par['num_motion_dirs']//2)
                dir1 = int(par['num_motion_dirs']//2 + sample_cat*par['num_motion_dirs']//2)
                possible_dirs = np.setdiff1d(list(range(dir0, dir1)), sample_dir)
                test_dir = possible_dirs[np.random.randint(len(possible_dirs))]
            else:
                test_dir = sample_cat*(par['num_motion_dirs']//2) + np.random.randint(par['num_motion_dirs']//2)
                test_dir = np.int_((test_dir+par['num_motion_dirs']//2)%par['num_motion_dirs'])

            trial_info['neural_input'][:emt, eod:eot, t] += np.reshape(self.motion_tuning[0][:,test_dir],(-1,1))
            if match==1:
                trial_info['desired_output'][1,eod:,t]=1
            else:
                trial_info['desired_output'][0,eod:,t]=1


            trial_info['sample'][t]=sample_dir
            trial_info['match'][t]=match
            trial_info['test'][t]=test_dir

            #plt.imshow(trial_info['desired_output'][:, :, t])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(trial_info['neural_input'][:, :, t])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(trial_info['train_mask'][:,:])
            #plt.colorbar()
            #plt.show()
        return trial_info

    def create_tuning_functions(self):

        """
        Generate tuning functions for the Postle task
        """
        motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))
        fix_tuning = np.zeros((par['num_fix_tuned'], par['num_receptive_fields']))
        rule_tuning = np.zeros((par['num_rule_tuned'], par['num_rules']))

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields'])))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/par['num_motion_dirs']))

        for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
            for i in range(len(stim_dirs)):
                for r in range(par['num_receptive_fields']):
                    d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                    n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
                    motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

        for n in range(par['num_fix_tuned']):
            for i in range(2):
                if n%2 == i:
                    fix_tuning[n,i] = par['tuning_height']/2

        for n in range(par['num_rule_tuned']):
            for i in range(par['num_rules']):
                if n%par['num_rules'] == i:
                    rule_tuning[n,i] = par['tuning_height']/2

        return np.squeeze(motion_tuning), fix_tuning, rule_tuning


    def plot_neural_input(self, trial_info):

        print(trial_info['desired_output'][ :, 0, :].T)
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(1, 1, 1)
        t = np.arange(0,400+500+2000,par['dt'])
        t -= 900
        t0,t1,t2,t3 = np.where(t==-500), np.where(t==0),np.where(t==500),np.where(t==1500)
        #im = ax.imshow(trial_info['neural_input'][:,0,:].T, aspect='auto', interpolation='none')
        im = ax.imshow(trial_info['neural_input'][:,:,0], aspect='auto', interpolation='none')
        #plt.imshow(trial_info['desired_output'][:, :, 0], aspect='auto')
        ax.set_xticks([t0[0], t1[0], t2[0], t3[0]])
        ax.set_xticklabels([-500,0,500,1500])
        ax.set_yticks([0, 9, 18, 27])
        ax.set_yticklabels([0,90,180,270])
        f.colorbar(im,orientation='vertical')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel('Motion direction')
        ax.set_xlabel('Time relative to sample onset (ms)')
        ax.set_title('Motion input')
        plt.show()
        plt.savefig('stimulus.pdf', format='pdf')

        """
        f = plt.figure(figsize=(9,4))
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(trial_info['sample_dir'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 2)
        ax.imshow(trial_info['test_dir'],interpolation='none',aspect='auto')
        ax = f.add_subplot(1, 3, 3)
        ax.imshow(trial_info['match'],interpolation='none',aspect='auto')
        plt.show()
        """
