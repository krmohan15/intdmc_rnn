import numpy as np
from parameters import *
import model
import sys

#task_list = ['OICDMC']
trial_type='OICDMC'
#task_list = ['spatialDMS']

for j in range(0,10):
#for task in task_list:
    #j = sys.argv[1]
    print('Training network on ', trial_type, 'iteration',j)
    #print('Training network on ', task)
    save_fn = trial_type+ str(j) + '.pkl'
    save_fn = trial_type+ '.pkl'
    #save_fn = task+ '.pkl'
    updates = {'trial_type': trial_type, 'save_fn': save_fn}
    update_parameters(updates)
    #model.train_and_analyze()
    model.main()
