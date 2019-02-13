import numpy as np
from parameters import *
import model
import sys

task_list = ['OICDMC']
#task_list = ['spatialDMS']

for task in task_list:
    #j = sys.argv[1]
    print('Training network on ', task)
    save_fn = task  + '.pkl'
    updates = {'trial_type': task, 'save_fn': save_fn}
    update_parameters(updates)
    #model.train_and_analyze()
    model.main()
