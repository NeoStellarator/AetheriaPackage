import os
import sys
import shutil as sh
import json
import datetime as dt
import time

import numpy as np
import openmdao.api as om

sys.path.insert(0, os.path.abspath("."))

from AetheriaPackage.optimization import optimize_aetheria
import AetheriaPackage.alert as alert

#------------------------------------------------------------------------------------------
#                        BETA-SENSITIVITY -- OPTIMIZATIONS
#------------------------------------------------------------------------------------------

# set the target beta
beta = 0.500

fpath_opt_var = r'scripts\beta_sensitivity\beta_sensitivity_optimization_variables.csv'
work_dir = os.path.join('output', '_beta_sensitivity_3')

# determining initial estimate file
# -----------------------------------------------------------------------------------------
init_estimate_beta = 0 # '-1' for closest beta, '0' for default estimate, 'x' for beta=x (if it exists)

all_optim_folder  = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, f))]

beta_optim_folder = [] # betas of corresponding optim_folder
for optim_folder in all_optim_folder:
    optim_elem = optim_folder.split('_')
    for elem in optim_elem:
        if 'b=' in elem: beta_optim_folder.append(float(elem[2:]))

beta_optim_folder = np.array(beta_optim_folder)

if init_estimate_beta == 0:
    og_initial_estimate_file = r"input\\default_initial_estimate.json"
    print('Set initial conditions to the default (see input\\default_initial_estimate.json)')
else:
    if len(all_optim_folder) == 0:
        raise FileNotFoundError('There is no known optima to initialise this optimization!')

    if init_estimate_beta == -1:
        init_file_idx = np.argmin(np.abs(beta_optim_folder-beta))
    elif init_estimate_beta in beta_optim_folder:
        init_file_idx = np.where(beta_optim_folder==init_estimate_beta)
    else:
        raise FileNotFoundError(f'There is no known optima using beta={init_estimate_beta}')
    
    for file in os.listdir(all_optim_folder[init_file_idx]):
        if 'design' in file and 'state' in file: 
            og_initial_estimate_file = os.path.join(all_optim_folder[init_file_idx], file)
    print(f'\nSet initial conditions of b={beta_optim_folder[init_file_idx]:.3f}\n')

# copying the initial estimate file to the working directory (temporarily)
fname = os.path.split(og_initial_estimate_file)[-1]
sh.copy(og_initial_estimate_file, work_dir)
initial_estimate_path = os.path.join(work_dir, fname)

# ensure that the beta_crash coefficient is set
with open(initial_estimate_path, 'r') as f:
    init_data = json.load(f)

init_data['Fuselage']['beta_crash'] = beta

with open(initial_estimate_path, 'w') as f:
    json.dump(init_data, f, indent=6)

# running optimization
# -----------------------------------------------------------------------------------------
optimize_aetheria(init_estimate_path=initial_estimate_path,
                  optimization_variables_path=fpath_opt_var,
                  save_dir=work_dir,
                  move_init_estimate=True,
                  fname_addition=f'b={beta:.3f}')
