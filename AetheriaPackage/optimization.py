import os
import sys
import shutil as sh
import json
import datetime as dt
import time

import numpy as np
import pandas as pd
import openmdao.api as om

sys.path.insert(0, os.path.abspath("."))

from AetheriaPackage.integration import run_integration, multi_run
import AetheriaPackage.alert as alert

#TODO add scalers.
class VTOLOptimization(om.ExplicitComponent):
    def __init__(self, 
                 design_variables,
                 constraint_variables,
                 objective_variables,
                 max_inner_loops = 10,
                 save_dir=r'output', 
                 init_estimate_path=r"input/default_initial_estimate.json", 
                 move_init_estimate=False, 
                 fname_addition='',
                 **kwargs):
        '''
        Class to perform multidisciplinary optimization (MDO) of Aetheria. 

        In the following, 'dict_variable' should have the following structure:

        dict_varaible = {'alias':str , 'json_route':[str, str], 
                        'show': bool, 'unit':str, 'upper':float, 'lower':float}

            'alias': specifies the variable name within openmdao framework
            'json_route': specifies route in results json file
            'show': specify whether to show the variable state at the end of 
                    outer loop
            'unit': indicate unit (for show)
            'upper': specify upper bound of this variable 
            'lower': specify lower bound of this variable

        The 'upper' and 'lower' parameters are only required for the design and
        constraint variables
        
        :param design_variables: Design variables to be used. 
        :type design_varaibles: List[dict_variable]
        :param constraint_variables: Constraint variables to be used. 
        :type constraint_variables: List[dict_variable]
        :param objective_variables: Objective variables to be used. 
        :type objective_variables: List[dict_variable]
        :param max_inner_loops: Maximum number of inner loops.
        :type max_inner_loops: int
        :param init_estimate_path: Path to initial estimate .json file. 
            By default "input/initial_estimate.json"
        :type init_estimate_path: str
        :param save_dir: Path to save directory, where all results are to be stored.
            Referred to as working directory.
        :type save_dir: str
        :param move_init_estimate: Specify whether to move or copy the initial estimate
            file to the working directory. If False, the file is only copied. If True,
            the file is moved.
        :type move_init_estiamte: bool
        :param fname_addition: Addition to the important filenames. Can be used as
            identifier of optimization.
        :type fname_addition: str
                
        '''

        super().__init__(**kwargs)

        start_time = dt.datetime.now()
        
        # storing variable names
        self._des_var  = design_variables
        self._cons_var = constraint_variables
        self._obj_var  = objective_variables

        self.max_inner_loops = max_inner_loops

        # initialising working directory
        self.init_estimate_path = init_estimate_path
        
        with open(self.init_estimate_path, 'r') as f:
            init = json.load(f)
        
        # set the optimization label
        self.label = f'{fname_addition}_{start_time:%b-%d_%H.%M}'

        # initialising the save directory
        self.dir_path = os.path.join(save_dir, "run_optimizaton_" + self.label)
        os.mkdir(self.dir_path)

        self.json_path = os.path.join(self.dir_path, "design_state_" + self.label + ".json")
        with open(self.json_path, 'w') as f:
            json.dump(init, f, indent=6)

        # make a copy or move the initial estimate file
        new_init_estimate_path = os.path.join(self.dir_path, 
                                              f'copy_initial_estimate_{self.label}.json')
        if move_init_estimate:
            sh.move(self.init_estimate_path, new_init_estimate_path)
        else:
            sh.copy(self.init_estimate_path, new_init_estimate_path)
        
        # update the initial_estimate_apath
        self.init_estimate_path = new_init_estimate_path

        self.outerloop_counter = 1

    def setup(self):    
        # Design variables
        for v in self._des_var:
            self.add_input(v['alias'])

        # Output required
        for v in self._cons_var + self._obj_var:
            self.add_output(v['alias'])    

    def setup_partials(self):
        # Partial derivatives are done using finite differences
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        #------------- Loading changed values from optimizers ------------------
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for v in self._des_var:
            data[v['json_route'][0]][v['json_route'][1]] = inputs[v['alias']][0]

        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=6)
        
        #---------------- Computing new values -----------------------------

        multi_run(self.init_estimate_path, self.outerloop_counter, 
                  self.json_path, self.dir_path, max_inner_loops=self.max_inner_loops)

        #----------------- Giving new values to the optimizer -------------------------
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for v in self._cons_var + self._obj_var:
            outputs[v['alias']] = data[v['json_route'][0]][v['json_route'][1]]

        self.outerloop_counter += 1

        #---------------------- Give updates on the design -----------------------------

        print(f"\n{40*'-'}\nUpdates on Design Variables\n{40*'-'}")
        for v in self._des_var:
            if v['show']: print(f'{v['json_route'][0]} > {v['json_route'][1]} = {inputs[v['alias']][0]:.2f} [{v['unit']}]')
        for v in self._cons_var + self._obj_var:
            if v['show']: print(f'{v['json_route'][0]} > {v['json_route'][1]} = {outputs[v['alias']][0]:.2f} [{v['unit']}]')

def read_optimization_variables(fpath=r'input\default_optimization_variables.csv'):
    '''
    Function to fetch the design variables, constraints and objective functions, and
    return a list of each containing dictionaries in the form:

    dict_varaible = {'alias':str , 'json_route':[str, str], 
            'show': bool, 'unit':str, 'upper':float, 'lower':float}

        'alias': specifies the variable name within openmdao framework
        'json_route': specifies route in results json file
        'show': specify whether to show the variable state at the end of 
                outer loop
        'unit': indicate unit (for show)
        'upper': specify upper bound of this variable 
        'lower': specify lower bound of this variable

    The 'upper' and 'lower' parameters are only specified for the design and
    constraint variables

    :param fpath: Path of .csv file containing variable data.
    :type fpath: str

    :param return: design_variables, constraint_variables, objective_varaibles
    :type return: List[dict], List[dict], List[dict]
    '''

    variables = pd.read_csv(fpath).T.replace({np.nan: None}).to_dict()

    design_variables     = []
    constraint_variables = []
    objective_variables  = []

    for i in variables:
        var = variables[i]
        if var['var_type']   == 'design'    : design_variables.append(var)
        elif var['var_type'] == 'constraint': constraint_variables.append(var)
        elif var['var_type'] == 'objective': objective_variables.append(var)

    for var in design_variables+constraint_variables+objective_variables:
        var['json_route'] = var['json_route'].split('/')
        del var['var_type']
    
    return design_variables, constraint_variables, objective_variables

def optimize_aetheria(init_estimate_path=r"input/default_initial_estimate.json", 
                      optimization_variables_path=r'input/default_optimization_variables.csv',
                      optimizer='COBYLA',
                      beep_finish=3,
                      **kwargs):
    
    t0 = time.localtime()

    # Fetch the design variables, constraints and objective functions.
    # ----------------------------------------------------------------------------
    design_variables, constraint_variables, objective_variables = read_optimization_variables(optimization_variables_path)

    
    # setting up problem and constraint
    # ----------------------------------------------------------------------------

    # recover initial data, to set up design variable
    with open(init_estimate_path, 'r') as f:
        init_data = json.load(f)  


    # defining problem object
    des = 'Integrated_design'
    prob = om.Problem()
    prob.model.add_subsystem(des, VTOLOptimization(design_variables, 
                                                   constraint_variables, 
                                                   objective_variables,
                                                   init_estimate_path=init_estimate_path,
                                                   **kwargs))
    
    # Define design variables, constraints and objectives in the problem object.
    for v in design_variables:
        prob.model.add_design_var(des+'.'+v['alias'], lower=v['lower'], upper=v['upper'])
        prob.model.set_input_defaults(des+'.'+v['alias'], init_data[v['json_route'][0]][v['json_route'][1]])
    for v in constraint_variables:
        prob.model.add_constraint(des+'.'+v['alias'], lower=v['lower'], upper=v['upper'])
    for v in objective_variables:
        prob.model.add_objective(des+'.'+v['alias'])
    
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = optimizer 

    # prob.setup(check=True)

    try:
        prob.run_driver()
    finally:
        t1 = time.localtime()
        #   print final time
        line = f'''
        Start optimization : {time.asctime(t0)}
        End optimization: {time.asctime(t1)}
        Execution time: {(time.mktime(t1)-time.mktime(t0))/60:.1f} min
        '''
        print(line)
                
        # alert user that program is finished
        alert.play_sound(repetitions=beep_finish)


if __name__ == '__main__':
    optimize_aetheria(max_inner_loops=10)