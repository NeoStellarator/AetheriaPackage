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
import AetheriaPackage.repo_tools as repo_tools

class VTOLOptimization(om.ExplicitComponent):
    def __init__(self, 
                 design_variables,
                 constraint_variables,
                 objective_variables,
                 inner_convg_settings={'inner_max_loops':25, 
                                       'inner_eps_exit':0.001, 
                                       'inner_n_min_eps':3,
                                       'inner_convg_saveplot':False},
                 save_dir=r'output', 
                 init_estimate_path=r"input/default_initial_estimate.json", 
                 move_init_estimate=False, 
                 fname_addition='',
                 measure_perf=False,
                 **kwargs):
        '''
        Class to perform multidisciplinary optimization (MDO) of Aetheria. 

        In the following, 'dict_variable' should have the following structure
        for design_variables, constraint_variables, objective_variables:

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
        :type design_variables: List[dict_variable]
        :param constraint_variables: Constraint variables to be used. 
        :type constraint_variables: List[dict_variable]
        :param objective_variables: Objective variables to be used. 
        :type objective_variables: List[dict_variable]
        :param inner_convg_settings: Settings for inner convergence loops.
            inner_max_loops:  max. number of inner loops (default 25), 
            inner_eps_exit:   difference criteria for termination (default 0.001), 
            inner_n_min_eps:  minimum number of consecutive loops satifying 
                              difference criteria (default 3)
            inner_convg_saveplot: save inner convergence plots
        :param save_dir: Path to save directory, where all results are to be stored.
            Referred to as working directory.
        :type save_dir: str
        :param init_estimate_path: Path to initial estimate .json file. 
            By default "input/initial_estimate.json"
        :type init_estimate_path: str
        :param move_init_estimate: Specify whether to move (True) or copy (False) 
        the initial estimate file to the working directory
        :type move_init_estiamte: bool
        :param fname_addition: Addition to the important filenames. Can be used as
            identifier of optimization.
        :type fname_addition: str
        :param measure_perf: Specify whether to measure the performance of the main
        disciplines in the optimization. See integration.py. It is recommended to leave 
        it False for important executions - it may fail with a PERMISISONERROR.
        :type measure_perf: bool (default False)
        '''
        start_time = dt.datetime.now()
        
        super().__init__(**kwargs)

        # storing variable names
        self._des_var  = design_variables
        self._cons_var = constraint_variables
        self._obj_var  = objective_variables
        
        # setting inner convergence configuration
        self.inner_convg_settings = inner_convg_settings

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

        # other parameters
        self.measure_perf = measure_perf

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

        multi_run(self.init_estimate_path, 
                  self.outerloop_counter, 
                  self.json_path, 
                  self.dir_path, 
                  measure_perf=self.measure_perf, 
                  **self.inner_convg_settings)

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

    :param fpath: Path of .csv file containing design, constrain and objective 
    variables (with bounds). By default 'input/default_optimization_variables.csv'.
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
                      optimizer_settings={'optimizer':'COBYLA'},
                      beep_finish=3,
                      scaling_report=False,
                      **kwargs):
    '''
    Main function to optimize Aetheria.

    :param init_estimate_path: Path to initial estimate .json file. By default 
    "input/initial_estimate.json"
    :type init_estimate_path: str
    :param optimization_variables_path: Path of .csv file containing design, constrain and 
    objective variables (with bounds). By default 'input/default_optimization_variables.csv'.
    For more details, see read_optimization_variables function.
    :type optimization_variables_path: str
    :param optimizer_settings: Settings for the Sci pyOptimizeDriver. Should at least contain
    the optimization algorithm. By default {'optimizer':'COBYLA'}. For more details, see
    https://openmdao.org/newdocs/versions/latest/features/building_blocks/drivers/scipy_optimize_driver.html)
    :type optimizer_settings: dict
    :param beep_finish: Number of beep sounds to alert user of end-of-execution.
    :type beep_finish: int
    :param scaling_report: Show scaling report. By default False.
    :type scaling_report: bool
    :param kwargs: Extra parameters for VTOLOptimization.
    '''
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
        prob.model.add_design_var(des+'.'+v['alias'], lower=v['lower'], upper=v['upper'], ref0=v['lower'], ref=v['upper'])
        prob.model.set_input_defaults(des+'.'+v['alias'], init_data[v['json_route'][0]][v['json_route'][1]])
    for v in constraint_variables:
        prob.model.add_constraint(des+'.'+v['alias'], lower=v['lower'], upper=v['upper'], ref0=v['lower'], ref=v['upper'])
    for v in objective_variables:
        prob.model.add_objective(des+'.'+v['alias'], ref=v['order_magnitude'])
    
    prob.driver = om.ScipyOptimizeDriver()
    for setting, value in optimizer_settings.items():
        prob.driver.options[setting] = value 
    prob.setup()

    try:
        prob.run_driver()
    finally:

        if scaling_report:
            if 'save_dir' in kwargs:
                report_path = os.path.join(kwargs['save_dir'], 
                                           'driver_scaling_report.html')
            else:
                report_path = os.path.join(os.path.split(init_estimate_path)[0], 
                                           'driver_scaling_report.html')
            prob.driver.scaling_report(outfile=report_path, show_browser=True)
        
        t1 = time.localtime()
        #   print final time
        line = f'''
        Start optimization : {time.asctime(t0)}
        End optimization: {time.asctime(t1)}
        Execution time: {(time.mktime(t1)-time.mktime(t0))/60:.1f} min
        '''
        print(line)
                
        # alert user that program is finished
        repo_tools.play_sound(repetitions=beep_finish)


if __name__ == '__main__':
    optimize_aetheria(max_inner_loops=10,
                      scaling_report=True,
                      save_dir=r'D:\OneDrive - Delft University of Technology\Honours\02 Python\Honours Project\AetheriaPackage\output\_beta_sensitivity_7\test')