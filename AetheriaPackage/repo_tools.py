import os
import winsound
import time

import pandas as pd


def play_sound(duration:int=500, freq:float=800, repetitions:int=10, pause=100):
    '''Play beeping sound to alert user
    
    :param duration: Duration [ms] of a single beep 
    :type duration: float
    :param freq: Frequency in [Hz] of beep, in range 37 to 32,767
    :type freq: float
    :param repetitions: Number of beeps
    :type repetitions: int
    :param pause: Duration [ms] of a pause between beeps 
    :type pause: float
    :return: N/A
    :rtype: NoneType

    '''
    for i in range(repetitions):
        winsound.Beep(freq, duration)
        time.sleep(pause/1000)


'''
Measure the (computational) performance of the different functions using the following.
This code was meant to be used in integration.py

run 'start_time' at the beginning of the code to initialise the file and columns
run 'lap_time' to record the time on the file, with additional information if needed.

The analysis of the data can be found in scripts>comp_perf_eval.ipynb.

parameters
----------
extra_cols : list
    Specify any extra informaiton to be included in csv. Typically, [iteration, description]
data : kwargs
    Specify the values associated with each of 'extra_cols'. 
    e.g. iteration=(1,1), description='structures'

BUG: When measure_perf=True in MDO, after around 10-20 min of execution, a PERMISSION_ERROR
     pops. It is recommended to leave measure_perf=False for important optimizations. 

'''

def start_time(save_path, extra_cols):
    if os.path.exists(save_path):
        return
    
    extra_cols += ['ref. time (s)', 'diff. time (s)']
    
    header = ''
    for col in extra_cols: header += col + ','
    header = header[0:-1] # remove the final comma

    with open(save_path, 'w') as f:
        f.write(header+'\n')


def lap_time(save_path, **data):
    assert os.path.exists(save_path), 'File does not exist!'

    df = pd.read_csv(save_path, header=0)

    file_cols = df.columns.values.tolist()

    new_line = {}
    for col in file_cols:
        if col in data.keys(): new_line[col] = data[col]

    new_line['ref. time (s)'] = f'{time.time():.2f}'

    num_measurements = df.shape[0]

    if num_measurements == 0:
        new_line['diff. time (s)'] = None
    else:
        last_line = df.iloc[-1]
        time_diff = (float(new_line['ref. time (s)']) - 
                     float(last_line['ref. time (s)'])  )
        new_line['diff. time (s)'] = f'{time_diff:.2f}'
    
    df_new_line = pd.DataFrame([new_line])
    df = pd.concat([df, df_new_line], ignore_index=True)

    df.to_csv(save_path, index=False, mode='a')


if __name__ == '__main__':
    play_sound()