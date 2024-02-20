import subprocess
import os
import numpy as np
import time

# Relevant paths
xfoil_path = r'C:\Users\Alfon\Desktop\xfoilp4.exe' # Path to the XFOIL executable
save_path = r'C:\Users\Alfon\Desktop\airfoildata' # Path can not be too long as it wil fail at some point
airfoil_path = r'c:\Users\Alfon\Desktop\wortman.dat' # optional if airfoil is NACA
airfoil_name = 'WORTMANN FX 63-137' # if it is a NACA, add a space between the code and the word NACA

print(f'Savepath is valid: {os.path.isdir(save_path)}')

# maximum iterations
max_iter = 500

# Reynolds range
start_reyn = 1e6
ending_reyn = 5e6
step_reyn = 1e6

#AoA range
start_alpha = "-5"
end_alpha = "30"
step_alpha = "0.1"

if (airfoil_name.lower()).count('naca') == 1:
    naca = True
else:
    naca = False
    assert airfoil_path.count('.dat') == 1, 'If the airfoil is not a NACA, specify the .dat file!'

# XFOIL commands  start up command
startup_commands = [
    airfoil_name,
    'oper',
    'iter',
    str(max_iter),
    f'visc {start_reyn}'
    ''
]

if not naca:
    startup_commands.insert(0, f'load {airfoil_path}')

# Run XFOIL
process = subprocess.Popen([xfoil_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

for command in startup_commands:
    process.stdin.write(command + '\n')  # Send each command

process.stdin.flush()  # Flush the buffer
for Reyn in np.arange(start_reyn, ending_reyn + step_reyn, step_reyn):
    Reyn = int(Reyn)
    fname = os.path.join(save_path, f'{airfoil_name}_Re{Reyn}').replace('\\', '/')+'.txt'
    commands = [
        'oper',
        're',
        str(Reyn),
        "pacc",
        fname,
        '',
        f'aseq {start_alpha} {end_alpha} {step_alpha}',
        'pacc',
        ''
    ]
    for command in commands:
        process.stdin.write(command + '\n')
    process.stdin.flush()
    
output, error  = process.communicate()

# change filenames to end in .txt

for file in os.listdir(save_path):
    if os.path.splitext(file)[-1] == '':
        os.rename(os.path.join(save_path, file), os.path.join(save_path, file+'.txt'))