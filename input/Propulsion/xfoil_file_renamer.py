import os
import re

target_path = r'input\Propulsion\propellerairfoil'

airfoil = 'WORTMANN FX 63-137'
repeated_files = []

# expect filenames in the form "##########_Rexx.xxx_########.txt" or 
#                              "##########_Rexxxxxxxx_########.txt"

for old_fname in os.listdir(target_path):
    # remove the extension
    fname = os.path.splitext(old_fname)[0] 

    # extract Reynold's number
    Re_num = [i for i in re.split('_', fname) if i.count('Re') == 1]
    assert len(Re_num) == 1
    Re_num = int(float(Re_num[0][2:])) # this avoids a casting issue

    # mutliply by 1e6 if necessary (format from XFLR5 or XFOIL)
    if Re_num < 100: Re_num *= 1e6

    # construct new filename
    new_fname = f'{airfoil}_Re{Re_num}.txt'

    # accounting for the possibility of the same file name
    copy_fname = str(new_fname)
    count = 0
    while os.path.exists(os.path.join(target_path, new_fname)):
        repeated_files += new_fname
        count += 1
        new_fname = os.path.splitext(copy_fname)[0] + f'_({count}).txt'

    # rename file
    os.rename(os.path.join(target_path, old_fname), os.path.join(target_path, new_fname))

print(f'Finished renaming all files in {target_path} to desired format')
print(f'Repeated files: {repeated_files}')