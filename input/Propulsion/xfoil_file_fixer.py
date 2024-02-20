import os
import re

source_path = r'input\Propulsion\propellerairfoil'
target_path = r'input/Propulsion/testing2'


for fname in os.listdir(source_path):
    # new_fname = os.path.splitext(fname)[0] + '_new.txt'
    new_fname = fname

    with (open(os.path.join(source_path, fname),'r') as source, 
          open(os.path.join(target_path, new_fname), 'w') as target):
        header = True

        all_lines = source.read().splitlines()
        if all_lines[0].lower().strip() == 'xflr5 v6.59': modify = True
        else: modify = False

        for line in all_lines:
            if line.count('-') > 15:
                header = False
            elif not header and modify:                
                all_entries = line.split()
                sel_entries = all_entries[0:-3] + [all_entries[-1]]

                mline = ' '
                for i in sel_entries:
                    mline += i + 3*' '
                line = mline
            
            target.write(line+'\n')