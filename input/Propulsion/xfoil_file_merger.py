import os
import re

# initialise directories for read/write
dn_dir = r'input\Propulsion\propellerairfoil\down'
up_dir = r'input\Propulsion\propellerairfoil\up'
mg_dir = r'input\Propulsion\propellerairfoil\merge'

# store the list of up and down files
dn_files = os.listdir(dn_dir)
up_files = os.listdir(up_dir)

# confirm that the same reynolds number file is on both down/up directories
assert set(dn_files) == set(up_files), 'Both up and down directories must contain the same filenames'

for fname in dn_files:
    # construct new filename
    new_fname = re.split('_M', fname)[0]
    Re_num = float(new_fname[-5:])*1e6
    new_fname = new_fname[:-7] + f'Re{int(Re_num)}.txt'

    # rewrite if necessary
    if new_fname in os.listdir(mg_dir):
        os.remove(os.path.join(mg_dir, new_fname))

    # first copy the data from down file, without the final empty lines
    with (open(os.path.join(dn_dir, fname),'r') as source, 
          open(os.path.join(mg_dir, new_fname), 'a') as target):
        header = True
        for line in source.read().splitlines():
            if line.count('-') > 3:
                header = False
            elif not header and line == '':
                continue
            
            target.write(line+'\n')
    
    # append the data from up file, excluding the line with 0deg aoa
    # that is already found in down file
    with (open(os.path.join(up_dir, fname),'r') as source, 
          open(os.path.join(mg_dir, new_fname), 'a') as target):
        header = True
        for line in source.read().splitlines():
            if line.count('-') > 3: # find end of header
                header = False
                continue
            elif header or line == '': # do not write lines in header or empty lines
                continue
            elif line.split()[0] == '0.000': # do not write line with aoa for 0 deg
                continue
            
            target.write(line+'\n')