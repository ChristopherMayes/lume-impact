import subprocess
import os, errno
from hashlib import blake2b
from copy import deepcopy
import numpy as np
import json
import shutil
import datetime

from lume.tools import full_path, NpEncoder, find_executable, make_executable, native_type, fingerprint, isotime, execute, execute2



def parse_float(s):
    """
    Parse old-style float from string, replacing d->e for exponent
    """
    return float(s.lower().replace('d', 'e'))

def safe_loadtxt(filepath, **kwargs):
    """
    Similar to np.loadtxt, but handles old-style exponents d -> e
    """
    s = open(filepath).readlines()
    s = list(map(lambda x: x.lower().replace('d', 'e'), s))
    return np.loadtxt(s, **kwargs)



        
        
def runs_script(runscript=[], dir=None, log_file=None, verbose=True):
    """
    Basic driver for running a script in a directory. Will     
    """

    # Save init dir
    init_dir = os.getcwd()
    
    if dir:
        os.chdir(dir)
 
    log = []
    
    for path in execute(runscript):
        if verbose:
            print(path, end="")
        log.append(path)
    if log_file:
        with open(log_file, 'w') as f:
            for line in log:
                f.write(line)    
    
    # Return to init dir
    os.chdir(init_dir)                
    return log      




def find_property(s, key='name', separator=':', delims=[' ', ',', ';']):
    """
    Find property of the form key+delim+value
    
    Example: string = 'ax safsf name:QUAD01, ' should return 'QUAD01'
    
    """
    match=key+separator
    ix = s.find(match)
    if ix == -1:
        return None
    
    # Split out any other delims
    ss = s[ix+len(match):]
    for d in delims:
        ss = ss.split(d)[0]
    
    return ss

def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)    




    

#--------------------------------
# adding defaults to dicts
def fill_defaults(dict1, defaults, strict=True):
    """
    Fills a dict with defaults in a defaults dict. 
    
    dict1 must only contain keys in defaults.
    
    deepcopy is necessary!
    
    """
    # start with defaults
    for k in dict1:
        if k not in defaults and strict:
            raise Exception(f'Extraneous key: {k}. Allowable keys: '+', '.join(list(defaults)))
    for k, v in defaults.items():
        if k not in dict1:
            dict1[k] =  deepcopy(v)