import subprocess
import os, errno
from hashlib import blake2b
from copy import deepcopy
import numpy as np
import json
import shutil
import datetime


def execute(cmd, cwd=None):
    """
    
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    
    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")
        
    Useful in Jupyter notebook
    
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
        
# Alternative execute
def execute2(cmd, timeout=None, cwd=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors. 
    """
    
    output = {'error':True, 'log':''}
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout = timeout, cwd=cwd)
      #  p = subprocess.run(' '.join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout = timeout)
        output['log'] = p.stdout
        output['error'] = False
        output['why_error'] =''
    except subprocess.TimeoutExpired as ex:
        output['log'] = ex.stdout+'\n'+str(ex)
        output['why_error'] = 'timeout'
    except:
        output['log'] = 'unknown run error'
        output['why_error'] = 'unknown'
    return output
        
        
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


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))

def make_executable(path):
    """
    Makes path executable. 
    
    See: https://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


def find_executable(exename=None, envname=None):
    """
    Finds an executable from a given name or environmental variable.
    
    If neigher are files, the path will be searched for exename
    
    """
    
    # Simply return if this exists
    if exename and os.path.isfile(exename):
        assert os.access(exename, os.X_OK), f'File is not executable: {exename}'
        return full_path(exename)
        
    envexe = os.environ.get(envname)   
    if envexe and os.path.isfile(envexe):
        assert os.access(envexe, os.X_OK), f'File is not executable: {envexe}'
        return full_path(envexe)
    
    if not exename and not envname:
         raise ValueError('No exename or envname ')

    # Start searching
    search_path = []
    #search_path.append(os.environ.get(envname))
    search_path.append(os.getcwd())
    search_path.append(os.environ.get('PATH'))
    search_path_str = os.pathsep.join(search_path)
    bin_location = shutil.which(exename, path=search_path_str)
    
    if bin_location and os.path.isfile(bin_location):
        return full_path(bin_location)
    
    raise ValueError(f'Could not find executable: exename={exename}, envname={envname}')
    
    


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



class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()  




def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, 'tolist', lambda: value)()    


"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()    
    
    

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