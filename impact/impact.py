#import numpy as np

from .parsers import parse_impact_input, load_many_fort, FORT_STAT_TYPES, FORT_PARTICLE_TYPES, FORT_SLICE_TYPES, header_str
from .writers import write_impact_input
from . import tools
import numpy as np
import tempfile
import shutil
import os


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))

class Impact:
    """
    
    
    Files will be written into a temporary directory within workdir. 
    If workdir=None, a location will be determined by the system. 
    
    """
    def __init__(self,
                input_file='ImpactT.in',
                impact_bin='$IMPACTT_BIN',
                workdir=None,
                verbose=True):
        
        # Save init
        self.original_input_file = input_file
        self.workdir = workdir
        self.verbose=verbose
        self.impact_bin = impact_bin
        
        # These will be set
        self.timeout=None
        self.input = None
        self.output = {}
        self.tempdir = None # Actual working path. 
        self.auto_cleanup = True
        
        # Run control
        self.finished = False
        self.configured = False
        
        # Call configure
        if os.path.exists(input_file):
            self.configure()                
        else:
            self.vprint('Warning: Input file does not exist. Not configured.')

    def __del__(self):
        if  self.auto_cleanup:
            self.clean() # clean directory before deleting

    def clean(self):   
        # Only remove temporary directory. Never delete anything else!!!
        if self.tempdir:
            self.vprint('deleting: ', self.tempdir)
            shutil.rmtree(self.tempdir)            
            
    def configure(self):
        self.configure_impact(self.original_input_file, self.workdir)
        self.configured = True
        
        
    def configure_impact(self, input_file, workdir):
        for f in [self.original_input_file]:
            f = full_path(f)
            self.original_path, _ = os.path.split(f) # Get original path
            print(self.original_path)
            assert os.path.exists(f)
        # Parse input file. This should be a dict with: header, lattice, fieldmaps, input_particle_file
        self.input = parse_impact_input(self.original_input_file)      
        
        # Temporary directory for path
        self.tempdir = os.path.abspath(tempfile.TemporaryDirectory(prefix='temp_impactT_', dir=workdir).name)
        os.mkdir(self.tempdir)
     
        self.vprint('Configured for tempdir:', self.tempdir)
    
    def load_output(self):
        self.output['stats'] = load_many_fort(self.tempdir, FORT_STAT_TYPES, verbose=self.verbose)
        self.output['slice_info'] = load_many_fort(self.tempdir, FORT_SLICE_TYPES, verbose=self.verbose)
        
    def load_particles(self):
        self.particles = load_many_fort(self.tempdir, FORT_PARTICLE_TYPES, verbose=self.verbose)
        
        
        
    def run(self):
        if not self.configured:
            self.vprint('not configured to run')
            return
        self.run_impact(verbose=self.verbose, timeout=self.timeout)        
    
    def run_impact(self, verbose=False, timeout=None):
        
        # Check that binary exists
        self.impact_bin = full_path(self.impact_bin)
        assert os.path.exists(self.impact_bin)
        
        
        init_dir = os.getcwd()
        os.chdir(self.tempdir)
        
        # Write input
        self.write_input()
        
        runscript = [self.impact_bin]

        if timeout:
            res = tools.execute2(runscript, timeout=timeout)
            log = res['log']
            self.error = res['error']
            self.output['run_error'] = self.error
            self.output['why_run_error'] = res['why_error']

        else:
            # Interactive output, for Jupyter
            log = []
            for path in tools.execute(runscript):
                if verbose:
                    print(path, end="")
                log.append(path)

        self.log = log
                        
        # Load output    
        self.load_output()
        self.load_particles()
        
        # Return to init_dir
        os.chdir(init_dir)         
        
        self.finished = True
        
    def write_input(self,  input_filename='ImpactT.in'):
        
        path = self.tempdir
        assert os.path.exists(path)
        
        filePath = os.path.join(path, input_filename)
        # Write main input file
        write_impact_input(filePath, self.input['header'], self.input['lattice'])
        
        # Write fieldmaps
        for fmap, data in self.input['fieldmaps'].items():
            file = os.path.join(path, fmap)
            np.savetxt(file, data)
        
        # Input particles (if required)
        # Symlink
        if self.input['header']['Flagdist'] == 16:
            src = self.input['input_particle_file']
            dest = os.path.join(path, 'partcl.data')
            if not os.path.exists(dest):
                os.symlink(src, dest)
            else:
                self.vprint('partcl.data already exits, will not overwrite.')

    def vprint(self, *args):
        # Verbose print
        if self.verbose:
            print(*args)
    
        
    def __str__(self):
        path = self.tempdir
        s = header_str(self.input['header'])
        if self.finished:
            s += 'Impact-T finished in '+path
        elif self.configured:
            s += 'Impact-T configured in '+path
        else:
            s += 'Impact-T not configured.'
        return s
        
        
    
        