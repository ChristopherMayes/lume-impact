#import numpy as np

from .parsers import parse_impact_input, load_many_fort, FORT_STAT_TYPES, FORT_PARTICLE_TYPES, FORT_SLICE_TYPES, header_str
from .writers import write_impact_input
from .lattice import ele_dict_from
from . import tools
import numpy as np
import tempfile
import shutil
from time import time
import os



class Impact:
    """
    
    
    Files will be written into a temporary directory within workdir. 
    If workdir=None, a location will be determined by the system. 
    This behavior can
    
    """
    def __init__(self,
                input_file='ImpactT.in',
                impact_bin='$IMPACTT_BIN',
                workdir=None,
                use_mpi = False,
                mpi_exe = 'mpirun', # If needed
                path = None, # Actual simulation path. If set, will not make a temporary directory. 
                verbose=True):
        
        # Save init
        self.original_input_file = input_file
        self.workdir = workdir
        self.verbose=verbose
        self.impact_bin = impact_bin
        self.mpi_exe = mpi_exe
        self.use_mpi = use_mpi
        self.path = path # Actual working path. 
        
        # These will be set
        self.timeout=None
        self.input = None
        self.output = {}
        self.auto_cleanup = True
        self.ele = {} # Convenience lookup of elements in lattice by name
        
        
        # Run control
        self.finished = False
        self.configured = False
        self.using_tempdir = False
        
        # Call configure
        if os.path.exists(input_file):
            self.configure()                
        else:
            self.vprint('Warning: Input file does not exist. Not configured.')

    def __del__(self):
        if self.auto_cleanup:
            self.clean() # clean directory before deleting

    def clean(self, override=False):   
        # Only remove temporary directory. Never delete anything else!!!
        if self.using_tempdir or override:
            self.vprint('deleting: ', self.path)
            shutil.rmtree(self.path)
        else: 
            self.vprint('Warning: no cleanup because path is not a temporary directory:', self.path)
            
    def configure(self):
        self.configure_impact(self.original_input_file, self.workdir)
        self.configured = True

        
    def configure_impact(self, input_file, workdir):
        for f in [self.original_input_file]:
            f = tools.full_path(f)
            self.original_path, _ = os.path.split(f) # Get original path
            print(self.original_path)
            assert os.path.exists(f)
        # Parse input file. This should be a dict with: header, lattice, fieldmaps, input_particle_file
        self.input = parse_impact_input(self.original_input_file)      
        
        # Set ele dict:
        self.ele = ele_dict_from(self.input['lattice'])
        
        # Temporary directory for path
        if not self.path:
            self.path = os.path.abspath(tempfile.TemporaryDirectory(prefix='temp_impactT_', dir=workdir).name)
            os.mkdir(self.path)
            self.using_tempdir = True
        else:
            self.using_tempdir = False
     
        self.vprint(header_str(self.input['header']))
        self.vprint('Configured to run in:', self.path)
        
        
    
    def load_output(self):
        self.output['stats'] = load_many_fort(self.path, FORT_STAT_TYPES, verbose=self.verbose)
        self.output['slice_info'] = load_many_fort(self.path, FORT_SLICE_TYPES, verbose=self.verbose)
        
    def load_particles(self):
        self.particles = load_many_fort(self.path, FORT_PARTICLE_TYPES, verbose=self.verbose)
        
        
        
    def run(self):
        if not self.configured:
            self.vprint('not configured to run')
            return
        self.run_impact(verbose=self.verbose, timeout=self.timeout)        
    
    
    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script
        """
        
        if self.use_mpi:
            n_procs = self.input['header']['Npcol'] * self.input['header']['Nprow']
            runscript = [self.mpi_exe, '-n', str(n_procs), tools.full_path(self.impact_bin)]
        else:
            runscript = [tools.full_path(self.impact_bin)]
            
        if write_to_path:
            with open(os.path.join(self.path, 'run'), 'w') as f:
                f.write(' '.join(runscript))
            
        return runscript

    
    def run_impact(self, verbose=False, timeout=None):
        
        # Check that binary exists
        self.impact_bin = tools.full_path(self.impact_bin)
        assert os.path.exists(self.impact_bin)
        
        run_info = self.output['run_info'] = {}
        t1 = time()
        run_info['start_time'] = t1
        
        init_dir = os.getcwd()
        os.chdir(self.path)
        
        # Write input
        self.write_input()
        
        runscript = self.get_run_script()
        
        try: 
            if timeout:
                res = tools.execute2(runscript, timeout=timeout)
                log = res['log']
                self.error = res['error']
                run_info['error'] = self.error
                run_info['why_run_error'] = res['why_error']
    
            else:
                # Interactive output, for Jupyter
                log = []
                counter = 0
                for path in tools.execute(runscript):
                    # Fancy clearing of old lines
                    counter +=1
                    if verbose:
                        if counter < 15:
                            print(path, end='')
                        else:
                            print('\r', path.strip()+', elapsed: '+str(time()-t1), end='')
                    log.append(path)
                self.vprint('Finished.')
            self.log = log
                            
            # Load output    
            self.load_output()
            self.load_particles()
        except Exception as ex:
            print('Run Aborted', ex)
            run_info['error'] = True
            run_info['why_run_error'] = str(ex)
        finally:
            run_info['run_time'] = time() - t1
            # Return to init_dir
            os.chdir(init_dir)    
 
        self.finished = True
        
    def write_input(self,  input_filename='ImpactT.in'):
        
        path = self.path
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
        path = self.path
        s = header_str(self.input['header'])
        if self.finished:
            s += 'Impact-T finished in '+path
        elif self.configured:
            s += 'Impact-T configured in '+path
        else:
            s += 'Impact-T not configured.'
        return s
        