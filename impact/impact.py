from .parsers import parse_impact_input, load_many_fort, FORT_STAT_TYPES, FORT_PARTICLE_TYPES, FORT_SLICE_TYPES, header_str, header_bookkeeper, parse_impact_particles, load_stats, load_slice_info
from . import writers, fieldmaps
from .lattice import ele_dict_from, ele_str
from .particles import impact_particles_to_particle_data, identify_species
from . import tools, readers

from pmd_beamphysics import ParticleGroup

import h5py
import numpy as np

import tempfile
from time import time
import os




class Impact:
    """
    
    Files will be written into a temporary directory within workdir. 
    If workdir=None, a location will be determined by the system. 
    
    
    """
    def __init__(self,
                input_file=None, #'ImpactT.in',
                impact_bin='$IMPACTT_BIN',
                use_tempdir=True,
                workdir=None,
                use_mpi = False,
                mpi_exe = 'mpirun', # If needed
                verbose=True):
        
        # Save init
        self.original_input_file = input_file
        self.use_tempdir = use_tempdir
        
        if workdir:
            workdir = tools.full_path(workdir)
            assert os.path.exists(workdir), 'workdir does not exist: '+workdir   
        self.workdir = workdir
            
        self.verbose=verbose
        self.impact_bin = impact_bin
        self.mpi_exe = mpi_exe
        self.use_mpi = use_mpi

        
        # These will be set
        self.timeout=None
        self.input = {'header':{}, 'lattice':[]}
        self.output = {}
        
        self._units = {}
        self.ele = {} # Convenience lookup of elements in lattice by name
        
        
        # Run control
        self.finished = False
        self.configured = False
        
        # Call configure
        if input_file:
            self.load_input(input_file)
            self.configure()
        else:
            self.vprint('Warning: Input file does not exist. Not configured. Please set header and lattice.')
            
    def configure(self):
        self.configure_impact(workdir=self.workdir)
        
    def configure_impact(self, input_filePath=None, workdir=None):     
        
        if input_filePath:
            self.load_input(input_filePath)
        
        # Header Bookkeeper
        self.input['header'] = header_bookkeeper(self.input['header'], verbose=self.verbose)
        
        if  len(self.input['lattice']) == 0:
            self.vprint('Warning: lattice is empty. Not configured')
            self.configured = False
            return
   
        # Set ele dict:
        self.ele = ele_dict_from(self.input['lattice'])
            
        # Set paths
        if self.use_tempdir:

            # Need to attach this to the object. Otherwise it will go out of scope.
            self.tempdir = tempfile.TemporaryDirectory(dir=workdir)
            self.path = self.tempdir.name
            
        else:
            # Work in place
            self.path = self.original_path        
     
        self.vprint(header_str(self.input['header']))
        self.vprint('Configured to run in:', self.path)
        
        self.configured = True
        
        
    def load_input(self, input_filePath):
        f = tools.full_path(input_filePath)
        self.original_path, _ = os.path.split(f) # Get original path
        self.input = parse_impact_input(f)
    
    def load_output(self):
        """
        Loads stats, slice_info, and particles.
        """
        self.output['stats'], u = load_stats(self.path, species=self.species, verbose=self.verbose)
        self._units.update(u)
        
        self.output['slice_info'], u = load_slice_info(self.path, self.verbose)
        self._units.update(u)
        
        self.load_particles()
        
    def load_particles(self):
        # Standard output
        self.vprint('Loading particles')
        self.output['particles'] = load_many_fort(self.path, FORT_PARTICLE_TYPES, verbose=self.verbose)   
        # Additional particle files:
        for e in self.input['lattice']:
            if e['type'] == 'write_beam':
                name = e['name']
                fname = e['filename']
                full_fname = os.path.join(self.path, fname)
                if os.path.exists(full_fname):
                    self.particles[name] = parse_impact_particles(full_fname)
                    self.vprint(f'Loaded write beam particles {name} {fname}')

        # Convert all to ParticleGroup
        for name, pdata in self.particles.items():
            pg_data = impact_particles_to_particle_data(pdata, macrocharge=self.macrocharge)
            self.particles[name] = ParticleGroup(data = pg_data)
            self.vprint(f'Converted {name} to ParticleGroup')
      
    
    # Convenience routines
    @property
    def particles(self):
        return self.output['particles']
    
    def stat(self, key):
        """Con"""
        return self.output['stats'][key]
    
    def units(self, key):
        """pmd_unit of a given key"""
        return self._units[key]
        
    
    #--------------
    # Run
    def run(self):
        if not self.configured:
            self.vprint('not configured to run')
            return
        self.run_impact(verbose=self.verbose, timeout=self.timeout)        
    
    
    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script. Optionally writes a file 'run' with this line to path.
        """
        
        n_procs = self.input['header']['Npcol'] * self.input['header']['Nprow']
        
        if self.use_mpi:
            runscript = [self.mpi_exe, '-n', str(n_procs), tools.full_path(self.impact_bin)]
        else:
            if n_procs > 1:
                print('Error: n_procs > 1 but use_mpi = False')
                raise
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
        
        self.vprint('Running Impact-T in '+self.path)
        
        # Write input
        self.write_input()
        
        runscript = self.get_run_script()
        run_info['run_script'] = ' '.join(runscript)
        
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
        writers.write_impact_input(filePath, self.input['header'], self.input['lattice'])
        
        # Write fieldmaps
        for name, fieldmap in self.input['fieldmaps'].items():
            file = os.path.join(path, name)
            fieldmaps.write_fieldmap(file, fieldmap)

        # Input particles (if required)
        # Symlink
        if self.input['header']['Flagdist'] == 16:
            src = self.input['input_particle_file']
            dest = os.path.join(path, 'partcl.data')
            # Don't worry about overwriting in temporary directories
            if self.tempdir and os.path.exists(dest):
                os.remove(dest)
            
            if not os.path.exists(dest):
                writers.write_input_particles_from_file(src, dest, self.input['header']['Np'] )
            else:
                self.vprint('partcl.data already exits, will not overwrite.')
        
                
    def set_attribute(self, attribute_string, value):
        """
        Convenience syntax to set the header or element attribute. 
        attribute_string should be 'header:key' or 'ele_name:key'
        
        Examples of attribute_string: 'header:Np', 'SOL1:solenoid_field_scale'
        
        """
        name, attrib = attribute_string.split(':')
        if name == 'header':
            self.input['header'][attrib] = value
        else:
            self.ele[name][attrib] = value
    
        
                
    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.
        
        If no file is given, a file based on the fingerprint will be created.
        
        """
        if not h5:
            h5 = 'impact_'+self.fingerprint()+'.h5'
         
        if isinstance(h5, str):
            g = h5py.File(h5, 'w')
            self.vprint(f'Archiving to file {h5}')
        else:
            g = h5
            
        # All input
        writers.write_input_h5(g, self.input, name='input')

        # All output
        writers.write_output_h5(g, self.output, name='output', units=self._units) 
        
        return h5

    
    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.
        
        See: Impact.archive
        """
        if isinstance(h5, str):
            g = h5py.File(h5, 'r')
            self.vprint(f'Reading archive file {h5}')
        else:
            g = h5
        
        self.input = readers.read_input_h5(g['input'], verbose=self.verbose)
        self.output, self._units = readers.read_output_h5(g['output'], verbose=self.verbose)   

        
        self.vprint('Loaded from archive. Note: Must reconfigure to run again.')
        self.configured = False     
        
        if configure:    
            self.configure()           
                       
            
    @property
    def total_charge(self):
        H = self.input['header']
        return H['Bcurr']/H['Bfreq']
    
    @property
    def species(self):
        H = self.input['header']
        return identify_species(H['Bmass'], H['Bcharge'])
    
    @property
    def macrocharge(self):
        H = self.input['header']
        Np = H['Np']
        if Np == 0:
            self.vprint('Error: zero particles. Returning zero macrocharge')
            return 0
        else:
            return H['Bcurr']/H['Bfreq']/Np
        
        
    def fingerprint(self):
        """
        Data fingerprint using the input. 
        """
        return tools.fingerprint(self.input)
    
    def print_lattice(self):
        """
        Pretty printing of the lattice
        """
        for ele in self.input['lattice']:
            line = ele_str(ele)
            print(line)
    
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
        