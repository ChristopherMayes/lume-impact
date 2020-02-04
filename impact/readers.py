#import numpy as np

from .fieldmaps import read_fieldmap_h5

from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import read_dataset_and_unit_h5

#============================
# Corresponding read routines to 

"""
Read routines that correspond to the write routines in writers.py

"""
    
def read_attrs_h5(h5):
    """
    Simple read attributes from h5 handle
    """
    d = {}
    for k in h5.attrs:
        d[k] = h5.attrs[k]
    return d

def read_datasets_h5(h5):
    """
    Simple read datasets from h5 handle into numpy arrays
    """
    d = {}
    for k in h5:
        d[k] = h5[k][:]
    return d

def read_list_h5(h5):
    """
    Read list from h5 file.
    
    A list is a group of groups named with their index, and attributes as the data. 
    
    The format corresponds to that written in write_lattice_h5
    """
    
    # Convert to ints for sorting
    ixlist = sorted([int(k) for k in h5])
    # Back to strings
    ixs = [str(i) for i in ixlist]
    eles = []
    for ix in ixs:
        e = read_attrs_h5(h5[ix])
        eles.append(e)
    return eles 

def read_input_h5(h5, verbose=False):
    """
    Read all Impact-T input from h5 handle.
    """
    d = {}
    d['header'] = read_attrs_h5(h5['header'])
    d['lattice'] = read_list_h5(h5['lattice'])
    if 'input_particle_file' in h5.attrs:
        d['input_particle_file'] = h5.attrs['input_particle_file']
        if verbose:
            print('h5 read:', 'input_particle_file')
    if 'fieldmaps' in h5:
        d['fieldmaps'] = {}
        for k in h5['fieldmaps']:
            d['fieldmaps'][k] = read_fieldmap_h5(h5['fieldmaps'][k]) 
        if verbose:
            print('h5 read fieldmaps:', list( d['fieldmaps']))
            
    return d


def read_output_h5(h5, expected_units=None, verbose=False):
    """
    Reads a properly archived Impact output and returns a dicts:
        output
        units
    
    
    Corresponds exactly to the output of writers.write_output_h5
    """
    
    o = {}
    o['run_info'] = dict(h5.attrs)
    
    units = {}
    
    if 'stats' in h5:
        name2 = 'stats'
        if verbose:
            print(f'reading {name2}')        
        g = h5[name2]
        o[name2] = {}
        for key in g:
            if expected_units:
                expected_unit = expected_units[key] 
            else:
                expected_unit = None
            o[name2][key], units[key] = read_dataset_and_unit_h5(g[key], expected_unit=expected_unit) 
            
    #TODO: this could be simplified
    if 'slice_info' in h5:
        name2 = 'slice_info'
        if verbose:
            print(f'reading {name2}')        
        g = h5[name2]
        o[name2] = {}
        for name3 in g:
            g2 = g[name3]
            o[name2][name3]={}
        
            for key in g2:
                if expected_units:
                    expected_unit = expected_units[key] 
                else:
                    expected_unit = None
                o[name2][name3][key], units[key] = read_dataset_and_unit_h5(g2[key], expected_unit=expected_unit)         
    
    
    if 'particles' in h5:
        o['particles'] = read_particles_h5(h5['particles'])        
        
    return o, units

def read_particles_h5(h5):
    """
    Reads particles from h5
    """
    dat = {}
    for g in h5:
        dat[g] = ParticleGroup(h5=h5[g])
    return dat      

