#import numpy as np

from .fieldmaps import read_fieldmap_h5

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


def read_output_h5(h5, verbose=False):
    """
    Read all Impact-T output from h5 handle.
    
    Corresponds exactly to the output of write_impact_output_h5
    """
    d = {}
    if 'run_info' in h5:
        d['run_info'] = read_attrs_h5(h5['run_info'])
        if verbose:
            print('h5 read run_info')
    
    if 'stats' in h5:
        d['stats'] = read_datasets_h5(h5['stats'])
        if verbose:
            print('h5 read stats')
            
    if 'slice_info' in h5:
        g = h5['slice_info']
        slice_info = d['slice_info'] = {}
        for k in g:
            slice_info[k] = read_datasets_h5(g[k])
        if verbose:
            print('h5 read slice_info')            
            
    return d