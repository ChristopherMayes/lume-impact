#import numpy as np

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

def read_input_h5(h5):
    """
    Read all Impact-T input from h5 handle.
    """
    d = {}
    d['header'] = read_attrs_h5(h5['header'])
    d['lattice'] = read_list_h5(h5['lattice'])
    if 'input_particle_file' in h5.attrs:
        d['input_particle_file'] = h5.attrs['input_particle_file']
    if 'fieldmaps' in h5:
        d['fieldmaps'] = {}
        for k in h5['fieldmaps']:
            d['fieldmaps'][k] = h5['fieldmaps'][k][:]
    return d