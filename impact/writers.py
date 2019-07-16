#import numpy as np

from .parsers import header_lines
from .lattice import lattice_lines



def write_attrs_h5(h5, data, name=None):
    """
    Simple function to write dict data to attribues in a group with name
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
        
    for key in data:
        g.attrs[key] = data[key]
    return g

def write_datasets_h5(h5, data, name=None):
    """
    Simple function to write dict datasets in a group with name
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    for key in data:
        g[key] = data[key]
    return g


def write_impact_input(filePath, header, eles):
    """
    Write
    
    Note that the filename ultimately needs to be ImpactT.in
    
    """
    lines =  header_lines(header) + lattice_lines(eles)
    with open(filePath, 'w') as f:
        for line in lines:
            f.write(line+'\n')

            
            
def write_lattice_h5(h5, eles, name='lattice'):
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    for i, ele in enumerate(eles):
        write_attrs_h5(g, ele, name=str(i) )            
            
            
            
def write_impact_input_h5(h5, input, name='input', include_fieldmaps=True):
    """
    Write header
    
    Note that the filename ultimately needs to be ImpactT.in
    
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    # ImpactT.in as text
    header  = input['header']
    lattice = input['lattice']
    lines =  header_lines(header) + lattice_lines(lattice)
    data = '\n'.join(lines)   
    g.attrs['ImpactT.in'] = '\n'.join(lines)
    
    # Header
    write_attrs_h5(g, input['header'], name='header')
    
    # Eles
    write_lattice_h5(g, input['lattice'])
    
    # particle filename
    if 'input_particle_file' in input:
        g.attrs['input_particle_file'] = input['input_particle_file']
    
    # Any fieldmaps
    if 'fieldmaps' in input and include_fieldmaps:
        write_datasets_h5(g, input['fieldmaps'], name='fieldmaps')
    

def write_impact_output_h5(h5, output, name='output'):
    """
    Write all output data. Should be dict of dicts
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    if 'run_info' in output:
        write_attrs_h5(h5, output['run_info'], name='run_info' )
    
    if 'stats' in output:
         write_datasets_h5(h5, output['stats'], 'stats')
    
    if 'slice_info' in output:
        g2 = g.create_group('slice_info')
        for key, data in output['slice_info'].items():
             write_datasets_h5(g2, data, key)
    
    

def write_impact_particles_h5(h5, particle_data, name=None, total_charge=1.0, speciesType='electron'):
    # Write particle data at a screen in openPMD BeamPhysics format
    # https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md

    if name:
        g = h5.create_group(name)
    else:
        g = h5

    g.attrs['speciesType'] = speciesType

    g.attrs['totalCharge'] = total_charge

    n_particle = len(particle_data['x'])

    # Position
    g['position/x']=particle_data['x'] # in meters
    g['position/y']=particle_data['y']
    g['position/z']=particle_data['z']
    g['position'].attrs['unitSI'] = 1.0
    g['position'].attrs['unitDimension']=(1., 0., 0., 0., 0., 0., 0.) # m

    # momenta
    g['momentum/x']=particle_data['GBx'] # gamma*beta_x
    g['momentum/y']=particle_data['GBy'] # gamma*beta_y
    g['momentum/z']=particle_data['GBz'] # gamma*beta_z
    g['momentum'].attrs['unitSI']= 2.73092449e-22 # m_e *c in kg*m / s
    g['momentum'].attrs['unitDimension']=(1., 1., -1., 0., 0., 0., 0.) # kg*m / s
    
    # Constant records
    
    # Weights. All particles should have the same weight (macro charge)
    weight = total_charge / n_particle
    g2 = g.create_group('weight')
    g2.attrs['value']  = weight
    g2.attrs['shape'] = (n_particle)
    g2.attrs['unitSI'] = 1.0
    g2.attrs['unitDimension'] = (0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb    



    
#============================
# Corresponding read routines
    
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
            d[k] = h5['fieldmaps'][k][:]
    return d