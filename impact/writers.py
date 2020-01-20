#import numpy as np

from .parsers import header_lines
from .lattice import lattice_lines
from .tools import fstr


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
        g2 = g.create_group('fieldmaps')
        
        for name, fieldmap in input['fieldmaps'].items():
            write_fieldmap_h5(g2, fieldmap, name=name)


    
def write_fieldmap_h5(h5, fieldmap, name=None):
    """

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    # Look for symlink fieldmaps
    if 'filePath' in fieldmap:
        g.attrs['filePath'] = fieldmap['filePath']
        return
    
    # Must be real fieldmap
    
    # Info attributes
    write_attrs_h5(g, fieldmap['info'], name='info')
    # Data as single dataset
    g['data'] = fieldmap['data']    
    
    
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
    

    

        
    
    
    
    
    
def write_input_particles_from_file(src, dest, n_particles, skiprows=1):
    """
    Write a partcl.data file from a source file, setting the number of particles as the first line. 
    
    If the source is another partcl.data file, use skiprows=1
    If the source does not have a header line, use skiprows=0
    
    Warning: Does not randomize or check the length of the file!
    """
    with open(src, 'rt') as fsrc:
        for _ in range(skiprows):
            fsrc.readline()
        with open(dest, 'wt') as fdest:
            fdest.write(str(n_particles)+'\n')
            for _ in range(n_particles):
                line = fsrc.readline()
                fdest.write(line)    
    
    


def write_impact_particles_h5(h5, particle_data, name=None, total_charge=1.0, time=0.0, speciesType='electron'):
    # Write particle data at a screen in openPMD BeamPhysics format
    # https://github.com/DavidSagan/openPMD-standard/blob/EXT_BeamPhysics/EXT_BeamPhysics.md

    

    if name:
        g = h5.create_group(name)
    else:
        g = h5

    n_particle = len(particle_data['x'])
    #-----------
    g.attrs['speciesType'] = fstr(speciesType)
    g.attrs['numParticles'] = n_particle
    g.attrs['chargeLive']  = total_charge
    g.attrs['totalCharge'] = total_charge
    g.attrs['chargeUnitSI'] = 1

    # Position
    g['position/x']=particle_data['x'] # in meters
    g['position/y']=particle_data['y']
    g['position/z']=particle_data['z']
    g['position'].attrs['unitSI'] = 1.0
    for component in ['position/x', 'position/y', 'position/z', 'position']: # Add units to all components
        g[component].attrs['unitSI'] = 1.0
        g[component].attrs['unitDimension']=(1., 0., 0., 0., 0., 0., 0.) # m

    # momenta
    g['momentum/x']=particle_data['GBx'] # gamma*beta_x
    g['momentum/y']=particle_data['GBy'] # gamma*beta_y
    g['momentum/z']=particle_data['GBz'] # gamma*beta_z
    for component in ['momentum/x', 'momentum/y', 'momentum/z', 'momentum']: 
        g[component].attrs['unitSI']= 2.73092449e-22 # m_e *c in kg*m / s
        g[component].attrs['unitDimension']=(1., 1., -1., 0., 0., 0., 0.) # kg*m / s
    
    # Constant records
    
    # Weights. All particles should have the same weight (macro charge)
    weight = total_charge / n_particle
    g2 = g.create_group('weight')
    g2.attrs['value']  = weight
    g2.attrs['shape'] = (n_particle)
    g2.attrs['unitSI'] = 1.0
    g2.attrs['unitDimension'] = (0., 0., 1, 1., 0., 0., 0.) # Amp*s = Coulomb    

    # Time
    g2 = g.create_group('time')
    g2.attrs['value']  = 0.0
    g2.attrs['shape'] = (n_particle)
    g2.attrs['unitSI'] = 1.0
    g2.attrs['unitDimension'] = (0., 0., 1, 0., 0., 0., 0.) # s

    # Status
    g2 = g.create_group('status')
    g2.attrs['value']  = 1
    g2.attrs['shape'] = (n_particle)
    g2.attrs['unitSI'] = 1.0
    g2.attrs['unitDimension'] = (0., 0., 0, 0., 0., 0., 0.) # dimensionless

