import numpy as np



def convert_particles_t_to_s(particles, s, style='Bmad'):
    """
    Drifts particles to a common s (z) position. 
    
    Outputs coordinates in Bmad style. 
    
    """
    
    ##mec2 = 0.51099895000e6
    
    # 
    x = particles['x']
    y = particles['y']
    z = particles['z']

    GBx = particles['GBx']
    GBy = particles['GBy']
    GBz = particles['GBz']
    
    # Get these quantities
    GB2 = GBx**2 + GBy**2 + GBz**2
    GB = np.sqrt(GB2)
    gamma = np.sqrt(1 + GB2) 
    beta = np.sqrt(GB2)/gamma # Total beta
    beta_z = GBz/gamma
    
    dz = z - s # Position relative to s
    xnew = x - dz * GBx/GBz
    ynew = y - dz * GBy/GBz
    cdt = -dz/beta_z
    betacdt = beta*cdt
    
    # Get ref:
    GB0 = np.mean(GB)
    G0 = np.mean(gamma)
    
    assert style == 'Bmad'

    # Form structured array
    dtype={'names': ['x', 'px/p0', 'y', 'py/p0', '-betacdt', 'delta'],
           'formats': 6*[np.float]}
    
    n = len(x)
    new_particles = np.empty(n, dtype=dtype)
    new_particles['x'] = xnew
    new_particles['px/p0'] = GBx / GB0
    new_particles['y'] = ynew
    new_particles['py/p0'] = GBy / GB0
    new_particles['-betacdt'] = -betacdt
    new_particles['delta'] = GB/GB0 -1
    
    return new_particles
    
    
def write_bmad_particles_ascii(particles, filename='test.beam0', charge=250e-12, mean_z=0, t_center=0, ix_ele=0, ix_bunch=1, species='electron'):
    n = len(particles)
    header = f'{ix_ele}\n{ix_bunch}\n{n}\nBEGIN_BUNCH\n{species}\n{charge}\n{mean_z}\n{t_center}'
    footer = 'END_BUNCH'
    
    np.savetxt(filename, particles, header=header, footer=footer, comments='')
    print('Written:', filename)    