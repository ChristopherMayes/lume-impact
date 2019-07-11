import numpy as np
import os

#-----------------
# Parsing ImpactT input file


#-----------------
# ImpactT Header
# lattice starts at line 10

# Header dicts
HNAMES={}
HTYPES={}
HDEFAULTS = {}
# Line 1
HNAMES[1]  = ['Npcol', 'Nprow']
HTYPES[1] = [int, int]
HDEFAULTS[1] = [1,1]

# Line 2
HNAMES[2] = ['Dt', 'Ntstep', 'Nbunch']
HTYPES[2] = [float, int, int]
HDEFAULTS[2] = [0, 100000000, 1] # Dt must be set

# Line 3
HNAMES[3]    = ['Dim', 'Np', 'Flagmap', 'Flagerr', 'Flagdiag', 'Flagimg', 'Zimage']
HTYPES[3]    = [int,   int,   int,      int,        int,       int,        float];
HDEFAULTS[3] = [999,   0,     1,        0,          2,         1,          0.02]

# Line 4
HNAMES[4] = ['Nx', 'Ny', 'Nz', 'Flagbc', 'Xrad', 'Yrad', 'Perdlen']
HTYPES[4] = [int, int, int, int, float, float, float]
HDEFAULTS[4] = [32, 32, 32, 1, 0.015, 0.015, 100.0]

# Line 5
HNAMES[5] = ['Flagdist', 'Rstartflg', 'Flagsbstp', 'Nemission', 'Temission']
HTYPES[5] = [int, int, int, int, float ]
HDEFAULTS[5] = [16, 0, 0, 400, 1.4e-11]

# Line 6-8
HNAMES[6] = ['sigx(m)', 'sigpx', 'muxpx', 'xscale', 'pxscale', 'xmu1(m)', 'xmu2']
HTYPES[6] = [float for i in range(len(HNAMES[6]))]
HDEFAULTS[6] = [0.0 for i in range(len(HNAMES[6]))]
HNAMES[7] = ['sigy(m)', 'sigpy', 'muxpy', 'yscale', 'pyscale', 'ymu1(m)', 'ymu2']
HTYPES[7] = HTYPES[6] 
HDEFAULTS[7] = [0.0 for i in range(len(HNAMES[7]))]
HNAMES[8] = ['sigz(m)', 'sigpz', 'muxpz', 'zscale', 'pzscale', 'zmu1(m)', 'zmu2']
HTYPES[8] = HTYPES[6]
HDEFAULTS[8] = [0.0 for i in range(len(HNAMES[8]))]

# Line 9
HNAMES[9] = ['Bcurr', 'Bkenergy', 'Bmass', 'Bcharge', 'Bfreq', 'Tini']
HTYPES[9] = [float for i in range(len(HNAMES[9]))]
HDEFAULTS[9] = [1.0, 1.0, 510998.946, -1.0, 2856000000.0, 0.0]

# Collect all these
HEADER_NAMES=[]
HEADER_TYPES=[]
for i in range(1,10):
    HEADER_NAMES.append(HNAMES[i])
    HEADER_TYPES.append(HTYPES[i])
# Flattened version
ALL_HEADER_NAMES = [item for sublist in HEADER_NAMES for item in sublist]
ALL_HEADER_TYPES = [item for sublist in HEADER_TYPES for item in sublist]
HEADER_TYPE_OF = dict(zip(ALL_HEADER_NAMES, ALL_HEADER_TYPES))



#-----------------
# Some help for keys above
help = {}
# Line 1
help['Npcol'] =  'Number of columns of processors, used to decompose domain along Y dimension.'
help['Nprow'] =  'Number of rows of processors, used to decompose domain along Z dimension.'
# Line 2
help['Dt'] =  'Time step size (secs).'
help['Ntstep'] = 'Maximum number of time steps.'
help['Nbunch'] = 'The initial distribution of the bunch can be divided longitudinally into Nbunch slices. See the manual.'
# Line 3
help['Dim'] = 'Random seed integer'
help['Np'] = 'Number of macroparticles to track'
help['Flagmap']='Type of integrator. Currently must be set to 1.' 
help['Flagerr']='Error study flag. 0 - no misalignment and rotation errors; 1 - misalignment and rotation errors are allowed for Quadrupole, Multipole (Sextupole, Octupole, Decapole) and SolRF elements. This function can also be used to simulate the beam transport through rotated beam line elements such as skew quadrupole etc.'
help['Flagdiag']='Diagnostics flag: 1 - output the information at given time, 2 - output the information at the location of bunch centroid by drifting the particles to that location, 3 or more - no output.'
help['Flagimg']='Image charge flag. If set to 1 then the image charge forces due to the cathode are included. The cathode is always assumed to be at z = 0. To not include the image charge forces set imchgF to 0.'
help['Zimage']='z position beyond which image charge forces are neglected. Set z small to speed up the calculation but large enough so that the results are not affected.'
# Line 4
help['Nx'] = 'Number of mesh points in x'
help['Ny'] = 'Number of mesh points in y'
help['Nz'] = 'Number of mesh points in z'
help['Flagbc'] = 'Field boundary condition flag: Currently must be set to 1 which corresponds to an open boundary condition.'
help['Xrad'] = 'Computational domain size in x'
help['Yrad'] = 'Computational domain size in x'
help['Perdlen'] = 'Computational domain size in z. Must be greater than the lattice length'
# Line 5
help['Flagdist'] = 'Type of the initial distribution'
help['Rstartflg'] = 'If restartf lag = 1, restart the simulation from the previous check point. If restartf lag = 0, start the simulation from the beginning.'
help['Flagsbstp'] = 'Not used.'
help['Nemission'] = 'There is a time period where the laser is shining on the cathode and electrons are being emitted. Nemisson gives the number of numerical emission steps. More steps gives more accurate modeling but the computation time varies linearly with the number of steps. If Nemission < 0, there will be no cathode model. The particles are assumed to start in a vacuum.'
help['Temission'] = 'Laser pulse emission time (sec.) Note, this time needs to be somewhat greater than the real emission time in the initial longitudinal distribution so that the time step size is changed after the whole beam is a few time steps out of the cathode.'

# Line 6-8
help['sigx(m)'] = 'Distribution sigma_x in meters'
help['sigpx']   = 'Distribution sigma_px, where px is gamma*beta_x'
help['muxpx']   = 'Distribution correlation <x px>, where px is gamma*beta_x'
help['xscale']  = 'Scale factor for distribution x'
help['pxscale'] = 'Scale factor for distribution px'
help['xmu1(m)'] = 'Distribution mean for x in meters'
help['xmu2']    = 'Distribution mean for px, where px is gamma*beta_x'

help['sigy(m)'] = 'Distribution sigma_y in meters'
help['sigpy']   = 'Distribution sigma_py, where px is gamma*beta_y'
help['muypy']   = 'Distribution correlation <y py>, where py is gamma*beta_y'
help['yscale']  = 'Scale factor for distribution y'
help['pxycale'] = 'Scale factor for distribution py'
help['ymu1(m)'] = 'Distribution mean for y in meters'
help['ymu2']    = 'Distribution mean for py, where py is gamma*beta_y'

help['sigz(m)'] = 'Distribution sigma_z in meters'
help['sigpz']   = 'Distribution sigma_pz, where pz is gamma*beta_z'
help['muzpz']   = 'Distribution correlation <z pz>, where pz is gamma*beta_z'
help['zscale']  = 'Scale factor for distribution z'
help['pzscale'] = 'Scale factor for distribution pz'
help['zmu1(m)'] = 'Distribution mean for z in meters'
help['zmu2']    = 'Distribution mean for pz, where pz is gamma*beta_z'


# Line 9
help['Bcurr'] = 'Beam current in Amps'
help['Bkenergy'] = 'Initial beam kinetic energy in eV. WARNING: this one is only used to drift the particle out of the wall. The real initial beam energy needs to be input from ”xmu6” in the initial distribution or the particle data file for the readin distribution.'
help['Bmass'] = 'Mass of the particles in eV.'
help['Bcharge'] = 'Particle charge in units of proton charge.'
help['Bfreq'] = 'Reference frequency in Hz.'
help['Tini'] = 'Initial reference time in seconds.'




def header_is_good(header_dict):
    """
    Sanity check of header. 
    """
    good = False
    
    if header_dict['Flagmap'] != 1:
        return False
    if header_dict['Flagbc'] != 1:
        return False    

    
    h = header_dict.copy()
    # These keys must be in header
    for k in ALL_HEADER_NAMES:
        if k not in h:
            print('Missing key:', k)
            return False
        else:
            v1 = h.pop(k, None)
            # Check type conversion
            v2 = HEADER_TYPE_OF[k](v1)
            if v1 - v2 !=0:
                print('Type conversion failed: ', v1, v2)
                return False
    

    
    if len(h) == 0:
        good = True
    
    return good

def header_str(H):
    """
    Summary information about the header
    """
    qb_pC = H['Bcurr']/H['Bfreq']*1e12
    Nbunch = H['Nbunch']
  
    if H['Flagimg']:
        start_condition = 'Cathode start at z = 0 m\n   emission time: '+str(H['Temission'])+' s\n   image charges neglected after z = '+str(H['Zimage'])+' m'
        
    else:
        start_condition = 'Free space start'
        
    if H['Rstartflg'] == 0:
        restart_condition = 'Simulation starting from the beginning'
    elif ['Rstartflg'] == 1:
        restart_condition = 'Restarting simulation from checkpoint.'
    else:
        restart_condition = 'Bad restart condition: '+str(['Rstartflg'])
        

    dist_type = DIST_TYPE[H['Flagdist']]
    
    lines = [
        '================ Impact-T Summary ================',
        f'{Nbunch} bunch'
        f'total charge: {qb_pC} pC',
        f'Distribution type: {dist_type}',
        start_condition,
        'Tracking '+str(H['Np'])+' particles',
        'Processor domain: '+str(H['Nprow'])+ ' x '+str(H['Npcol'])+' = '+str(H['Nprow']*H['Npcol'])+' CPUs',
        'Computational domain: '+str(H['Xrad'])+' m x '+str(H['Yrad'])+' m x '+str(H['Perdlen'])+' m',
        'Space charge grid: '+str(H['Nx'])+' x '+str(H['Ny'])+' x '+str(H['Nz']),
        'Maximum time steps: '+str(H['Ntstep']),
        'Random Seed: '+str(H['Dim']),
        'Reference Frequency: '+str(H['Bfreq'])+' Hz',
        'Initial reference time: '+str(H['Tini'])+' s',
         restart_condition,
        '==================================================',
        '\n'
    ]
    
    return '\n'.join(lines)


#-----------------
# Distribution types

DIST_TYPE = {
    1:'uniform',
    2:'gauss3',
    3:'waterbag',
    4:'semigauss',
    5:'kv3d',
    16:'read',
    24:'readParmela',
    25:'readElegant',
    27:'colcoldzsob'
}
# TODO: ijk distribution

# Inverse
DIST_ITYPE={}
for k, v in DIST_TYPE.items():
    DIST_ITYPE[v]=k



#-----------------
# Util

def is_commented(line, commentchar='!'):
    if len(line) == 0:
        return True
    return line.strip()[0] == commentchar

# Strip comments
def remove_comments(lines):
    return [l for l in lines if len(l) >0 and not is_commented(l)]


def parse_line(line, names, types):
    """
    parse line with expected names and types
    
    Example:
        parse_line(line, ['dt', 'ntstep', 'nbunch'], [float, int,int])
    
    """
    x = line.split()
    values =  [types[i](x[i]) for i in range(len(x))]
    return dict(zip(names, values))



def parse_header(lines):
    x = remove_comments(lines)
    d = {}
    for i in range(9):
        d.update(parse_line(x[i], HEADER_NAMES[i], HEADER_TYPES[i]))
    return(d)

# 
def ix_lattice(lines):
    """
    Find index of beginning of lattice, end of header
    
    """
    slines = remove_comments(lines)
    latline = slines[9]
    for i in range(len(lines)):
        if lines[i] == latline:
            return i
        
        
        

def header_lines(header_dict):
    """
    Re-forms the header dict into lines to be written to the Impact-T input file
    """
    
    
    line0 = '! Impact-T input file'
    lines = [line0]
    for i in range(1,10):
        names = HNAMES[i]
        x = ' '.join([str(header_dict[n]) for n in names])
        lines.append(x)
    #' '.join(lines)    
    return lines

        
        

#-----------------
# Lattice
"""

The Impact-T lattice elements are defined with lines of the form:
Blength, Bnseg, Bmpstp, Btype, V1 ... V23

The V list depends on the Btype. 

"""

#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Parsers for each type of line

"""
Element type. 
"""
ele_type = {0:'drift', 
            1:'quadrupole',
            2:'constfoc',
            3:'solenoid',
            4:'dipole',
            5:'multipole', 
            101:'drift_tube_linac',
            204:'srf_cavity',
            105:'solrf', 
            110:'emfield',
            111:'emfield_cartesian',
            112:'emfield_cylindrical',
            113:'emfield_analytical',
            -1:'offset_beam',
            -2:'write_beam',
            -3:'write_beam_for_restart',
            -4:'change_timestep',
            -5:'rotationally_symmetric_to_3d',
            -6:'wakefield',
            -7:'merge_bins',
            -8:'spacecharge',
            -9:'write_slice_info', 
            -11:'collomate',
            -12:'matrix',
            -13:'dielectric_wakefield',
            -15:'point_to_point_spacecharge',
            -16:'heat_beam',
            -17:'rotate_beam', 
            -99:'stop'
           }
# Inverse dictionary
itype_of = {}
for k in ele_type:
    itype_of[ele_type[k]] = k 
    
def parse_type(line):
    """
    Parse a lattice line. This is the fourth item in the line. 
    
    Returns the type as a string. 
    
    """
    if is_commented(line):
        return 'comment'
    i=int(line.split()[3])
    if i in ele_type:
        return ele_type[i]
    else:
        #print('Undocumented: ', line)
        return 'undocumented'

#-----------------------------------------------------------------
def parse_drift(line):
    """
    Drift (type 0)
    
    V1: zedge
    V2: radius Not used.
    """
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['zedge'] = float(v[1]) 
    d['radius'] = float(v[2]) 
    return d
def drift_v(ele):
    """
    Drift V list from ele dict

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    v = [ele, ele['zedge'], 0.0]     
    # optional
    if 'radius' in ele: v[2] = ele['radius']
    return v

    
    
#-----------------------------------------------------------------      
def parse_misalignments(v):
    """
    Parse misalignment portion of V list
    Common to several elements
    """
    d = {}
    d['x_offset'] = float(v[0])
    d['y_offset'] = float(v[1])
    d['x_rotation'] = float(v[2])
    d['y_rotation'] = float(v[3])
    d['z_rotation'] = float(v[4])
    return d

def misalignment_v(ele):
    """
    V list for misalignments
    """
    v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if 'x_offset' in ele: v[0] = ele['x_offset']
    if 'y_offset' in ele: v[1] = ele['y_offset']
    if 'z_offset' in ele: v[2] = ele['z_offset']
    if 'x_rotation' in ele: v[3] = ele['x_rotation']
    if 'y_rotation' in ele: v[4] = ele['y_rotation' ]     
    if 'z_rotation' in ele: v[5] = ele['z_rotation' ]         
    return v
    

    
#-----------------------------------------------------------------   
def parse_quadrupole(line):
    """
    Quadrupole (type 1)
    
    V1: zedge
    V2: quad gradient (T/m)
    V3: file ID
        If > 0, then include fringe field (using Enge function) and
        V3 = effective length of quadrupole.
    V4: radius (m)
    V5: x misalignment error (m)
    V6: y misalignment error (m)
    V7: rotation error x (rad)
    V8: rotation error y (rad)
    V9: rotation error z (rad)
    
    If V9 != 0, skew quadrupole
    V10: rf quadrupole frequency (Hz)
    V11: rf quadrupole phase (degree)
    """    
        
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['zedge'] = float(v[1]) 
    d['b1_gradient'] = float(v[2]) 
    if float(v[3]) > 0:
        d['L_effective'] = float(v[3])
    else:
        d['file_id'] = int(v[3])
    d['radius'] = float(v[4])
    d2 = parse_misalignments(v[5:10])
    d.update(d2)
    if v[10] != '/':
        d['rf_frequency'] = float(v[10])
        d['rf_phase_deg'] = float(v[11])
    return(d)    

def quadrupole_v(ele):
    """
    Quadrupole V list from ele dict
    
    V[0] is the original ele

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    v = [ele, ele['zedge'], ele['b1_gradient'], 0, 0.0]
    if 'file_id' in ele:
        v[3] = ele['file_id']
    else:
        v[3] = ele['L_effective']
        
    # optional
    if 'radius' in ele: v[4] = ele['radius']
    
    #misalignment list
    v += misalignment_v(ele)
    
    if 'rf_frequency' and 'rf_phase_deg' in ele:
        v += [ele['rf_frequency'], ele['rf_phase_deg']]
    
    return v


#-----------------------------------------------------------------  
def parse_solrf(line):
    """
    Solrf (type 105)
    
    V1: zedge, the real used field range in z is [zedge,zedge+Blength].
    V2: scale of RF field
    V3: RF frequency
    V4: theta0  Initial phase in degree.
    V5: file ID
    V6: radius
    V7: x misalignment error
    V8: y misalignment error
    V9: rotation error x
    V10: rotation error y
    V11: rotation error z
    V12: scale of solenoid B field. [Only used with SolRF element.]
    """
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['zedge'] = float(v[1]) 
    d['rf_field_scale'] = float(v[2]) 
    d['rf_frequency'] = float(v[3]) 
    d['theta0_deg'] = float(v[4])  #
    d['filename'] = 'rfdata'+str(int(float(v[5])))
    d['radius'] = float(v[6])
    d2 = parse_misalignments(v[7:12])
    d.update(d2)
    d['solenoid_field_scale'] = float(v[12]) 
    return d

def solrf_v(ele):
    """
    Solrf V list from ele dict.
    
    V[0] is the original ele

    """
    
    # Let v[0] be the original ele, so the indexing looks the same.
    file_id = int(ele['filename'].split('rfdata')[1])
    v = [ele, ele['zedge'], 1.0, 0.0, 0.0, file_id, 0.0]
    
    # optional
    if 'rf_field_scale' in ele: v[2] = ele['rf_field_scale']
    if 'rf_frequency' in ele: v[3] = ele['rf_frequency']
    if 'theta0_deg' in ele: v[4] = ele['theta0_deg']
    if 'radius' in ele: v[6] = ele['radius']
        
    #misalignment list
    v1 = misalignment_v(ele)
    v += [v1[0], v1[1], v1[3], v1[4], v1[5]] # solrf doesn't have z_offset
    
    if 'solenoid_field_scale' in ele:
        v.append(ele['solenoid_field_scale'])
    else:
        v.append(0.0)
    
    return v


#-----------------------------------------------------------------  
def parse_dipole(line):
    """
    Dipole (type 4)
    
    V1: zedge
    V2: x field strength (T)
    V3: y field strength (T)
    V4: file ID file ID to contain the geometry information of bend. 
    V5: half of gap width (m).
    V6: x misalignment error Not used.
    V7: y misalignment error Not used.
    V8: rotation error x Not used.
    V9: rotation error y Not used.
    V10: rotation error z Not used.
    
    """
    v = line.split()[3:-1] # V data starts with index 4
    d={}
    d['zedge'] = float(v[1]) 
    d['b_field_x'] = float(v[2]) 
    d['b_field'] = float(v[3]) 
    d['filename'] = 'rfdata'+str(int(float(v[4])))
    d['half_gap'] = float(v[5]) 
    return d
   
    
def dipole_v(ele):
    """
    dipole Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    # Get file integer
    f = ele['filename']
    ii = int(f.split('rfdata')[1])
    
    v = [ele, ele['zedge'], ele['b_field_x'], ele['b_field'], ii, ele['half_gap'] ] 
    
    return v    
    
#-----------------------------------------------------------------      
def parse_offset_beam(line, warn=False):
    """
    offset_beam (type -1)
    
    If btype = −1, steer the transverse beam centroid at given location V2(m) to position 
    x offset        V3(m)
    Px (γβx) offset V4
    y offset        V5(m)
    Py (γβy) offset V6
    z offset        V7(m)
    Pz (γβz) offset V8
    
    Assumes zero if these are not present.
    """

    v = line.split()[3:-2] # V data starts with index 4, ends with '/'
    d={}
    ##print (v, len(v))
    d['s']  = float(v[2]) 
    olist = ['x_offset', 'px_offset', 'y_offset','py_offset','z_offset','pz_offset' ]
    for i in range(6):
        if i+3 > len(v)-1:
            val = 0.0
        else:
            ## print('warning: offset_beam missing numbers. Assuming zeros', line)
            val = float(v[i+3])
        d[olist[i]] = val
    return d



def offset_beam_v(ele):
    """
    offset_beam Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, ele['s']] + [ele[x] for x in ['x_offset', 'px_offset', 'y_offset','py_offset','z_offset','pz_offset' ]]
    
    return v



#-----------------------------------------------------------------  
def parse_write_beam(line):
    """
    Write_beam (type -2)
    
    If btype = −2, output particle phase-space coordinate information at given location V3(m)
    into filename fort.Bmpstp with particle sample frequency Bnseg. Here, the maximum number
    of phase- space files which can be output is 100. Here, 40 and 50 should be avoided
    since these are used for initial and final phase space output.
    """
    x = line.split() 
    v = x[3:] # V data starts with index 4, ends with '/'
    d={}
    d['filename']='fort.'+x[2]
    d['sample_frequency'] = int(x[1])
    d['s'] = float(v[3]) 
    if int(v[0]) in [40, 50]:
        print('warning, overwriting file fort.'+x[2])
    return d



def write_beam_v(ele):
    """
    write_beam Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele['s']]   
    return v


#-----------------------------------------------------------------  
def parse_write_beam_for_restart(line):
    """
    Write_beam_for_restart (type -3)
    
    If btype = −3, output particle phase-space and prepare restart at given location V3(m)
    into filename fort.(Bmpstp+myid). Here, myid is processor id. On single processor, it is 0.
    If there are multiple restart lines in the input file, only the last line matters.

    """
    x = line.split() 
    v = x[3:] # V data starts with index 4, ends with '/'
    d={}
    d['filename']='fort.'+x[2]+'+myid'
    d['s'] = float(v[3]) 
    return d

def write_beam_for_restart_v(ele):
    """
    write_beam_for_restart Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele['s']]   
    return v



#-----------------------------------------------------------------  

def parse_change_timestep(line):
    """
    Change_timestep (type -4)
    
    If btype = −4, change the time step size from the initial Dt (secs)
    into V4 (secs) after location V3(m). The maximum number of time step change is 100.

    """
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['dt'] = float(v[4])
    d['s'] = float(v[3])

    return d


def change_timestep_v(ele):
    """
    change_timestep Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele['s'], ele['dt']]   
    return v


#----------------------------------------------------------------- 

def parse_rotationally_symmetric_to_3d(line):
    """
    If btype = −5, switch the simulation from azimuthal symmetry to fully 3d simulation after location V3(m). 
    This location should be set as large negative number such as -1000.0 in order to start the 3D simulation immediately after the electron emission.
    If there are multiple such lines in the input file, only the last line matters.
    
    """
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['s'] = float(v[3])

    return d    
    

def rotationally_symmetric_to_3d_v(ele):
    """
    rotationally_symmetric_to_3d Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele['s']]   
    return v

#-----------------------------------------------------------------  
def parse_wakefield(line):
    """
    Wakefield (type -6)
    
    If btype = −6, turn on the wake field effects between location V3(m) and V4(m).
    If Bnseg is greater than 0, the longitudinal and transverse wake function will
    be read in from file “fort.Bmpstp”. If Bnseg ≤ 0, the code will use analytical
    wake function described as follows. For analytical wake functions, the wake function
    parameters (iris radius) a = V5(m), (gap) g = V6(m), (period) L = V7(m).
    Here, the definition of these parameter can be found from
    SLAC-PUB-9663, “Short-Range Dipole Wakefields in Accelerating Structures for the NLC,” by Karl L.F. Bane.
    This will be updated in the future since the parameters a, g, L might change from cell to cell within a single structure.
    For the backward traveling wave structure, the iris radius “a” has to be set greater
    than 100, gap “g” set to the initialization location of BTW. For backward traveling wave structures,
    the wakes are hardwired inside the code following the report:
    P. Craievich, T. Weiland, I. Zagorodnov, “The short-range wakefields in the BTW accelerating structure of the ELETTRA linac,” ST/M-04/02.
    For −10 < a < 0, it uses the analytical equation from the 1.3 GHz Tesla standing wave structure.
    For a < −10, it assumes the 3.9 GHz structure longitudinal wake function.
    For external supplied wake function, The maximum number data point is 1000.
    The data points are assumed uniformly distributed between 0 and V7(m).
    The V6 has to less than 0.
    Each line of the fort.Bmpstp contains longitudinal wake function (V/m) and transverse wake function (V/m/m).
    
    """
    
    x = line.split()
    Bnseg = int(x[1])
    v = x[3:] # V data starts with index 4
    d={}
    d['s_begin'] = float(v[3])
    d['s_end'] = float(v[4])
    
    if Bnseg > 0:
        d['method'] = 'from_file'
        d['filename'] = 'fort.'+str(Bnseg)
    else:
        d['method'] = 'analytical'
        d['iris_radius'] = float(v[5])
        d['gap'] = float(v[6])
        d['period'] = float(v[7])
    
    
    return d

def wakefield_v(ele):
    """
    wakefield Impact-T style V list

    """
    
    
    # Let v[0] be the original ele, so the indexing looks the same.
    # V1 and V2 are not used.
    dummy = 1
    v = [ele, dummy, dummy,  ele['s_begin'], ele['s_end'] ]

    if ele['method'] == 'analytical':
         v += [ele['iris_radius'], ele['gap'], ele['period'] ]
    
    
    return v


#-----------------------------------------------------------------  
def parse_stop(line):
    """
    Stop (type -99)
    
    If bytpe = −99, stop the simulation at given location V3(m).
    
    
    """
    v = line.split()[3:] # V data starts with index 4
    d={'s':float(v[3])}
    return d


def stop_v(ele):
    """
    stop Impact-T style V list

    """
    
    
    # Let v[0] be the original ele, so the indexing looks the same.
    # V1 and V2 are not used.
    # Bad documentation? Looks like V1 and V3 are used
    dummy = 0.0
    v = [ele, ele['s'], dummy, ele['s']]
    
    return v



#-----------------------------------------------------------------  
def parse_spacecharge(line):
    """
    Spacecharge (type -8)
    
    if bytpe = −8, switch on/off the space-charge calculation at given location V3(m)
    according to the sign of V2 (> 0 on, otherwise off).
    
    """
    v = line.split()[3:] # V data starts with index 4
    d={}
    d['s'] = float(v[3])
    if float(v[2]) >0:
        d['is_on'] = True
    else:
        d['is_on'] = False

    return d

def spacecharge_v(ele):
    """
    spacecharge Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    
    if ele['is_on']:
        sign = 1.0
    else:
        sign = -1
        
    
    
    v = [ele, dummy, sign, ele['s'] ]
    
    return v


#----------------------------------------------------------------- 
def parse_write_slice_info(line):
    """
    If bytpe = −9, output slice-based information at given location V3(m) into file “fort.Bmpstp” using “Bnseg” slices.
    
    """
    
    x = line.split()
    Bnseg = int(x[1])
    Bmpstp = int(x[2])
    v = x[3:] # V data starts with index 4
    
    d={}
    d['n_slices'] = Bnseg
    d['filename'] = 'fort.'+str(Bmpstp)
    d['s'] = float(v[3])

    return d    
    
    
def write_slice_info_v(ele):
    """
    write_slice_info Impact-T style V list

    """
    # Let v[0] be the original ele, so the indexing looks the same.
    dummy = 0.0
    v = [ele, dummy, dummy, ele['s']]   
    return v

    
#-----------------------------------------------------------------  

#def parse_bpm(line):
#    """
#    BPM ????
#    """
#    return {}
#

#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Fieldmaps

def fieldmap_names(eles, prefix='rfdata'):
    """
    Extracts the unique fieldmap file names from eles. 
    This does not check if the files exist. 
    
    All fieldmaps should start with 'rfdata'
    """
    fmaps = {ele['filename'] for ele in eles if 'filename' in ele and ele['filename'].startswith(prefix) }
    return fmaps

def load_fieldmaps(fmap_names, dir):
    """
    Load fieldmap data as dict of np.array 
    Fieldmaps are simple 1d ASCII files
    """
    fmapdata={}
    for f in fmap_names:
        file = os.path.join(dir, f)
        fmapdata[f] = np.loadtxt(file)
    return fmapdata


#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Master element parsing


ele_parsers = {#'bpm': parse_bpm,
               'drift':parse_drift,
               'quadrupole':parse_quadrupole,
               'dipole':parse_dipole,
               'solrf':parse_solrf,
               'offset_beam':parse_offset_beam,
               'write_beam':parse_write_beam,
               'change_timestep':parse_change_timestep,
               'rotationally_symmetric_to_3d':parse_rotationally_symmetric_to_3d,
               'wakefield':parse_wakefield,
               'spacecharge':parse_spacecharge,
               'write_slice_info':parse_write_slice_info,
               'write_beam_for_restart':parse_write_beam_for_restart,
               'stop':parse_stop
              }  

def parse_ele(line):
    """
    Parse an Impact-T lattice line. 
    
    Returns an ele dict.

    Parses everything after / as the 'description'
    
    """
    if is_commented(line):
        return {'type':'comment', 'comment':line, 'L':0}
    
    # Ele
    e = {}
    
    x = line.split('/')
    # Look for extra info past /
    # 
    if len(x) > 1:
        # Remove spaces and comment character !
        e['description'] =  x[1].strip().strip('!')
        
    x = x[0].split()
    
    e['original'] = line # Save original line
    e['L'] = float(x[0])
    e['itype'] = int(x[3])
    if e['itype'] < 0:
        e['nseg'] = int(x[1])  # Only for type < 0
        e['bmpstp']= int(x[2]) # Only for type < 0
    
    if e['itype'] in ele_type:
        e['type'] = ele_type[e['itype']] 
        if e['itype'] >= -99:
            d2 = ele_parsers[e['type']](line)
            e.update(d2)        
    else:
        print('Warning: undocumented type', line)
        e['type'] ='undocumented'
    
    return e

        
        
#-----------------------------------------------------------------  

def add_s_position(elelist, s0=0):
    """
    Add 's' to element list according to their length. 
    s is at the end of an element.
    Assumes the list is in order.
    
    TODO: This isn't right
    """
    #s0 = -2.1459294
    s = s0
    for ele in elelist:
        if 's' not in ele and 'zedge' in ele:
            ele['s'] = ele['zedge'] + ele['L']
        # Skip these. 
       # if e['type'] in ['change_timestep', 'offset_beam', 'spacecharge', 'stop', 'write_beam', 'write_beam_for_restart']:
            continue
       # s = s + e['L']
       # e['s'] = s

def create_names(elelist):
    """
    Invent a name for elements
    """
    counter = {}
    for t in list(ele_type.values())+['comment', 'undocumented']:
        counter[t] = 0
    for e in elelist:
        t = e['type']
        counter[t] = counter[t]+1
        e['name'] = e['type']+'_'+str(counter[t])

def parse_lattice(lines):
    eles = [parse_ele(line) for line in lines]
    add_s_position(eles)
    create_names(eles)
    return eles
 

    
    
def parse_impact_input(filePath):
    """
    Parse and ImpactT.in file into header, lattice, fieldmaps
    
    
    """
    # Full path
    path, _ = os.path.split(filePath)
    
    # Read lines
    with open(filePath, 'r') as f:    
        data = f.read()
        lines = data.split('\n')
    
    header=parse_header(lines)
    
    # Check for input particles. Must be named 'partcl.data'.
    if header['Flagdist'] == 16:
        pfile = os.path.join(path, 'partcl.data')
        pfile = os.path.abspath(pfile)
        if not os.path.exists(pfile):
            print('Warning: partcl.data missing in path:', path)
    else:
        pfile = None
    
    
    # Find index of the line where the lattice starts
    ix = ix_lattice(lines)

    # Gather lattice lines
    latlines = lines[ix:]
    
    # This parses all lines. 
    eles = parse_lattice(latlines)
    
    # Get fieldmaps
   
    fmap_names = fieldmap_names(eles)
    fieldmaps = load_fieldmaps(fmap_names, path)
    

    # Ouput dict
    d = {}
    d['original_input'] = data
    d['input_particle_file'] = pfile
    d['header'] = header
    d['lattice'] = eles
    d['fieldmaps'] = fieldmaps
    
    return d
        
    
    
#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Particles

def parse_impact_particles(filePath, names=('x', 'GBx', 'y', 'GBy', 'z', 'GBz')):
    """
    Parse Impact-T input and output particle data.
    Typical filenames: 'partcl.data', 'fort.40', 'fort.50'
    
    Returns a strucured numpy array
    
    Impact-T input/output particles distribions are ASCII files with columns:
    x (m)
    gamma_beta_x (dimensionless)
    y (m)
    gamma*beta_y (dimensionless)
    z (m)
    gamma*beta_z (dimensionless)
    
    """
    
    
    dtype={'names': names,
           'formats': 6*[np.float]}
    pdat = np.loadtxt(filePath, skiprows=1, dtype=dtype)

    return pdat    
    

#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Parsers for Impact-T fort.X output

def load_fortX(filePath, keys):
    data = {}
    #Load the data 
    fortdata = np.loadtxt(filePath)
    for count, key in enumerate(keys):
        data[key] = fortdata[:,count]
    return data

FORT_KEYS = {}
FORT_UNITS = {}

FORT_KEYS[18] = ['t', 'z', 'gamma', 'E_kinetic','beta', 'r_max','deltaGamma_rms']
FORT_UNITS[18] = ['s','m', '1',     'MeV',      '1',     'm',    '1']
def load_fort18(filePath, keys=FORT_KEYS[18]):
    ''' From impact manual v2:
        1st col: time (secs)
        2nd col: distance (m)
        3rd col: gamma
        4th col: kinetic energy (MeV) 5th col: beta
        6th col: Rmax (m) R is measured from the axis of pipe 
        7th col: rms energy deviation normalized by MC^2
    '''
    return load_fortX(filePath, keys)

FORT_KEYS[24] =  ['t', 'z', 'x_centroid', 'x_rms', 'GBx_centroid', 'GBx_rms', 'x_twiss','x_normemit']
FORT_UNITS[24] = ['s', 'm', 'm',          'm',     '1',            '1',       '??',     'm-rad' ]
def load_fort24(filePath, keys=FORT_KEYS[24]):
    '''From impact manual:
    fort.24, fort.25: X and Y RMS size information
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: centroid location (m)
    4th col: RMS size (m)
    5th col: Centroid momentum normalized by MC
    6th col: RMS momentum normalized by MC
    7th col: Twiss parameter
    8th col: normalized RMS emittance (m-rad)
    '''
    return load_fortX(filePath, keys)
    
FORT_KEYS[25] = ['t', 'z', 'y_centroid', 'y_rms','GBy_centroid', 'GBy_rms', 'y_twiss','y_normemit']    
FORT_UNITS[25] = FORT_UNITS[24]
def load_fort25(filePath, keys=FORT_KEYS[25]):
    '''
    Same columns as fort24, Y RMS
    '''
    return load_fortX(filePath, keys)


FORT_KEYS[26] = ['t', 'z_centroid','z_rms','GBz_centroid', 'GBz_rms', 'z_twiss','z_normemit']
FORT_UNITS[26]= FORT_UNITS[25]
def load_fort26(filePath, keys=FORT_KEYS[26]):
    '''From impact manual:
    fort.26: Z RMS size information
    1st col: time (secs)
    2nd col: centroid location (m)
    3rd col: RMS size (m)
    4th col: Centroid momentum normalized by MC 5th col: RMS momentum normalized by MC
    6th col: Twiss parameter
    7th col: normalized RMS emittance (m-rad)
    '''
    return load_fortX(filePath, keys)

FORT_KEYS[27] =  ['t', 'z', 'x_max','GBx_max','y_max', 'GBy_max', 'z_max','GBz_max']
FORT_UNITS[27] = ['s', 'm', 'm',    '1',      'm',     '1',       'm',    '1']
def load_fort27(filePath, keys=FORT_KEYS[27]):
    '''
    fort.27: maximum amplitude information
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: Max. X (m)
    4th col: Max. Px (MC)
    5th col: Max. Y (m)
    6th col: Max. Py (MC)
    7th col: Max. Z (m) (with respect to centroid) 
    8th col: Max. Pz (MC)
    '''
    return load_fortX(filePath, keys)

FORT_KEYS[28] =  ['t', 'z', 'numparticles_min', 'numparticles_max','numparticles']
FORT_UNITS[28] = ['s', 'm', '1',                '1',               '1' ]
def load_fort28(filePath, keys=FORT_KEYS[28] ):
    '''From impact manual:
    fort.28: load balance and loss diagnostic
    1st col: time (secs)
    2nd col: z distance (m)
    3rd col: min # of particles on a PE
    4th col: max # of particles on a PE
    5th col: total # of particles in the bunch
    '''
    return load_fortX(filePath, keys)
   
FORT_KEYS[29] =  ['t', 'z', 'x_moment3','GBx_moment3','y_moment3', 'GBy_moment3', 'z_moment3','GBz_moment3']    
FORT_UNITS[29] = ['s', 'm', 'm',        '1',          'm',         '1',           'm',        '1']
def load_fort29(filePath, keys=FORT_KEYS[29]):
    '''
    fort.29: cubic root of 3rd moments of the beam distribution
    1st col: time (secs) 
    2nd col: z distance (m) 
    3rd col: X (m)
    4th col: Px (MC)
    5th col: Y (m)
    6th col: Py (MC)
    7th col: Z (m)
    8th col: Pz (MC)
    '''            
    return load_fortX(filePath, keys)

FORT_KEYS[30] =  ['t', 'z', 'x_moment4','GBx_moment4','y_moment4', 'GBy_moment4', 'z_moment4','GBz_moment4']
FORT_UNITS[30] = FORT_UNITS[29]
def load_fort30(filePath, keys=FORT_KEYS[30] ):
    '''
    fort.30: Fourth root of 4th moments of the beam distribution
    1st col: time (secs) 2nd col: z distance (m) 3rd col: X (m)
    4th col: Px (MC)
    5th col: Y (m)
    6th col: Py (MC)
    7th col: Z (m)
    8th col: Pz (MC)
    '''            
    return load_fortX(filePath, keys)




def load_fort60_and_70(filePath):
    """
    1st col: bunch length (m)
    2nd col: number of macroparticles per cell
    3rd col: current profile
    4th col: x slice emittance (m-rad)
    5th col: y slice emittance (m-rad)
    6th col: energy spread per cell without taking out correlation (eV)
    7th col: uncorrelated energy spread per cell (eV)
    """
    fortdata = np.loadtxt(filePath)
    data = {}
    keys = ['slice_z', 
            'particles_per_cell',
            'current',
            'x_emittance',
            'y_emittance',
            'energy_spread',
            'uncorreleted_energy_spread']
    for count, key in enumerate(keys):
        data[key] = fortdata[:,count]
    return data
       
    
# Wrapper functions to provide keyed output    
    
def load_fort40(filePath):
    """
    Returns dict with 'initial_particles'
    """
    data = parse_impact_particles(filePath)
    return {'initial_particles':data}    

def load_fort50(filePath):
    """
    Returns dict with 'final_particles'
    """
    data = parse_impact_particles(filePath)
    return {'final_particles':data}    
    
def load_fort60(filePath):
    """
    Returns dict with 'initial_particle_slices'
    """
    data = load_fort60_and_70(filePath)
    return {'initial_particle_slices':data}

def load_fort70(filePath):
    """
    Returns dict with 'final_particle_slices'
    """
    data = load_fort60_and_70(filePath)
    return {'final_particle_slices':data}



def fort_files(path):
    """
    Find fort.X fliles in path
    """
    assert os.path.isdir(path)
    flist = os.listdir(path)
    fortfiles=[]
    for f in flist:
        if f.startswith('fort.'):
            fortfiles.append(os.path.join(path,f))
    return fortfiles    
    
    
def fort_type(filePath):
    """
    Extract the integer type of a fort.X file, where X is the type.
    """
    fullpath = os.path.abspath(filePath)
    p, f = os.path.split(fullpath)
    s = f.split('.')
    if s[0] != 'fort':
        print('Error: not a fort file:', filePath)
    else:
        return int(s[1])    


FORT_DESCRIPTION = {
    18:'Time and energy',
    24:'RMS X information',
    25:'RMS Y information',
    26:'RMS Z information',
    27:'Max amplitude information',
    28:'Load balance and loss diagnostics',
    29:'Cube root of third moments of the beam distribution',
    30:'Fourth root of the fourth moments of the beam distribution',
    34:'Dipole ONLY: X output information in dipole reference coordinate system',
    35:'Dipole ONLY: Y output information in dipole reference coordinate system',
    36:'Dipole ONLY: Z output information in dipole reference coordinate system ',
    37:'Dipole ONLY: Maximum amplitude information in dipole reference coordinate system',
    38:'Dipole ONLY: Reference particle information in dipole reference coordinate system',
    40:'initial particle distribution at t = 0',
    50:'final particle distribution projected to the centroid location of the bunch',
    60:'Slice information of the initial distribution',
    70:'Slice information of the final distribution'
    
    
}





FORT_LOADER = {
    18:load_fort18,
    24:load_fort24,
    25:load_fort25,
    26:load_fort26,
    27:load_fort27,
    28:load_fort28,
    29:load_fort29,
    30:load_fort30,
    40:load_fort40,
    50:load_fort50,
    60:load_fort60,
    70:load_fort70
}

# Form large unit dict for these types of files
UNITS = {}
for i in [18, 24, 25, 26, 27, 28, 29, 30]:
    for j, k in enumerate(FORT_KEYS[i]):
        UNITS[k] = FORT_UNITS[i][j]
    
def fort_type(filePath):
    """
    Extract the integer type of a fort.X file, where X is the type.
    """
    fullpath = os.path.abspath(filePath)
    p, f = os.path.split(fullpath)
    s = f.split('.')
    if s[0] != 'fort':
        print('Error: not a fort file:', filePath)
    else:
        type = int(s[1])    
        if type not in FORT_DESCRIPTION:
            print('Warning: unknown fort type for:', filePath)
        if type not in FORT_LOADER:
            print('Warning: no fort loader yet for:', filePath)
        return type


def load_fort(filePath, verbose=True):
    """
    Loads a fort file, automatically detecting its type and selecting a loader.
    
    """
    type = fort_type(filePath)
    
    if verbose:
        if type in FORT_DESCRIPTION:
            print('Loaded fort', type,':', FORT_DESCRIPTION[type])
        else:
            print('unknown type:',type)
            
    if type in FORT_LOADER:
        dat = FORT_LOADER[type](filePath)
    else:
        print('ERROR: need parser for:', f)
    return dat    
    

FORT_STAT_TYPES     = [18, 24, 25, 26, 27, 28, 29, 30]
FORT_PARTICLE_TYPES = [40,50]
FORT_SLICE_TYPES    = [60,70]
   
def load_many_fort(path, types=FORT_STAT_TYPES, verbose=False):
    """
    Loads a large dict with data from many fort files.
    Checks that keys do not conflict.
    
    Default types are for typical statistical information along the simulation path. 
    
    """
    fortfiles=fort_files(path)
    alldat = {}
    for f in fortfiles:
        type = fort_type(f)
        if type not in types:
            continue
        
        dat = load_fort(f, verbose=verbose)
        for k in dat:
            if k not in alldat:
                alldat[k] = dat[k]
            else:
                # Check that this data is the same as what's already in there
                assert np.all(alldat[k] == dat[k]), 'Conflicting data for key:'+k
        
    return alldat    


