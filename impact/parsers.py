import numpy as np


#-----------------
# Parsing ImpactT input file


#-----------------
# ImpactT Header
# lattice starts at line 10

# Line 1
names1  = ['Npcol', 'Nprow']
types1 = [int, int]

# Line 2
names2 = ['Dt', 'Ntstep', 'Nbunch']
types2 = [float, int, int]

# Line 3
names3 = ['Dim', 'Np', 'Flagmap', 'Flagerr', 'Flagdiag', 'Flagimg', 'Zimage']
types3 = [int, int, int, int, int, int, float];

# Line 4
names4 = ['Nx', 'Ny', 'Nz', 'Flagbc', 'Xrad', 'Yrad', 'Perdlen']
types4 = [int, int, int, int, float, float, float]

# Line 5
names5 = ['Flagdist', 'Rstartflg', 'Flagsbstp', 'Nemission', 'Temission']
types5 = [int, int, int, int, float ]

# Line 6-8
names6 = ['sigx(m)', 'sigpx', 'muxpx', 'xscale', 'pxscale', 'xmu1(m)', 'xmu2']
types6 = [float for i in range(len(names6))]
names7 = ['sigy(m)', 'sigpy', 'muxpy', 'yscale', 'pyscale', 'ymu1(m)', 'ymu2']
types7 = types6
names8 = ['sigz(m)', 'sigpz', 'muxpz', 'zscale', 'pzscale', 'zmu1(m)', 'zmu2']
types8 = types6

# Line 9
names9 = ['Bcurr', 'Bkenergy', 'Bmass', 'Bcharge', 'Bfreq', 'Tini']
types9 = [float for i in range(len(names9))]

allnames = [names1, names2, names3, names4, names5, names6, names7, names8, names9]
alltypes = [types1, types2, types3, types4, types5, types6, types7, types8, types9]

#-----------------
# Some help for keys above
help = {}
help['Npcol'] =  'Number of columns of processors, used to decompose domain along Y dimension.'
help['Npro1w'] =  'Number of rows of processors, used to decompose domain along Z dimension.'
help['Dt'] =  'Time step size (secs).'
help['Ntstep'] = 'Maximum number of time steps.'



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
        d.update(parse_line(x[i], allnames[i], alltypes[i]))
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
    v += [v1[0], v1[1], v1[3], v1[4], v1[5]] # solrd doesn't have z_offset
    
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
   
    
#-----------------------------------------------------------------      
def parse_offset_beam(line):
    """
    offset_beam (type -1)
    
    If btype = −1, steer the transverse beam centroid at given location V2(m) to position 
    x offset V3(m)
    Px (γβx) offset V4
    y offset V5(m)
    Py (γβy) offset V6
    z offset V7(m)
    Pz (γβz) offset V8.
    """

    v = line.split()[3:-2] # V data starts with index 4, ends with '/'
    d={}
    ##print (v, len(v))
    d['z']  = float(v[2]) 
    olist = ['x_offset', 'px_offset', 'y_offset','py_offset','z_offset','pz_offset' ]
    for i in range(6):
        if i+3 > len(v)-1:
            val = 0.0
        else:
            print('warning: offset_beam missing numbers. Assuming zeros', line)
            val = float(v[i+3])
        d[olist[i]] = val
    return d


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
    v = [ele, 0.0, 0.0, ele['s'], ele['dt']]   
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



def parse_stop(line):
    """
    Stop (type -99)
    
    """
    v = line.split()[3:] # V data starts with index 4
    d={'s':float(v[3])}
    return d


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
        d['is_on'] = 'True'
    else:
        d['is_on'] = 'False'

    return d


#-----------------------------------------------------------------  

#def parse_bpm(line):
#    """
#    BPM ????
#    """
#    return {}
#

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
               'wakefield':parse_wakefield,
               'spacecharge':parse_spacecharge,
               'write_beam_for_restart':parse_write_beam_for_restart,
               'stop':parse_stop
              }  

def parse_ele(line):
    """
    Parse an Impact-T lattice line. 
    
    Returns an ele dict.

    
    """
    ##print(line)
    x = line.split()
    e = {}
    if is_commented(line):
        return {'type':'comment', 'comment':line, 'L':0}
    
    e['original'] = line # Save original line
    e['L'] = float(x[0])
    e['itype'] = int(x[3])
    if e['itype'] < 0:
        e['nseg'] = int(x[1]) # Only for type < 0
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
    """
    #s0 = -2.1459294
    s = s0
    for e in elelist:
        s = s + e['L']
        e['s'] = s

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