#import numpy as np
from .parsers import drift_v, quadrupole_v, solrf_v, change_timestep_v, wakefield_v, stop_v, write_beam_v, offset_beam_v, dipole_v, spacecharge_v, write_beam_for_restart_v, itype_of
        
import numpy as np    
        
#-----------------------------------------------------------------  
# Print eles in a MAD style syntax
def ele_str(e):
    line = ''
    if e['type']=='comment':
        c = e['comment']
        if c == '!':
            return ''
        else:
            #pass
            return c
    
        
    line = e['name']+': '+e['type']
    l = len(line)
    for key in e:
        if key in ['s', 'name', 'type', 'original', 'itype']: 
            continue
        val = str(e[key])
        s =  key+'='+val
        l += len(s)
        if l > 80:
            append = ',\n      '+s
            l = len(append)
        else:
            append = ', '+s
        line = line + append
    return line        






ele_v_function = {
    'dipole':dipole_v,
    'drift':drift_v,
    'quadrupole':quadrupole_v,
    'solrf':solrf_v,
    'stop':stop_v,
    'change_timestep':change_timestep_v,
    'offset_beam':offset_beam_v,
    'wakefield':wakefield_v,
    'write_beam':write_beam_v,
    'write_beam_for_restart':write_beam_for_restart_v,
    'spacecharge':spacecharge_v
                  
                 }    

def ele_line(ele):
    """
    Write Impact-T stype element line

    All real eles start with the four numbers:
    Length, Bnseg, Bmpstp, itype
    
    With additional numbers depending on itype.

    """
    type = ele['type']
    if type == 'comment':
        return ele['comment']
    itype = itype_of[type] 
    if itype < 0:
        Bnseg = ele['nseg']
        Bmpstp = ele['bmpstp']
    else:
        Bnseg = 0
        Bmpstp = 0
    dat = [ele['L'], Bnseg, Bmpstp, itype]
    
    if type in ele_v_function:
        v =  ele_v_function[type](ele)
        dat += v[1:]
    else:
        print('ERROR: ele_v_function not yet implemented for type: ', type)
        return
    
    line = str(dat[0])
    for d in dat[1:]:
        line = line + ' ' + str(d)
    return line + ' /' + '!name:'+ele['name']



def lattice_lines(eles):
    lines = []
    for e in eles:
        lines.append(ele_line(e))
    return lines

#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Higher level functions
def ele_dict_from(eles):
    """
    Use names as keys. Names must be unique.
    
    """
    ele_dict = {}
    for ele in eles:
        if ele['type'] == 'comment':
            continue
        name = ele['name']
        assert name not in ele_dict
        ele_dict[name] = ele
    return ele_dict


#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Layout
# Info for plotting

ELE_HEIGHT = {
    'change_timestep':1,
    'comment':1,
    'dipole':2,
    'drift':1,
    'offset_beam':1,
    'quadrupole':5,
    'solrf':3,
    'spacecharge':1,
    'stop':1,
    'wakefield':1,
    'write_beam':1,
    'write_beam_for_restart':1
}
ELE_COLOR = {
    'change_timestep':'black',
    'comment':'black',
    'dipole':'red',
    'drift':'black',
    'offset_beam':'black',
    'quadrupole':'blue',
    'solrf':'green',
    'spacecharge':'black',
    'stop':'black',
    'wakefield':'brown',
    'write_beam':'black',
    'write_beam_for_restart':'black'
}

def ele_shape(ele):
    """
    
    """
    type = ele['type']
    q_sign = -1 # electron
    
    factor = 1.0

    if type == 'quadrupole':
        b1 = q_sign*ele['b1_gradient']
        if b1 > 0:
            # Focusing
            top = b1
            bottom = 0
        else:
            top  = 0
            bottom = b1
    else:
        top =ELE_HEIGHT[type]
        bottom = -top
    
    c = ELE_COLOR[type]
    
    d = {}
    d['left'] = ele['s']-ele['L']
    d['right'] = ele['s']
    d['top'] = top
    d['bottom'] = bottom
    # Center points
    d['x'] =  ele['s']-ele['L']/2
    d['y'] = 0
    d['color'] = ELE_COLOR[type]
    d['name'] = ele['name']
    
    d['all'] = ele_str(ele)#'\n'.join(str(ele).split(',')) # Con
    d['description'] = ele['description']
    
    return d

def ele_shapes(eles):
    """
    Form dataset of al element info
    
    Only returns shapes for physical elements
    """
    # Automatically get keys
    keys = list(ele_shape(eles[0]))
    # Prepare lists
    data = {}
    for k in keys:
        data[k] = []
    for e in eles:
        type = e['type']
        if type in ['comment']:
            continue
        if itype_of[type] <0:
            continue
        d = ele_shape(e)
        for k in keys:
            data[k].append(d[k])
    return data


#-----------------------------------------------------------------  
#-----------------------------------------------------------------  
# Helpers

def sanity_check_ele(ele):
    """
    Sanity check that writing an element is the same as the original line
    """
    if ele['type'] == 'comment':
        return True
    
    dat1 = ele_line(ele).split('/')[0].split()
    dat2 = ele['original'].split('/')[0].split()
    
    itype = itype_of[ele['type']]
    if itype >=0:
        # These aren't used
        dat2[1]=0
        dat2[2]=0
    if itype in [ itype_of['offset_beam']]:
        # V1 is not used
        dat2[4]=0      
    if itype in [ itype_of['spacecharge']]:
        # V1 is not used, only v2 sign matters
        dat2[4]=0  
        if float(dat2[5]) >0:
            dat2[5]=1.0
        else:
            dat2[5]=-1.0            
        
    if itype in [ itype_of['write_beam'], itype_of['stop'], itype_of['write_beam_for_restart'] ]:
        # Only V3 is used
        dat2[4]=0
        dat2[5]=0
    if itype in [itype_of['change_timestep']]:
        # Only V3, V4 is used
        dat2[4]=0
        dat2[5]=0
        
        
    dat1 = np.array([float(x) for x in dat1])
    dat2 = np.array([float(x) for x in dat2])
    if len(dat1) != len(dat2):
        #print(ele)
        #print('bad lengths:')
        #print(dat1)
        #print(dat2)
        return True
    good = np.all(dat2-dat1 ==0)
    
    if not good:
        print('------ Not Good ----------')
        print(ele)
        print('This    :', dat1)
        print('original:', dat2)
    
    return good

    