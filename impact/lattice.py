#import numpy as np
from .parsers import drift_v, quadrupole_v, solrf_v, change_timestep_v, itype_of
        
        
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






ele_v_function = {'drift':drift_v,
                  'quadrupole':quadrupole_v,
                  'solrf':solrf_v,
                  'change_timestep':change_timestep_v}    
"""
Write Impact-T stype element line

"""
def ele_line(ele):
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
    return line + ' /'