#import numpy as np

        
        
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