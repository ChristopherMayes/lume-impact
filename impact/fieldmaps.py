import numpy as np
import os

def write_fieldmap(filePath, fieldmap):
    """
    Master routine for writing a fieldmap
    """
    
    # Look for symlink
    if 'filePath' in fieldmap:
        write_fieldmap_symlink(fieldmap, filePath)
        return
    
    format = fieldmap['info']['format']
    if format == 'rfdata':
        write_fieldmap_rfdata(filePath, fieldmap)
    elif format == 'solenoid_T7':
        write_solenoid_fieldmap(fieldmap, filePath)
    else:
        print('Missing writer for fieldmap:', fieldmap)
        raise
        
        
# Simple routines for symlinking fieldmaps

def read_fieldmap_symlink(filePath):
    return {'filePath':os.path.abspath(filePath)}
    
def write_fieldmap_symlink(fieldmap, filePath):
    if os.path.exists(filePath):
        return
        # do nothing
    else:
        os.symlink(fieldmap['filePath'], filePath)

    
# -----------------------
# rfdata fieldmaps
def read_fieldmap_rfdata(filePath):
    """
    Read Impact-T rfdata file, which should be simple two-column ASCII data
    """
    
    info = {}
    info['format'] = 'rfdata'
    info['filePath'] = os.path.abspath(filePath) 
    
    # Read data
    d = {}
    d['info'] = info
    d['data'] = np.loadtxt(filePath)
    return d
    
    
def write_fieldmap_rfdata(filePath, fieldmap):
    """
    
    """
    np.savetxt(filePath, fieldmap['data'])

    
    
    
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

    
def read_fieldmap_h5(h5):
    """
    
    """
    if 'filePath' in h5.attrs:
        return {'filePath':h5.attrs['filePath']}
    
    info = dict(h5['info'].attrs)
    data = h5['data'][:]
    
    return {'info':info, 'data':data}    
    
    
    
    
    
    
# -----------------------
# Ty fieldmaps
def read_solenoid_fieldmap(filePath):
    """
    Read a T7 style file.
    
    Format:
    
    Header:
        zmin, zmax, nz
        rmin, rmax, nr
    Data:
        Br, Bz
        (repeating)
        
    min, max are in cm
    Er 
    
    """
    d = {}
    # Read header 
    with open(filePath) as f:
        line1 = f.readline()
        line2 = f.readline()
    zmin, zmax, nz = line1.split()
    rmin, rmax, nr = line2.split()

    info = {}
    info['format'] = 'solenoid_T7'
    info['zmin'] = float(zmin)
    info['zmax'] = float(zmax)
    info['nz'] = int(nz)
    info['rmin'] = float(rmin)
    info['rmax'] = float(rmax)
    info['nr'] = int(nr)
    
    # Read data
    d['info'] = info
    d['data'] = np.loadtxt(filePath, skiprows=2)
    
    return d

def write_solenoid_fieldmap(fieldmap, filePath):
    """
    Save fieldmap data to file in T7 format.
    fieldmap must have:
    ['info'] = dict with keys: zmin, zmax, nz, rmin, rmax, nr
    ['data'] = array of data
    """
    info = fieldmap['info']
    line1 = ' '.join([str(x) for x in [info['zmin'], info['zmax'], info['nz']]])
    line2 = ' '.join([str(x) for x in [info['rmin'], info['rmax'], info['nr']]])
    header = line1+'\n'+line2
    # Save data
    np.savetxt(filePath, fieldmap['data'], header=header, comments='')




def process_fieldmap_solrf(data):
    """
    Processes array of raw data from a rfdataX file into a dict.
    
    From the documentation:
    Here, the rfdataV5 file contains the Fourier coefficients for both E fields and B fields.
    The first half contains E fields, and the second half contains B fields.
    See manual. 
    
    fourier_coeffients are as Impact-T prefers: 
    [0] is
    [1::2] are the cos parts
    [2::2] are the sin parts 
    Recontruction of the field at z shown in :
        fieldmap_reconsruction
    
    """
    
    d = {}
    d['Ez'] = {}
    d['Bz'] = {}
    
    # Ez
    n_coef = int(data[0])  # Number of Fourier coefs of on axis
    d['Ez']['z0'] = data[1] # distance before the zedge.
    d['Ez']['z1'] = data[2] # distance after the zedge.
    d['Ez']['L']  = data[3] # length of the Fourier expanded field.
    i1 = 4
    i2 = 4+n_coef
    d['Ez']['fourier_coefficients'] = data[i1:i2] # Fourier coefficients on axis
    
    # Bz
    data2 = data[i2:]
    n_coef = int(data2[0])  # Number of Fourier coefs of on axis
    d['Bz']['z0'] = data2[1] # distance before the zedge.
    d['Bz']['z1'] = data2[2] # distance after the zedge.
    d['Bz']['L']  = data2[3] # length of the Fourier expanded field.
    i1 = 4
    i2 = 4+n_coef
    d['Bz']['fourier_coefficients'] = data2[i1:i2] # Fourier coefficients on axis
    
    return d
    
#process_fieldmap_solrf(I.input['fieldmaps']['rfdata102'])  

def fieldmap_reconsruction(fdat, z):
    """
    Transcription of Ji's routine
    
    Field at z relative to the element's zedge
    
    """
    z0 = fdat['z0']  
    z1 = fdat['z1']  # Not used?
    
    zlen = fdat['L']
    
    if zlen == 0:
        return 0
    
    rawdata = fdat['fourier_coefficients']
    
    ncoefreal = (len(rawdata) -1)//2

    zmid = zlen/2
    
    Fcoef0 = rawdata[0]    # constant factor
    Fcoef1 = rawdata[1::2] # cos parts
    Fcoef2 = rawdata[2::2] # sin parts

    kk = 2*np.pi*(z-zmid -z0) / zlen
    
    ilist = np.arange(ncoefreal)+1
    
    res = Fcoef0/2 + np.sum(Fcoef1* np.cos(ilist*kk))  + np.sum(Fcoef2* np.sin(ilist*kk))

    return  res




def riffle(a, b):
    return np.vstack((a,b)).reshape((-1,),order='F')


def create_fourier_coefficients(zdata, edata, n=None):
    """
    Literal transcription of Ji's routine RFcoeflcls.f90
    
    Fixes bug with scaling the field by the max or min seen.
    
    Vectorized two loops
    
    """
    ndatareal=len(zdata)
    
    # Cast to np arrays for efficiency
    zdata = np.array(zdata)
    edata = np.array(edata)
    
    # Proper scaling
    e_min = edata.min()
    e_max = edata.max()
    if abs(e_min) > abs(e_max):
        scale = e_min
    else:
        scale = e_max
    edata /=  scale
    
    if not n:
        n = len(edata)

    Fcoef = np.zeros(n)
    Fcoef2 = np.zeros(n)
    zlen = zdata[-1] - zdata[0]
    
    zhalf = zlen/2.0
    zmid = (zdata[-1]+zdata[0])/2
    h = zlen/(ndatareal-1)
    
    pi = np.pi
    print("The RF data number is: ",ndatareal,zlen,zmid,h)
    
    jlist = np.arange(n)
    
    zz = zdata[0] - zmid
    Fcoef  = (-0.5*edata[0]*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf
    Fcoef2 = (-0.5*edata[0]*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf
    zz = zdata[-1] - zmid
    Fcoef  += -(0.5*edata[-1]*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf          
    Fcoef2 += -(0.5*edata[-1]*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf
        

    for i in range(ndatareal):
        zz = i*h+zdata[0]
        klo=0
        khi=ndatareal-1
        while (khi-klo > 1):
            k=(khi+klo)//2
            if(zdata[k] - zz > 1e-15):
                khi=k
            else:
                klo=k

        hstep=zdata[khi]-zdata[klo]
        slope=(edata[khi]-edata[klo])/hstep
        ez1 =edata[klo]+slope*(zz-zdata[klo])
        zz = zdata[0]+i*h - zmid

        Fcoef += (ez1*np.cos(jlist*2*pi*zz/zlen)*h)/zhalf
        Fcoef2 += (ez1*np.sin(jlist*2*pi*zz/zlen)*h)/zhalf

    return np.hstack([Fcoef[0], riffle(Fcoef[1:], Fcoef2[1:])]) 
