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
    data = h5['data']
    
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