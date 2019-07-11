import numpy as np



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
    """
    z0 = fdat['z0']  # Not used?
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

    kk = 2*np.pi*(z-zmid) / zlen
    
    ilist = np.arange(ncoefreal)+1
    
    res = Fcoef0/2 + np.sum(Fcoef1* np.cos(ilist*kk))  + np.sum(Fcoef2* np.sin(ilist*kk))

    return  res