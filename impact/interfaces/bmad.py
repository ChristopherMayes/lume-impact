from pmd_beamphysics import FieldMesh
from pmd_beamphysics.fields.analysis import accelerating_voltage_and_phase
import numpy as np

def ele_info(tao, ele_name):
    edat = tao.ele_head(ele_name)
    edat.update(tao.ele_gen_attribs(ele_name))
    s = edat['s']
    L = edat['L']
    edat['s_begin'] = s-L
    edat['s_center'] = (s + edat['s_begin'])/2    
    
    return edat

def tao_create_impact_solrf_ele(tao,
                            ele_name,
                            *, 
                            style='fourier',
                            n_coef=30,
                            spline_s=1e-6,
                            spline_k = 5,
                            file_id=666, 
                            output_path=None,
                            cache=None
                               ):
    """
    Create an Impact-T solrf element from a running PyTao Tao instance.
    
    Parameters
    ----------
    
    tao: Tao object
    
    ele_name: str
    
    style: str, default: 'fourier'
    
    zmirror: bool, default: None
        Mirror the field about z=0. This is necessary for non-periodic field such as electron guns.
        If None, will autmatically try to detect whether this is necessary.
        
    spline_s: float, default: 0
    
    spline_k: float, default: 0
    
    file_id: int, default: 666
    
    output_path: str, default: None
        If given, the rfdata{file_id} file will be written to this path    
    
    cache: dict, default: None
        FieldMesh file cache dict: {filename:FieldMesh(filename)}
        If not none, this will cache fieldmaps and update this dict.
        
    
    Returns
    -------
    dict with:
      line: str
          Impact-T style element line
          
      ele: dict
          LUME-Impact style element
          
      fmap: dict with:
            data: ndarray
            
            info: dict with
                Ez_scale: float
            
                Bz_scale: float
            
                Ez_err: float, optional
                
                Bz_err: float, optional
            
            field: dict with
                Bz: 
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'
                Ez: 
                    z0: float
                    z1: float
                    L: float
                    fourier_coefficients: ndarray 
                        Only present when style = 'fourier'
                    derivative_array: ndarray
                        Only present when style = 'derivatives'
    
    
    """
    
    
    # Ele info from Tao
    edat = ele_info(tao, ele_name)
    
    # FieldMesh
    grid_params = tao.ele_grid_field(ele_name, 1, 'base', as_dict=False)
    field_file = grid_params['file'].value   
    if cache is not None:
        if field_file in cache:
            field_mesh = cache[field_file]
        else:
            # Add to cache
            field_mesh = FieldMesh(field_file)   
            cache[field_file] = field_mesh
            
    else:
        field_mesh = FieldMesh(field_file)   
    
    ele_key = edat['key'].upper() 
    freq = edat.get('RF_FREQUENCY', 0)
    assert np.allclose(freq, field_mesh.frequency), f'{freq} != {field_mesh.frequency}'    
    
    #master_parameter = field_mesh.attrs.get('masterParameter', None)
    master_parameter = grid_params['master_parameter'].value
    if master_parameter == '<None>':
        master_parameter = None
    
    if ele_key == 'E_GUN':
        zmirror = True
    else:
        zmirror = False
        
    L_fm = field_mesh.dz * (field_mesh.shape[2]-1)
    z0 = field_mesh.coord_vec('z')
    
    
    # Find zedge
    eleAnchorPt = field_mesh.attrs['eleAnchorPt']
    if eleAnchorPt == 'beginning':
        zedge = edat['s_begin']
    elif eleAnchorPt == 'center':
        # Use full fieldmap!!!
        zedge = edat['s_center'] - L_fm/2
    else:
        raise NotImplementedError(f'{eleAnchorPt} not implemented')        
        
        
    outdat = {}
    info = outdat['info'] = {}
        
    # Phase and scale
    if ele_key == 'SOLENOID':
        assert  master_parameter is not None
        scale = edat[master_parameter]   
        
        bfactor = np.abs(field_mesh.components['magneticField/z'][0,0,:]).max() 
        if not np.isclose(bfactor, 1):
            scale *= bfactor
        phi0_tot = 0
        phi0_oncrest = 0
        
    elif ele_key in ('E_GUN', 'LCAVITY'):
        if master_parameter is None:
            scale = edat['FIELD_AUTOSCALE']
        else:
            scale = edat[master_parameter]
        
        Ez0 = field_mesh.components['electricField/z'][0,0,:]
        efactor = np.abs(Ez0).max()               
        if not np.isclose(efactor, 1):
            scale *= efactor
        
        # Get ref_time_start
        ref_time_start = tao.ele_param(ele_name, 'ele.ref_time_start')['ele_ref_time_start']
        phi0_ref = freq*ref_time_start
        
        #phi0_fieldmap = field_mesh.attrs['RFphase'] / (2*np.pi) # Bmad doesn't use at this point
        phi0_fieldmap = grid_params['phi0_fieldmap'].value 
        
        
        phi0_user = sum([edat['PHI0'], edat['PHI0_ERR'] ])
        phi0_oncrest = sum([edat['PHI0_AUTOSCALE'], phi0_fieldmap, -phi0_ref]) 
        phi0_tot =  (phi0_oncrest + phi0_user) % 1
        theta0_deg = phi0_tot * 360  
        
        # Useful info for scaling
        acc_v0, acc_phase0 = accelerating_voltage_and_phase(z0, Ez0/np.abs(Ez0).max(), field_mesh.frequency)
        print(f"v=c acceleating voltage per max field {acc_v0} (V/(V/m))")
        
        # Add phasing info
        info['v=c acceleating voltage per max field'] = acc_v0
        info['phi0_oncrest'] = phi0_oncrest % 1
            

    else:
        raise NotImplementedError
    
    # Call the fieldmesh method
    dat = field_mesh.to_impact_solrf(
                                    zedge=zedge,
                                   name=ele_name,
                                   scale=scale,
                                   phase=phi0_tot*(2*np.pi),
                                   style=style,
                                   n_coef=n_coef,
                                   spline_s=spline_s,
                                    spline_k = spline_k,
                                  x_offset = edat['X_OFFSET'],
                                  y_offset = edat['Y_OFFSET'],
                                   zmirror=zmirror,
                                   file_id=file_id,
                                output_path=output_path)
    # Add this to output
    outdat.update(dat)
    

    return outdat


def tao_create_impact_quadrupole_ele(tao, ele_name):
    """
    Create an Impact-T quadrupole element from a running PyTao Tao instance.
    
    Parameters
    ----------
    
    tao: Tao object
    
    ele_name: str    
    
    Returns
    -------
        dict with:
        line: str
            Impact-T style element line
            
        ele: dict
            LUME-Impact style element
        
    """
    
    edat = ele_info(tao, ele_name)
    L_eff = edat['L']
    L = 2*L_eff # Account for some fringe
    radius = edat['X1_LIMIT']
    assert radius > 0
    
    zedge = edat['s_center'] - L/2
    b1_gradient = edat['B1_GRADIENT']
    x_offset = edat['X_OFFSET']
    y_offset = edat['Y_OFFSET']  
    tilt = edat['TILT']  

    ele = {
     'L': L, 
     'type': 'quadrupole',
     'zedge': zedge, 
     'b1_gradient': b1_gradient,
      'L_effective': L_eff,
      'radius': radius,
      'x_offset': x_offset,
      'y_offset': y_offset,
      'x_rotation': 0.0,
      'y_rotation': 0.0,
      'z_rotation': tilt,
     's': edat['s'],
     'name': ele_name}
    
    line = f"{L} 0 0 1 {zedge} {L_eff} {radius} {x_offset} {y_offset} 0 0 0 "
    
    return {
         'ele': ele,
         'line': line
            }



