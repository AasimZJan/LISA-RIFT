from RIFT.LISA.injections.LISA_injections import *
import os
import numpy as np
param_dict = {}


param_dict.update({'m1':15e6,
                   'm2':15e6,
                   's1z':0.0,
                   's2z':0.0,
                   'eccentricity':0.1,
                   'meanPerAno':0.1,
                   'beta':np.pi/3,
                   'lambda':np.pi/4,
                   'dist':274.566 * 1e3,
                   'inclination':0.0,
                   'phi_ref':5*np.pi/4,
                   'psi':np.pi/6,
                   'tref':2621440.0,
                   'deltaT':5,
                   'deltaF':float(1.0/(2621440.0*2)),
                   'fmin':0.00004,
                   'fref':None,
                   'wf-fref':0.00004,
                   'approx':"SEOBNRv5EHM",
                   'modes':[(2,2),(3,2),(4,4)],
                   'save_path':f"{os.getcwd()}/frames",
                   'path_to_NR_hdf5':None,
                   'NR_taper_percent':2.5,
                   'psd_path':f"{os.getcwd()}",
                   'snr_fmin':0.0001,
                   'snr_fmax':0.1})

data_dict = generate_lisa_TDI_dict(param_dict)
generate_lisa_injections(data_dict, param_dict, get_snr=True)
