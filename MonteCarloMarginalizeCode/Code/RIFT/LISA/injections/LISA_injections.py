import numpy as np
import RIFT.lalsimutils as lsu
from RIFT.LISA.response.LISA_response import *
import lal
import lalsimulation
import matplotlib.pyplot as plt
import os
import h5py
import json
__author__ = "A. Jan"

###########################################################################################
# FOR INJECTIONS
###########################################################################################
def create_lisa_injections(hlmf, fmax, fref, beta, lamda, psi, inclination, phi_ref, tref, provide_full_output = False):
    print(f"create_lisa_injections function has been called with following arguments:\n{locals()}")
    tf_dict, f_dict, amp_dict, phase_dict = get_tf_from_phase_dict(hlmf, fmax=fmax, fref=fref)
    A = 0.0
    E = 0.0
    T = 0.0
    modes = list(hlmf.keys())
    response = {}
    mode_TDI = {}
    mode_content = {}
    for mode in modes:
        H_0 = transformed_Hplus_Hcross(beta, lamda, psi, inclination, -phi_ref, mode[0], mode[1]) 
        L1, L2, L3 = Evaluate_Gslr(tf_dict[mode] + tref, f_dict[mode], H_0, beta, lamda)
        time_shifted_phase = phase_dict[mode] + 2*np.pi*tref*f_dict[mode]
        tmp_data = amp_dict[mode] * np.exp(1j*time_shifted_phase)  
        # I belive BBHx conjugates because the formalism is define for A*exp(-1jphase), but I need to check with ROS and Mike Katz.
        A += np.conj(tmp_data * L1)
        E += np.conj(tmp_data * L2)
        T += np.conj(tmp_data * L3)
        response[mode], mode_TDI[mode]  = {}, {}
        response[mode]["L1"], response[mode]["L2"], response[mode]["L3"] = np.conj(L1), np.conj(L2), np.conj(L3)
        mode_TDI[mode]["L1"], mode_TDI[mode]["L2"], mode_TDI[mode]["L3"] = np.conj(tmp_data*L1), np.conj(tmp_data*L2), np.conj(tmp_data*L3)
        mode_content[mode] = np.conj(tmp_data)
    A_lal, E_lal, T_lal = create_lal_frequency_series(f_dict[modes[0]], A, hlmf[modes[0]].deltaF), create_lal_frequency_series(f_dict[modes[0]], E, hlmf[modes[0]].deltaF), create_lal_frequency_series(f_dict[modes[0]], T, hlmf[modes[0]].deltaF)
    data_dict = {}
    data_dict["A"], data_dict["E"], data_dict["T"] = A_lal, E_lal, T_lal
    if provide_full_output:
        return data_dict, response, mode_TDI, mode_content
    else:
        return data_dict

def load_psd(param_dict):
    """
    Load Power Spectral Density (PSD) data for the LISA instrument. 

    Parameters:
    param_dict (dict): A dictionary containing parameters for loading the PSD, 
                       should contain 'psd_path', 'deltaF', and 'snr_fmin'.

    Returns:
    dict: A dictionary containing PSD for A, E, and T channels.
    """
    print(f"Reading PSD to calculate SNR for LISA instrument from {param_dict['psd_path']}.")
    psd = {}
    psd["A"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"] + "/A-psd.xml.gz", "A")
    psd["A"] = lsu.resample_psd_series(psd["A"],  param_dict['deltaF'])
    psd_fvals = psd["A"].f0 + param_dict['deltaF']*np.arange(psd["A"].data.length)
    psd["A"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0 

    psd["E"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"]+ "/E-psd.xml.gz", "E")
    psd["E"] = lsu.resample_psd_series(psd["E"],  param_dict['deltaF'])
    psd_fvals = psd["E"].f0 + param_dict['deltaF']*np.arange(psd["E"].data.length)
    psd["E"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0

    psd["T"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"]+ "/T-psd.xml.gz", "T")
    psd["T"] = lsu.resample_psd_series(psd["T"],  param_dict['deltaF'])
    psd_fvals = psd["T"].f0 +  param_dict['deltaF']*np.arange(psd["T"].data.length)
    psd["T"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0
    return psd

def calculate_snr(data_dict, fmin, fmax, fNyq, psd, save_path, only_positive_modes=True):
    """
    Calculate the zero-noise Signal-to-Noise Ratio (SNR) for LISA signals.

    Parameters:
    data_dict (dict): A dictionary containing the A, E, and T signal data,
    fmin (float): The minimum frequency for integration in Hz,
    fmax (float): The maximum frequency for integration in Hz,
    fNyq (float): The Nyquist frequency in Hz,
    psd (dict): A dictionary containing the PSD for A, E, and T channels.
    save_path (str): where to save snr dict

    Returns:
    float: The total zero-noise SNR calculated across all channels.
    """

    assert data_dict["A"].deltaF == data_dict["E"].deltaF == data_dict["T"].deltaF
    print(f"Integrating from {fmin} to {fmax} Hz.")

    # create instance of inner product 
    IP_A = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["A"], False, False, 0.0,)
    IP_E = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["E"], False, False, 0.0,)
    IP_T = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["T"], False, False, 0.0,)
    
    IP_factor = 1
    if only_positive_modes:
        IP_factor = 2

    # calculate SNR of each channel 
    A_snr, E_snr, T_snr = np.sqrt(IP_factor*IP_A.ip(data_dict["A"], data_dict["A"])), np.sqrt(IP_factor*IP_E.ip(data_dict["E"], data_dict["E"])), np.sqrt(IP_factor*IP_T.ip(data_dict["T"], data_dict["T"]))
    
    # combine SNR
    snr = np.real(np.sqrt(A_snr**2 + E_snr**2 + T_snr**2)) # SNR (zero noise) = sqrt(<h|h>)
    
    print(f"A-channel snr = {A_snr.real:0.3f}, E-channel snr = {E_snr.real:0.3f}, T-channel snr = {T_snr.real:0.3f},\n\tTotal SNR = {snr:0.3f}.")
    snr_dict = {'A':A_snr.real, 'E':E_snr.real, 'T':T_snr.real}
    with open(f"{save_path}/snr-report.txt", 'w') as f:
        json.dump(snr_dict, f)
        f.flush()
    return snr

def create_PSD_injection_figure(data_dict, psd, injection_save_path, snr):
    """
    Create a frequency-domain injection figure with PSD plotted against A, E and T data..

    Parameters:
    data_dict (dict): A dictionary containing signal data for each channel,
    psd (dict):  A dictionary containing the PSD for A, E, and T channels,
    injection_save_path (str): The file path where the generated figure will be saved.
    snr (float): The SNR to display in the figure title.

    Returns:
    None: This function saves the figure to the specified path.
    """
    channels = list(data_dict.keys())
    fvals = get_fvals(data_dict[channels[0]])

    # plot data
    plt.title(f"Injection vs PSD (SNR = {snr:0.2f})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characterstic strain")
    psd_fvals = psd[channels[0]].f0 + data_dict[channels[0]].deltaF*np.arange(psd[channels[0]].data.length)

    for channel in channels:
        # For m > 0, hlm is define for f < 0 in lalsimulation. That's why abs is over fvals too.
        data = np.abs(2*fvals*data_dict[channel].data.data) # we need get both -m and m modes, right now this only has positive modes present.
        plt.loglog(-fvals, data, label = channel, linewidth = 1.2)
        plt.loglog(psd_fvals, np.sqrt(psd_fvals * psd[channel].data.data), label = channel + "-psd", linewidth = 0.8)
    
    plt.legend(loc="upper right")

    # place x-y limits
    plt.gca().set_ylim([10**(-24), 10**(-17)])
    plt.gca().set_xlim([10**(-5), 1])
    plt.grid(alpha = 0.5)

    # save
    plt.savefig(injection_save_path + "/injection-psd.png", bbox_inches = "tight")
    plt.cla()


    # plot in time domain
    plt.xlabel("Time [days]")
    plt.ylabel("h(t)")
    for channel in channels:
        TD_data = lsu.DataInverseFourier(data_dict[channel])
        tvals = np.arange(0, TD_data.data.length, 1) * TD_data.deltaT/3600/24
        # Fourier convention is different, so reverse time
        tvals = tvals[::-1]
        # The frequency serires should be double sided to yield a REAL time series. Here we instead take 2*real part.
        plt.plot(tvals, 2*TD_data.data.data.real, label = channel, linewidth = 1.2)
    
    plt.legend(loc="upper right")
    plt.grid(alpha = 0.5)
    
    # save
    plt.savefig(injection_save_path + "/injection-TD.png", bbox_inches = "tight")
    plt.cla()

def generate_lisa_TDI(P_inj, lmax=4, modes=None, tref=0.0, fref=None, provide_full_output=False, path_to_NR_hdf5=None, NR_taper_percent=1):
    print(f"generate_lisa_TDI function has been called with following arguments:\n{locals()}")
    P = lsu.ChooseWaveformParams()

    P.m1 = P_inj.m1
    P.m2 = P_inj.m2
    P.s1z = P_inj.s1z
    P.s2z = P_inj.s2z
    P.dist = P_inj.dist
    P.fmin = P_inj.fmin
    P.fmax = 0.5/P_inj.deltaT
    P.deltaF = P_inj.deltaF
    P.deltaT = P_inj.deltaT


    P.phiref = 0.0  
    P.inclination = 0.0 
    P.psi = 0.0 
    P.fref = P_inj.fref 
    P.tref = 0.0

    P.approx = P_inj.approx
    hlmf = lsu.hlmoff_for_LISA(P, Lmax=lmax, modes=modes, path_to_NR_hdf5=path_to_NR_hdf5, NR_taper_percent=NR_taper_percent)
    modes = list(hlmf.keys())

    # create TDI
    output = create_lisa_injections(hlmf, P.fmax, fref, P_inj.theta, P_inj.phi, P_inj.psi, P_inj.incl, P_inj.phiref, tref, provide_full_output)

    if provide_full_output:
        return output[0], output[1], output[2], output[3]
    else:
        return output

def generate_lisa_TDI_dict(param_dict):
    # print(param_dict)
    P = lsu.ChooseWaveformParams()
    P.m1 = param_dict["m1"] * lal.MSUN_SI
    P.m2 = param_dict["m2"] * lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = 0.0, 0.0, param_dict["s1z"]
    P.s2x, P.s2y, P.s2z = 0.0, 0.0, param_dict["s2z"]
    P.dist = param_dict["dist"] * 1e6 * lal.PC_SI

    P.deltaT, P.deltaF = param_dict["deltaT"], param_dict["deltaF"]
    P.fref = param_dict["wf-fref"]
    P.approx = lalsimulation.GetApproximantFromString(param_dict["approx"])
    P.fmin, P.fmax = param_dict["fmin"], 0.5/P.deltaT
    P.psi, P.phiref, P.inclination, P.tref, P.theta, P.phi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    P.eccentricity, P.meanPerAno = 0.0, 0.0

    modes = np.array(param_dict["modes"])
    lmax = np.max(modes[:,0])

    path_to_NR_hdf5 = param_dict["path_to_NR_hdf5"] if 'path_to_NR_hdf5' in param_dict else None

    number_of_bins = 1/(P.deltaF*P.deltaT)
    power_of_number_of_bins = np.log2(number_of_bins)
    assert power_of_number_of_bins == np.ceil(power_of_number_of_bins), f'Number of bins needs to be a power of 2, increase 1/deltaF from {1/P.deltaF} to {1/ (2**np.ceil(power_of_number_of_bins)*P.deltaT)}.'

    print("###############")
    if 1/P.deltaF/60/60/24 >0.5:
        print(f"Data length = {1/P.deltaF/60/60/24:2f} days.")
    else:
        print(f"Data length = {1/P.deltaF/60/60:2f} hrs.")


    print(f"\nWaveform is being generated with m1 = {P.m1/lsu.lsu_MSUN}, m2 = {P.m2/lsu.lsu_MSUN}, s1z = {P.s1z}, s2z = {P.s2z}, distance = {P.dist/1e6/lal.PC_SI}")
    print(f"deltaF = {P.deltaF}, fmin  = {P.fmin}, fmax = {P.fmax}, deltaT = {P.deltaT}, modes = {list(modes)}, lmax = {lmax}, tref = {param_dict['tref']}")
    print(f"phiref = {param_dict['phi_ref']}, psi = {param_dict['psi']}, inclination = {param_dict['inclination']}, beta = {param_dict['beta']}, lambda = {param_dict['lambda']}")
    print(f"path_to_NR_hdf5 = {path_to_NR_hdf5}, approx = {lalsimulation.GetStringFromApproximant(P.approx)}\n")
    print("###############")

    hlmf = lsu.hlmoff_for_LISA(P, Lmax=lmax, modes=modes, path_to_NR_hdf5=path_to_NR_hdf5, NR_taper_percent=param_dict["NR_taper_percent"]) 
    modes = list(hlmf.keys())

    # create injections
    if not('provide_full_output' in param_dict.keys()):
        param_dict['provide_full_output'] = False
    data_dict = create_lisa_injections(hlmf, P.fmax, param_dict["fref"], param_dict["beta"], param_dict["lambda"], param_dict["psi"], param_dict["inclination"], param_dict["phi_ref"], param_dict["tref"], provide_full_output = param_dict['provide_full_output']) 
    # if param_dict['provide_full_output'] = True then the output is data_dict, response, mode_TDI, mode_content
    return data_dict

def create_h5_files_from_data_dict(data_dict, save_path):
    """This function takes in data dictionary and creates h5 files from them. Assumes the data is stores as COMPLEX16FrequencySeries.
        Args:
            data_dict (dictonary): contains data for A, E, T channels,
            save_path (string): path to where you want to save the h5 files.
        Output:
            None"""
    A_h5_file = h5py.File(f'{save_path}/A-fake_strain-1000000-10000.h5', 'w')
    A_h5_file.create_dataset('data', data=data_dict["A"].data.data)
    A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = data_dict["A"].deltaF, float(data_dict["A"].epoch), data_dict["A"].data.length, data_dict["A"].f0 
    A_h5_file.close()

    E_h5_file = h5py.File(f'{save_path}/E-fake_strain-1000000-10000.h5', 'w')
    E_h5_file.create_dataset('data', data=data_dict["E"].data.data)
    E_h5_file.attrs["deltaF"], E_h5_file.attrs["epoch"], E_h5_file.attrs["length"], E_h5_file.attrs["f0"] =  data_dict["E"].deltaF, float(data_dict["E"].epoch), data_dict["E"].data.length, data_dict["E"].f0
    E_h5_file.close()

    T_h5_file = h5py.File(f'{save_path}/T-fake_strain-1000000-10000.h5', 'w')
    T_h5_file.create_dataset('data', data=data_dict["T"].data.data)
    T_h5_file.attrs["deltaF"], T_h5_file.attrs["epoch"], T_h5_file.attrs["length"], T_h5_file.attrs["f0"] = data_dict["T"].deltaF, float(data_dict["T"].epoch), data_dict["T"].data.length, data_dict["T"].f0
    T_h5_file.close()

    return None

def generate_lisa_injections(data_dict, param_dict, get_snr = True):
    if not(os.path.exists(param_dict['save_path'])):
        print(f"Provided path doesn't exist {param_dict['save_path']}, creating it.")
        os.mkdir(param_dict["save_path"])
    create_h5_files_from_data_dict(data_dict, param_dict["save_path"])
    cmd = f"util_WriteInjectionFile.py --parameter m1 --parameter-value  {param_dict['m1']} \
              --parameter m2 --parameter-value {param_dict['m2']} \
              --parameter s1x --parameter-value 0.0 --parameter s1y --parameter-value 0.0 --parameter s1z --parameter-value {param_dict['s1z']} \
              --parameter s2x --parameter-value 0.0 --parameter s2y --parameter-value 0.0 --parameter s2z --parameter-value {param_dict['s2z']}  \
              --parameter eccentricity --parameter-value 0 --approx {param_dict['approx']}  --parameter dist --parameter-value {param_dict['dist']}  \
              --parameter fmin --parameter-value {param_dict['fmin']}  --parameter incl --parameter-value {param_dict['inclination']}  \
              --parameter tref --parameter-value {param_dict['tref']}  --parameter phiref --parameter-value {param_dict['phi_ref']}  \
              --parameter theta --parameter-value {param_dict['beta']}  --parameter phi --parameter-value  {param_dict['lambda']}   \
              --parameter psi --parameter-value {param_dict['psi']} "
    print(f"Executing command to create mdc.xml.gz\n{cmd}")
    os.system(cmd)
    os.system(f"mv mdc.xml.gz {param_dict['save_path']}/mdc.xml.gz")
    os.system(f"ls {param_dict['save_path']}/*h5 | lal_path2cache > {param_dict['save_path']}/local.cache")
    os.system(f" util_SimInspiralToCoinc.py --sim-xml {param_dict['save_path']}/mdc.xml.gz --event 0 --ifo A --ifo E --ifo T ; mv coinc.xml {param_dict['save_path']}/coinc.xml")
    if get_snr and 'psd_path' in param_dict:
        psd = load_psd(param_dict)
        snr = calculate_snr(data_dict, param_dict['snr_fmin'], param_dict['snr_fmax'], 0.5/param_dict['deltaT'], psd, param_dict["save_path"])
        create_PSD_injection_figure(data_dict, psd, param_dict["save_path"], snr)
