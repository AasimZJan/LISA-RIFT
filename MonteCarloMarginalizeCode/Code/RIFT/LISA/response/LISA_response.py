"""This code is based on http://arxiv.org/abs/1806.10734 and 10.1103/PhysRevD.103.083011, with parts of code being taken from BBHx's C code. The response has been validated again BBHx with mismatch of around 10^(-10). This assumes fixed LISA arm length and is not the best formalism for Stellar mass binaries."""

# TO DO:
## Better documentation
## Impact of different fourier convention. Right now we follow theirs, not lal's.

# Future work
## 1) Add a different response code for Stellar mass binaries.

# Speed ups:
# https://stackoverflow.com/questions/49493482/numpy-np-multiply-vs-operator np.multiply vs * (no speedup)
# https://stackoverflow.com/questions/25870923/how-to-square-or-raise-to-a-power-elementwise-a-2d-numpy-array np.square vs **2 (no speedup for array of length 1e7). np.power is slow
# https://stackoverflow.com/questions/49459661/differences-between-numpy-divide-and-python-divide np.divide vs / (no speedup)

import numpy as np
import lal
import RIFT.lalsimutils as lsu
from scipy.interpolate import UnivariateSpline

__author__ = "A. Jan"
###########################################################################################
# Constants
###########################################################################################
e = 0.004824185218078991
omega0 = 1.9909865927683788 * 10**(-7) #1/seconds
a = 149597870700. #meters
C_SI = 299792458.
L = 2*np.sqrt(3)*a*e
YRSID_SI = 31558149.763545603

###########################################################################################
# FUNCTIONS
###########################################################################################
def create_lal_frequency_series(frequency_values, frequency_series, deltaF, epoch = 950000000, f0 = 0.0):
    """A helper function to create lal COMPLEX16FrequencySeries.
        Args:
            frequency_values (numpy.array): Frequency values at which the series is defined.
            frequency_series (numpy.array): Corresponding strain in frequency domain.
            epoch (float): Needed to create COMPLEX16FrequencySeries, by default it is 950000000.
            f0 (float): Needed to create COMPLEX16FrequencySeries, by default it is 0.0 . 
        Output:
            lal.COMPLEX16FrequencySeries object"""
    assert len(frequency_values) == len(frequency_series), "frequency_values and frequency_series don't have the same length."
    hf_lal = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, f0,  deltaF, lal.HertzUnit, len(frequency_values))
    hf_lal.data.data = frequency_series
    return hf_lal

def create_lal_COMPLEX16TimeSeries(deltaT, time_series,  epoch = 950000000, f0 = 0.0, data_is_real = True):
    """A helper function to create lal COMPLEX16TimeSeries.
        Args:
            deltaT (float): time step (1/sampling rate).
            time_series (numpy.array): strain in time domain.
            epoch (float): Needed to create COMPLEX16TimeSeries, by default it is 950000000.
            f0 (float): Needed to create COMPLEX16TimeSeries, by default it is 0.0 . 
        Output:
            lal.COMPLEX16TimeSeries object"""
    ht_lal = lal.CreateCOMPLEX16TimeSeries("ht_lal", epoch, f0, deltaT, lal.DimensionlessUnit, len(time_series))
    if data_is_real:
        ht_lal.data.data = time_series + 0j
    else:
        ht_lal.data.data = time_series
    return ht_lal


def get_fvals(frequency_series):
    """A function to evaulate frequency values of a COMPLEX16FrequencySeries. Goes from [-fNyq, fNyq - deltaF].
        Args:
            frequency_series (COMPLEX16FrequencySeries): 
        Output:
            frequency array (numpy.array)"""
    fvals = -frequency_series.deltaF*np.arange(frequency_series.data.length//2, -frequency_series.data.length//2, -1)
    return fvals



def get_Ylm(inclination, phiref, l ,m, s = -2):
    """A function to call spherical harmonics, should change to calling the GPU version in future but for now, I am only using to create injections so it should be fine. 
        Args:
            inclination (float): inclination in SSB frame,
            phiref (float): phase at coalescence,
            l, m: modes,
            s: spin weight, -2 by default.
        Output:
            Spin weighted spherical harmonics (complex)"""
    return lal.SpinWeightedSphericalHarmonic(inclination, phiref, s, l, m)

def get_closest_index(array, value):
    """A function that gives you the index at which the array has a value closest to the one you want.
        Args:
            array (np.array):
            value (float):
        Output:
            index (float)"""
    return np.argmin(np.abs(array - value))


def create_real_strain_from_double_sided_frequency_series(frequency_series, return_double_sided_frequency_series = False):
    fvals = get_fvals(frequency_series)
    tmp = np.zeros(frequency_series.data.length, dtype=complex)
    tmp = np.conj(frequency_series.data.data[::-1]) + frequency_series.data.data
    hf = create_lal_frequency_series(fvals, tmp, frequency_series.deltaF)
    if return_double_sided_frequency_series:
        return lsu.DataInverseFourier(hf), hf
    return lsu.DataInverseFourier(hf)

def convert_AET_from_FD_to_TD(data_dict_in_FD):
    """Converts the gravitational wave data for channels A, E, and T from the frequency domain (FD) to the time domain (TD).

    Args:
        data_dict_in_FD (dict): A dictionary containing frequency-domain datafor channels 'A', 'E', and 'T' (as COMPLEX16FrequencySeries).

    Output:
        dict: A dictionary containing time-domain data for channels 'A', 'E', and 'T' (as COMPLEX16TimeSeries)."""
    A_TD = lsu.DataInverseFourier(data_dict_in_FD['A'])
    A_TD.data.data.real *= 2
    A_TD.data.data.imag = np.zeros(A_TD.data.length)

    E_TD = lsu.DataInverseFourier(data_dict_in_FD['E'])
    E_TD.data.data.real *= 2
    E_TD.data.data.imag = np.zeros(E_TD.data.length)
    
    T_TD = lsu.DataInverseFourier(data_dict_in_FD['T'])
    T_TD.data.data.real *= 2
    T_TD.data.data.imag = np.zeros(T_TD.data.length)

    data_dict_in_TD = {'A':A_TD, 'E':E_TD, 'T':T_TD}

    return data_dict_in_TD

def transformed_Hplus_Hcross(beta, lamda, psi, theta, phiref, l, m):
    """This function transforms the plus and cross polarization from wave frame to SSB frame.
        Args: 
            beta  (float) = ecliptic latitude,  
            lamda (float) = ecliptic longitude, 
            psi   (float) = polarization
        Returns:
            transformed Plm (used to create the transfer function, equation 21 (http://arxiv.org/abs/2003.00357)
        """    

    # defining cos and sin for ease (checked)
    cl, sl = np.cos(lamda), np.sin(lamda)
    cb, sb = np.cos(beta),  np.sin(beta)
    cp, sp = np.cos(psi),   np.sin(psi)

    # polarization in waveframe (checked)
    Hp = [[1,0,0], [0,-1,0], [0,0,0]]
    Hc = [[0,1,0], [1,0,0] , [0,0,0]]

    # Rotation matrix (checked)
    O1 = np.zeros((3,3))
    
    O1[0,0] = cp*sl - cl*sb*sp
    O1[0,1] = -cl*cp*sb - sl*sp
    O1[0,2] = -cb*cl

    O1[1,0] = -cl*cp - sb*sl*sp
    O1[1,1] = -cp*sb*sl + cl*sp
    O1[1,2] = -cb*sl

    O1[2,0] = cb*sp
    O1[2,1] = cb*cp
    O1[2,2] = -sb
    
    # Transpose of Rotation matrix (checked)
    TO1 = np.zeros((3,3))

    TO1[0,0] = cp*sl - cl*sb*sp
    TO1[0,1] = -cp*cl - sl*sb*sp
    TO1[0,2] = cb*sp

    TO1[1,0] = -cl*cp*sb - sl*sp
    TO1[1,1] = -sl*cp*sb + cl*sp
    TO1[1,2] = cb*cp

    TO1[2,0] = -cb*cl
    TO1[2,1] = -cb*sl
    TO1[2,2] = -sb

    # Ylm factors (checked)
    # For injection, we shouldn't take -phiref as we do in marginalization. Confirm with ROS.
    Ylm = get_Ylm(theta, phiref, l, m, -2)
    Y_lm = (-1)**(l) * np.conj(get_Ylm(theta, phiref, l, -m, -2))

    Yfactorplus = 0.5*(Ylm + Y_lm)
    Yfactorcross = 0.5*1j*(Ylm - Y_lm)

    Plm = Yfactorplus*np.matmul(O1, np.matmul(Hp,TO1)) + Yfactorcross*np.matmul(O1, np.matmul(Hc,TO1))
    return Plm


def get_tf_from_phase(hlm, fmax, debug = False):#tested
    """This function differentiates phase to get tf. Similar to pycbc's time_from_frequencyseries (waveforms/utils.py) function but does not include their discontinuity check. (Now it does have those checks). 
        Args: 
            hlm (COMPLEX16FrequencySeries): The mode for which you are calculating tf , 
            fmax (float): maximum frequency (fNyq for RIFT). 
        Returns:
            tf (numpy.array): Time,
            frequency (numpy.array): Frequency (numpy.array).
        """    
    # send in hlm not shifted in time

    # get frequency and mode data
    freq = np.arange(-fmax, fmax+hlm.deltaF, hlm.deltaF)
    if debug:
        print(f"len(freq) = {len(freq)}, freq[0] = {freq[0]} Hz, freq[-1] = {freq[-1]} Hz, len(hlm) = {hlm.data.length}")

    # get amplitude and phase
    phase = np.unwrap(np.angle(hlm.data.data))
    # compute tf = -1/(2pi) * d(phase)/df
    phase  = phase - phase[0]  #Pycbc does this, doesn't change answer as expected.
    dphi = np.unwrap(np.diff(phase))
    time = -dphi / (2.*np.pi*np.diff(freq))
    # diff reduces len by 1 so artifically increasing it by adding an extra zero at the end
    tmp = np.zeros(len(time)+1)
    tmp[:-1] = time
    time = tmp

    # only focusing on f bins where data exists
    nzidx = np.nonzero(abs(hlm.data.data))[0]
    if debug:
        print(nzidx)
        print(f"tf[0](after stripping zeros) = {time[kmin]}, tf[-1](after stripping zeros) {time[kmax]}")
    kmin, kmax = nzidx[0], nzidx[-2]
    time[:kmin] = time[kmin]
    time[kmax:] = time[kmax]
    if debug:
        print(f"len(time) = {len(time)}, len(freq) = {len(freq)}")

    # saving data
    return time, freq[::-1]  #inverting frequency array to match http://arxiv.org/abs/1806.10734i fourier convention, will change it as we validate these codes.

def Evaluate_Gslr(tf, f, H, beta, lamda):
    """This function takes in tf, f, Plm (from transformed_Hplus_Hcross), beta and lamda to generate transfer function for a given mode. yslr = Sum_l^r Gslr * hlm.
        Args: 
            tf (numpy.array)= -1/2pi d(phase)/df, 
            f  (numpy.array)= frequency array, 
            H  (numpy.array)= The plus and cross polarization matrices transformed from wave to SSB frame ,
            beta (float)    = ecliptic latitude, 
            lamda (float)   = ecliptic longitude
        Returns:
            Transfer function L1 (numpy.array), Transfer function L2 (numpy.array), Transfer function L3 (numpy.array)
        """
    alpha = omega0*tf
    c, s = np.cos(alpha), np.sin(alpha)
    k = np.array([-np.cos(beta)*np.cos(lamda), -np.cos(beta)*np.sin(lamda), -np.sin(beta)])
    p0 = np.array([a*c, a*s, np.zeros(len(tf))]) # (3, N)
    kR = np.dot(k, p0) # (N,)
    phaseRdelay = 2.*np.pi/C_SI *f*kR #(N,)

    p1L = np.array([-a*e*(1 + s*s), a*e*c*s, -a*e*np.sqrt(3)*c]) # (3, N)
    p2L = np.array([a*e/2*(np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s - np.sqrt(3)*(1 + c*c)),  -a*e*np.sqrt(3)/2*(np.sqrt(3)*s - c)]) # (3, N)
    p3L = np.array([a*e/2*(-np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s + np.sqrt(3)*(1 + c*c)), -a*e*np.sqrt(3)/2*(-np.sqrt(3)*s - c)]) # (3, N)


    n1 = np.array([-1./2*c*s, 1./2*(1 + c*c), np.sqrt(3)/2*s]) # (3, N)
    kn1= np.dot(k, n1) #(N,)
    n1Hn1 = np.einsum("ij,ji->i",n1.T, np.einsum("ij,jk", H, n1))

    n2 = 1./4. * np.array([c*s - np.sqrt(3)*(1 + s*s), np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s - 3*c])
    kn2= np.dot(k, n2)
    n2Hn2 = np.einsum("ij,ji->i",n2.T, np.einsum("ij,jk", H, n2))

    n3 = 1./4*np.array([c*s + np.sqrt(3)*(1 + s*s), -np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s + 3*c])
    kn3= np.dot(k, n3)
    n3Hn3 = np.einsum("ij,ji->i",n3.T, np.einsum("ij,jk", H, n3))

    

    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    kp0 = np.dot(k, p0)

    factorcexp0 = np.exp(1j*2.*np.pi*f/C_SI * kp0)
    prefactor = np.pi*f*L/C_SI

    factorcexp12 = np.exp(1j*prefactor * (1.+kp1Lp2L/L))
    factorcexp23 = np.exp(1j*prefactor * (1.+kp2Lp3L/L))
    factorcexp31 = np.exp(1j*prefactor * (1.+kp3Lp1L/L))

    factorsinc12 = np.sinc( (prefactor * (1.-kn3))/np.pi)
    factorsinc21 = np.sinc( (prefactor * (1.+kn3))/np.pi)
    factorsinc23 = np.sinc( (prefactor * (1.-kn1))/np.pi)
    factorsinc32 = np.sinc( (prefactor * (1.+kn1))/np.pi)
    factorsinc31 = np.sinc( (prefactor * (1.-kn2))/np.pi)
    factorsinc13 = np.sinc( (prefactor * (1.+kn2))/np.pi)

    commonfac = 1j*prefactor*factorcexp0
    G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31 * np.exp(-1j*phaseRdelay)
    G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31 * np.exp(-1j*phaseRdelay)

    x = np.pi*f*L/C_SI
    z = np.exp(1j*2.*x)

    factor_convention = 2
    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*z
    factorT = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(1j*3.*x)

    Araw = 0.5 * ( (1.+z)*(G31 + G13) - G23 - z*G32 - G21 - z*G12 )
    Eraw = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13 - G31) + (2.+z)*(G12 - G32) + (1.+2.*z)*(G21 - G23) )
    Traw = 1/np.sqrt(6) * (G21 - G12 + G32 - G23 + G13 - G31)
    transferL1 = factor_convention * factorAE * Araw
    transferL2 = factor_convention * factorAE * Eraw
    transferL3 = factor_convention * factorT * Traw
    
    return transferL1, transferL2, transferL3


###########################################################################################
# FUNCTIONS USED IN RECOVERY
###########################################################################################
def get_amplitude_phase(hf): #tested
    # NOTE: Send in hlm not shifted in time, I add time shift in precomputation.
    """This function splits h(f) into Amplitude A(f) and phase phase(f). We then express h_lm as A_lm(f) * exp(i*phase(f)). Note the sign in the exponent.
       Args:
           hf (CreateCOMPLEX16FrequencySeries): Frequency domain waveform.
       Returns:
           Amplitude as a function of frequency (numpy.array), phase as a function of frequency (numpy.array) 
       """
    return np.abs(hf.data.data), np.unwrap(np.angle(hf.data.data))



def get_tf_from_phase_dict(hlm, fmax=None, fref=None, debug=True, derivative_method='analytical', shift=True):#tested
    """This function differentiates phase for each mode to get tf. Similar to pycbc's time_from_frequencyseries (waveforms/utils.py) function.
        Args: 
            hlm (dict): mode dict generated by std_and_conj_hlmoff or any other hlm(f) function in RIFT, 
            fmax (float): maximum frequency (fNyq for RIFT). 
        Returns:
            tf (dict): tf for each mode in hlm,
            frequency (dict): corresponding frequency for each mode in hlm,
            amplitude (dict): frequency domain amplitude for each mode,
            phase (dict): frequency domain phase for each mode.
        """
    # send in hlm not shifted in time
    tf_dict = {}
    freq_dict = {}
    amp_dict = {}
    phase_dict = {}
    print("Computing time frequency correspondence for mode")
    modes = np.array(list(hlm.keys()))
    #freq = -lsu.evaluate_fvals(hlm[tuple(modes[0])]) # THIS CONSUMES MOST TIME
    freq = -hlm[tuple(modes[0])].deltaF*np.arange(hlm[tuple(modes[0])].data.length//2, -hlm[tuple(modes[0])].data.length//2, -1) # this matches evaluate_fvals without using for loops
    if not(isinstance(fref, float) or isinstance(fref, int)):
        print(f'Provided fref is neither int or float, setting it to None')
        fref = None
    for mode in modes:
        print(f"\n\tMode = {mode}")
        mode = tuple(mode)
        # get amplitude and phase
        amp, phase = get_amplitude_phase(hlm[mode])
        
        # compute tf = -1/(2pi) * d(phase)/df
        if derivative_method == 'numerical-order-1':
            # faster, can be unstable for large deltaT
            dphi = np.unwrap(np.diff(phase)) 
            time = np.divide(-dphi, (2.*np.pi*hlm[mode].deltaF))
            # diff reduces len by 1 so artifically increasing it by adding an extra zero at the end
            tmp = np.zeros(len(time)+1)
            tmp[:-1] = time
            time = tmp
        elif derivative_method == 'analytical':
            # slower but stable
            spline = UnivariateSpline(freq, phase, s=0)
            dphase_df = spline.derivative()(freq)
            time = np.divide(-dphase_df, (2.*np.pi))
            
        # only focusing on f bins where data exists
        # I had to introduce this statement since sometimes a mode doesn't have data (odd m modes are not excited for q=1, so the mode content is all zero.)
        try:
            nzidx = np.nonzero(abs(hlm[mode].data.data))[0]
            kmin, kmax = nzidx[0], nzidx[-2]
            time[:kmin] = time[kmin]
            time[kmax:] = time[kmax]
        except:
            print(f"No data for {mode}")
            pass
        # saving data
        tf_dict[mode] = time 
        freq_dict[mode] = freq[::-1]
        amp_dict[mode] = amp
        phase_dict[mode] = phase
    # NOTE: This needs to be handled with care as we are dealing with phase of the waveform and we can introduce unnecessary systematics if not done correctly.
    if shift:
        modes = list(hlm.keys())
        print(f"Shifting of time and phase with fref = {fref}.")
        assert (2,2) in modes, "(2,2) mode needs to be present."
        # phase and tf shifts
        if not fref:
            # if fref not provided, set it to  frequency at max (f^2 * A_{2,2}(f)) (BBHx)
            fref = freq_dict[2,2][np.argmax(freq_dict[2,2]**2 * amp_dict[2,2])] # frequency at max (f^2 * A_{2,2}(f))
        
        # find tf at fref
        index_at_fref = get_closest_index(freq_dict[2,2], fref)
        tf_22_current = tf_dict[2,2][index_at_fref]
        phase_22_current = phase_dict[2,2][index_at_fref]
        
        # Time shift should be a multiple of deltaT
        time_shift = tf_22_current
        deltaT_here = 1/2/np.abs(freq[0])  # fNyq = 1/2/deltaT
        #time_shift = round(tf_22_current/deltaT_here) *  deltaT_here # integer multiple (closest) of deltaT
        reference_phase = 0.0
        
        # for loop needs to start with (2,2) mode
        modes.remove((2,2))
        modes.insert(0, (2,2)) 
        if debug:
            print(f"tf[2,2] at fref ({freq_dict[2,2][index_at_fref]} Hz) before shift is {tf_22_current}s (phase[2,2] = {phase_22_current}), time shift applied = {time_shift} s, difference = {time_shift - tf_22_current} s,  (deltaT used here = {deltaT_here} s) ")
            
        # subtract that from all modes. tf for (2,2) needs to be zero at fref, I will add t_ref to all modes later (create_lisa_injections for injections and precompute for recovery), making tf=t_ref at fref.
        for mode in modes:
            if debug:
                print(f"\tShifting {mode}")
            if debug:
                print(f"\t\tFor {mode}, phase = {phase_dict[mode][index_at_fref]} ({phase_dict[mode][index_at_fref]%(2*np.pi)}) before subtracting time shift such that tref[2,2] = 0 at fref. tf at fref is {tf_dict[mode][index_at_fref]}")
            tf_dict[mode] = tf_dict[mode]  - time_shift  # confirmed that I don't need to set all modes tf as 0. Conceptually, for the same time the other modes will be at a different frequency.
            # CAREFUL
            phase_dict[mode] = phase_dict[mode] - 2*np.pi*time_shift*freq_dict[mode]
            if debug:
                print(f"\t\tFor {mode}, phase = {phase_dict[mode][index_at_fref]} ({phase_dict[mode][index_at_fref]%(2*np.pi)}) after subtracting time shift and before setting phase[2,2] = 0 such that tref[2,2] = 0 at fref. tf at fref is {tf_dict[mode][index_at_fref]}")
            if mode == (2,2):
               phase_22_current = phase_dict[2,2][index_at_fref]
               difference = reference_phase - phase_22_current
            # CAREFUL
            phase_dict[mode] = phase_dict[mode] + mode[1]/2 * difference
            if debug:
                print(f"\t\tFor {mode}, phase = {phase_dict[mode][index_at_fref]} ({phase_dict[mode][index_at_fref]%(2*np.pi)}) after setting phase[2,2] = 0 at fref.")
        if debug:
            print(f"tf[2,2] at fref ({fref} Hz) after shift is {tf_dict[2,2][index_at_fref]} (phase[2,2] = {phase_dict[2,2][index_at_fref]}).")

    return tf_dict, freq_dict, amp_dict, phase_dict


def get_beta_lamda_psi_terms_Hp(beta, lamda, psi):
    """This function gives beta lamda psi terms for each term when we split up n_l * P_lm * n_l in equation 21 of http://arxiv.org/abs/2003.00357. We need this to bring out the psi dependence to marginalize over it. This gives those terms after we transform the plus polarization's frame to SSB frame.
       Args:
           beta (float): ecliptic latitude,
           lamda (float): ecliptic longitude,
           psi (numpy.array of shape (n,1): polarization angle array.
       Return:
           xx, xy, xz, yy, yz, zz terms (each is a numpy array of shape (n,1)
 
    """
    xx_term = (np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi))**2 + \
              (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi))

    xy_term = ((np.cos(psi)*np.sin(beta)*np.sin(lamda)-np.cos(lamda)*np.sin(psi)) * (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)) + 
               (np.cos(psi)*np.sin(lamda)-np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)* np.sin(psi))) + \
              ((-np.cos(psi)*np.sin(beta)*np.sin(lamda)+ np.cos(lamda)*np.sin(psi))*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi)) + 
               (np.cos(psi)*np.sin(lamda)-np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    xz_term = (np.cos(beta)*np.sin(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.sin(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) - 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)))
                        
    yy_term = (np.cos(psi)*np.sin(beta)*np.sin(lamda) - np.cos(lamda)*np.sin(psi))*(-np.cos(psi)*np.sin(beta)*np.sin(lamda)+ np.cos(lamda)*np.sin(psi)) + \
               (-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))**2
                                    
    yz_term = (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(beta)*np.sin(lamda) - np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              (-np.cos(beta)*np.cos(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
                np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    zz_term = (-np.cos(beta)**2 * np.cos(psi)**2 + np.cos(beta)**2 * np.sin(psi)**2)
    
    combined = np.vstack([xx_term, xy_term, xz_term, yy_term, yz_term, zz_term])
    return combined

def get_beta_lamda_psi_terms_Hc(beta, lamda, psi):
    """This function gives beta lamda psi terms for each term when we split up n_l * P_lm * n_l in equation 21 of http://arxiv.org/abs/2003.00357. We need this to bring out the psi dependence to marginalize over 
it. This gives those terms after we transform the cross polarization's frame to SSB frame.
       Args:
           beta (float): ecliptic latitude,
           lamda (float): ecliptic longitude,
           psi (numpy.array of shape (n,1): polarization angle array.
       Return:
           xx, xy, xz, yy, yz, zz terms (each is a numpy array of shape (n,1)
 
    """
    xx_term = 2*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))
    
    xy_term = ((-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              ((-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
                                     
    xz_term = (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)))
                         
    yy_term = 2*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))
                                   
    yz_term = (np.cos(beta)*np.sin(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.sin(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    zz_term = 2 * np.cos(beta)**2 * np.cos(psi) * np.sin(psi)

    combined = np.vstack([xx_term, xy_term, xz_term, yy_term, yz_term, zz_term])
    return combined


def Evaluate_Gslr_test_2(tf, f, beta, lamda):
    """This is the main function, takes in tf, f, beta and lamda to generate transfer function for a given mode for each xx, xy, xz, yy, yz and zz term. (need to explain this in paper)
        Args: 
            tf (numpy.array)= -1/2pi d(phase)/df, 
            f  (numpy.array)= frequency array, 
            beta (float)    = ecliptic latitude, 
            lamda (float)   = ecliptic longitude
        Returns:
            Transfer function L1 (numpy.array with xx, xy, xz, yy, yz, zz), Transfer function L2 (numpy.array with xx, xy, xz, yy, yz, zz), Transfer function L3 (numpy.array with xx, xy, xz, yy, yz, zz)
        """
    alpha = omega0*tf
    c, s = np.cos(alpha), np.sin(alpha)
    k = np.array([-np.cos(beta)*np.cos(lamda), -np.cos(beta)*np.sin(lamda), -np.sin(beta)])
    p0 = np.array([a*c, a*s, np.zeros(len(tf))]) # (3, N)
    kR = np.dot(k, p0) # (N,)
    phaseRdelay = 2.*np.pi/C_SI *f*kR #(N,)

    p1L =np.array([-a*e*(1 + s*s), a*e*c*s, -a*e*np.sqrt(3)*c]) # (3, N)
    p2L =np.array([a*e/2*(np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s - np.sqrt(3)*(1 + c*c)),  -a*e*np.sqrt(3)/2*(np.sqrt(3)*s - c)]) # (3, N)
    p3L =np.array([a*e/2*(-np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s + np.sqrt(3)*(1 + c*c)), -a*e*np.sqrt(3)/2*(-np.sqrt(3)*s - c)]) # (3, N)


    n1 = np.array([-1./2*c*s, 1./2*(1 + c*c), np.sqrt(3)/2*s]) # (3, N)
    kn1= np.dot(k, n1) #(N,)


    n2 = 1./4. * np.array([c*s - np.sqrt(3)*(1 + s*s), np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s - 3*c])
    kn2= np.dot(k, n2)


    n3 = 1./4*np.array([c*s + np.sqrt(3)*(1 + s*s), -np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s + 3*c])
    kn3= np.dot(k, n3)
    

    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    kp0 = np.dot(k, p0)

    factorcexp0 = np.exp(1j*2.*np.pi*f/C_SI * kp0)
    prefactor = np.pi*f*L/C_SI

    factorcexp12 = np.exp(1j*prefactor * (1.+kp1Lp2L/L))
    factorcexp23 = np.exp(1j*prefactor * (1.+kp2Lp3L/L))
    factorcexp31 = np.exp(1j*prefactor * (1.+kp3Lp1L/L))

    factorsinc12 = np.sinc( (prefactor * (1.-kn3))/np.pi)
    factorsinc21 = np.sinc( (prefactor * (1.+kn3))/np.pi)
    factorsinc23 = np.sinc( (prefactor * (1.-kn1))/np.pi)
    factorsinc32 = np.sinc( (prefactor * (1.+kn1))/np.pi)
    factorsinc31 = np.sinc( (prefactor * (1.-kn2))/np.pi)
    factorsinc13 = np.sinc( (prefactor * (1.+kn2))/np.pi)

    commonfac = 1j*prefactor*factorcexp0
    G12_term = commonfac * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23_term = commonfac * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31_term = commonfac * factorcexp31 * np.exp(-1j*phaseRdelay)
    
    G12xx = G12_term * n3[0,:]*n3[0,:] * factorsinc12 
    G21xx = G12_term * n3[0,:]*n3[0,:] * factorsinc21 
    G23xx = G23_term * n1[0,:]*n1[0,:] * factorsinc23 
    G32xx = G23_term * n1[0,:]*n1[0,:] * factorsinc32 
    G31xx = G31_term * n2[0,:]*n2[0,:] * factorsinc31
    G13xx = G31_term * n2[0,:]*n2[0,:] * factorsinc13 

    G12xy = G12_term * n3[0,:]*n3[1,:] * factorsinc12 
    G21xy = G12_term * n3[0,:]*n3[1,:] * factorsinc21 
    G23xy = G23_term * n1[0,:]*n1[1,:] * factorsinc23 
    G32xy = G23_term * n1[0,:]*n1[1,:] * factorsinc32 
    G31xy = G31_term * n2[0,:]*n2[1,:] * factorsinc31
    G13xy = G31_term * n2[0,:]*n2[1,:] * factorsinc13 

    G12xz = G12_term * n3[0,:]*n3[2,:] * factorsinc12 
    G21xz = G12_term * n3[0,:]*n3[2,:] * factorsinc21 
    G23xz = G23_term * n1[0,:]*n1[2,:] * factorsinc23 
    G32xz = G23_term * n1[0,:]*n1[2,:] * factorsinc32 
    G31xz = G31_term * n2[0,:]*n2[2,:] * factorsinc31
    G13xz = G31_term * n2[0,:]*n2[2,:] * factorsinc13 

    G12yy = G12_term * n3[1,:]*n3[1,:] * factorsinc12 
    G21yy = G12_term * n3[1,:]*n3[1,:] * factorsinc21 
    G23yy = G23_term * n1[1,:]*n1[1,:] * factorsinc23 
    G32yy = G23_term * n1[1,:]*n1[1,:] * factorsinc32 
    G31yy = G31_term * n2[1,:]*n2[1,:] * factorsinc31
    G13yy = G31_term * n2[1,:]*n2[1,:] * factorsinc13 

    G12yz = G12_term * n3[1,:]*n3[2,:] * factorsinc12 
    G21yz = G12_term * n3[1,:]*n3[2,:] * factorsinc21 
    G23yz = G23_term * n1[1,:]*n1[2,:] * factorsinc23 
    G32yz = G23_term * n1[1,:]*n1[2,:] * factorsinc32 
    G31yz = G31_term * n2[1,:]*n2[2,:] * factorsinc31
    G13yz = G31_term * n2[1,:]*n2[2,:] * factorsinc13 

    G12zz = G12_term * n3[2,:]*n3[2,:] * factorsinc12 
    G21zz = G12_term * n3[2,:]*n3[2,:] * factorsinc21 
    G23zz = G23_term * n1[2,:]*n1[2,:] * factorsinc23 
    G32zz = G23_term * n1[2,:]*n1[2,:] * factorsinc32 
    G31zz = G31_term * n2[2,:]*n2[2,:] * factorsinc31
    G13zz = G31_term * n2[2,:]*n2[2,:] * factorsinc13 

    x = np.pi*f*L/C_SI
    z = np.exp(1j*2.*x)

    factor_convention = 2
    #factor_convention = 2/np.sqrt(2) # for radler dataset

    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*z
    factorT = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(1j*3.*x)

    Araw_xx = 0.5 * ( (1.+z)*(G31xx + G13xx) - G23xx - z*G32xx - G21xx - z*G12xx )
    Araw_xy = 0.5 * ( (1.+z)*(G31xy + G13xy) - G23xy - z*G32xy - G21xy - z*G12xy )
    Araw_xz = 0.5 * ( (1.+z)*(G31xz + G13xz) - G23xz - z*G32xz - G21xz - z*G12xz )
    Araw_yy = 0.5 * ( (1.+z)*(G31yy + G13yy) - G23yy - z*G32yy - G21yy - z*G12yy )
    Araw_yz = 0.5 * ( (1.+z)*(G31yz + G13yz) - G23yz - z*G32yz - G21yz - z*G12yz )
    Araw_zz = 0.5 * ( (1.+z)*(G31zz + G13zz) - G23zz - z*G32zz - G21zz - z*G12zz )

    Eraw_xx = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xx - G31xx) + (2.+z)*(G12xx - G32xx) + (1.+2.*z)*(G21xx - G23xx) )
    Eraw_xy = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xy - G31xy) + (2.+z)*(G12xy - G32xy) + (1.+2.*z)*(G21xy - G23xy) )
    Eraw_xz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xz - G31xz) + (2.+z)*(G12xz - G32xz) + (1.+2.*z)*(G21xz - G23xz) )
    Eraw_yy = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13yy - G31yy) + (2.+z)*(G12yy - G32yy) + (1.+2.*z)*(G21yy - G23yy) )
    Eraw_yz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13yz - G31yz) + (2.+z)*(G12yz - G32yz) + (1.+2.*z)*(G21yz - G23yz) )
    Eraw_zz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13zz - G31zz) + (2.+z)*(G12zz - G32zz) + (1.+2.*z)*(G21zz - G23zz) )

    Traw_xx = 1/np.sqrt(6) * (G21xx - G12xx + G32xx - G23xx + G13xx - G31xx)
    Traw_xy = 1/np.sqrt(6) * (G21xy - G12xy + G32xy - G23xy + G13xy - G31xy)
    Traw_xz = 1/np.sqrt(6) * (G21xz - G12xz + G32xz - G23xz + G13xz - G31xz)
    Traw_yy = 1/np.sqrt(6) * (G21yy - G12yy + G32yy - G23yy + G13yy - G31yy)
    Traw_yz = 1/np.sqrt(6) * (G21yz - G12yz + G32yz - G23yz + G13yz - G31yz)
    Traw_zz = 1/np.sqrt(6) * (G21zz - G12zz + G32zz - G23zz + G13zz - G31zz)

    AE_term = factor_convention * factorAE
    transferL1_xx = AE_term * Araw_xx
    transferL1_xy = AE_term * Araw_xy
    transferL1_xz = AE_term * Araw_xz
    transferL1_yy = AE_term * Araw_yy
    transferL1_yz = AE_term * Araw_yz
    transferL1_zz = AE_term * Araw_zz

    transferL2_xx = AE_term * Eraw_xx
    transferL2_xy = AE_term * Eraw_xy
    transferL2_xz = AE_term * Eraw_xz
    transferL2_yy = AE_term * Eraw_yy
    transferL2_yz = AE_term * Eraw_yz
    transferL2_zz = AE_term * Eraw_zz

    T_term = factor_convention * factorT
    transferL3_xx = T_term *  Traw_xx
    transferL3_xy = T_term * Traw_xy
    transferL3_xz = T_term * Traw_xz
    transferL3_yy = T_term * Traw_yy
    transferL3_yz = T_term * Traw_yz
    transferL3_zz = T_term * Traw_zz

    return np.array([transferL1_xx, transferL1_xy, transferL1_xz, transferL1_yy, transferL1_yz, transferL1_zz]), np.array([transferL2_xx, transferL2_xy, transferL2_xz, transferL2_yy, transferL2_yz, transferL2_zz]), np.array([transferL3_xx, transferL3_xy, transferL3_xz, transferL3_yy, transferL3_yz, transferL3_zz])



