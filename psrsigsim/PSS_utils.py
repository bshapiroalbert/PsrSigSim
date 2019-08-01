"""PSS_utils.py
A place to organize methods used by multiple modules
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import scipy as sp
from astropy import units as u
from pint import models
#try:
#    import pyfftw
#    use_pyfftw = True
#except:
#    use_pyfftw = False


def shift_t(y, shift, use_pyfftw=False, PE='FFTW_EXHAUSTIVE', dt=1):
    """Shift timeseries data in time.
    Shift array, y, in time by amount, shift. For dt=1 units of samples
    (including fractional samples) are used. Otherwise, shift and dt are
    assumed to have the same physical units (i.e. seconds).
    Parameters
    ----------
    y : array like, shape (N,), real
        time series data
    shift : int or float
        amount to shift
    dt : float
        time spacing of samples in y (aka cadence)
    Returns
    -------
    out : ndarray
        time shifted data
    Examples
    --------
    >>>shift_t(y, 20)
    shift data by 20 samples

    >>>shift_t(y, 0.35, dt=0.125)
    shift data sampled at 8 Hz by 0.35 sec

    Uses np.roll() for integer shifts and the Fourier shift theorem with
    real FFT in general.  Defined so positive shift yields a "delay".
    """
    if isinstance(shift, int) and dt is 1:
        out = np.roll(y, shift)
    else:
        if use_pyfftw:
            pass
            # #print('Starting rfft')
            # #dummy_array = pyfftw.empty_aligned
            # (self.Nt, dtype=self.MD.data_type)
            # rfftw_Object = pyfftw.builders.rfft(y, planner_effort=PE)
            # 'FFTW_EXHAUSTIVE'
            # #print('Past rfft intialization')
            # yfft = rfftw_Object(y) # hermicity implicitely enforced by rfft
            # #print('Past rfft')
            # fs = np.fft.rfftfreq(len(y), d=dt)
            # phase = 1j*2*np.pi*fs*shift
            # yfft_sh = yfft * np.exp(phase)
            # irfftw_Object = pyfftw.builders.irfft(yfft_sh, planner_effort=PE)
            #'FFTW_EXHAUSTIVE'
            # out = irfftw_Object(yfft_sh)
        else:
            yfft = np.fft.rfft(y)  # hermicity implicitely enforced by rfft
            fs = np.fft.rfftfreq(len(y), d=dt)
            phase = 1j*2*np.pi*fs*shift
            yfft_sh = yfft * np.exp(phase)
            out = np.fft.irfft(yfft_sh)
    return out


def down_sample(ar, fact):
    """down_sample(ar, fact)
    down sample array, ar, by downsampling factor, fact
    """
    #TODO this is fast, but not as general as possible
    downsampled = ar.reshape(-1, fact).mean(axis=1)
    return downsampled


def rebin(ar, newlen):
    """rebin(ar, newlen)
    down sample array, ar, to newlen number of bins
    This is a general downsampling rebinner, but is slower than down_sample().
    'ar' must be a 1-d array
    """
    newBins = np.linspace(0, ar.size, newlen, endpoint=False)
    stride = newBins[1] - newBins[0]
    maxWid = int(np.ceil(stride))
    ar_new = np.empty((newlen, maxWid))  # init empty array
    ar_new.fill(np.nan)  # fill with NaNs (no extra 0s in mean)

    for ii, lbin in enumerate(newBins):
        rbin = int(np.ceil(lbin + stride))
        lbin = int(np.ceil(lbin))
        ar_new[ii, 0:rbin-lbin] = ar[lbin:rbin]

    return sp.nanmean(ar_new, axis=1)  # ingnore NaNs in mean


def top_hat_width(subband_df, subband_f0, DM):
    """top_hat_width(subband_df, subband_f0, DM)
    Returns width of a top-hat pulse to convolve with pulses for dipsersion
    broadening. Following Lorimer and Kramer, 2005 (sec 4.1.1 and A2.4)
    subband_df : subband bandwidth (MHz)
    subband_f0 : subband center frequency (MHz)
    DM : dispersion measure (pc/cm^3)
    return top_hat_width (milliseconds)
    """
    D = 4.148808e3  # sec*MHz^2*pc^-1*cm^3, dispersion const
    width_sec = 2*D * DM * (subband_df) / (subband_f0)**3
    return width_sec * 1.0e+3  # ms


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    # courtesy scipy recipes
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except TypeError:  # ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = int((window_size -1) // 2)
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window,
                                                           half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def find_nearest(array, value):
    """Returns the argument of the element in an array nearest to value.
    For half width at value use array[max:].
    """
    diff=np.abs(array-value)
    idx = diff.argmin()
    if idx == 0 or array[1] < value:
        idx = 1
    return idx


def acf2d(array, speed='fast', mode='full', xlags=None, ylags=None):
    """Courtesy of Michael Lam's PyPulse
    Calculate the autocorrelation of a 2 dimensional array.
    """
    from scipy.signal import fftconvolve, correlate

    if speed == 'fast' or speed == 'slow':
        ones = np.ones(np.shape(array))  # very close for either speed
        norm = fftconvolve(ones, ones, mode=mode)
        if speed=='fast':
            return fftconvolve(array, np.flipud(np.fliplr(array)),
                               mode=mode)/norm
        else:
            return correlate(array, array, mode=mode)/norm
    elif speed == 'exact':
        #NOTE: (r,c) convention is flipped from (x,y),
        # also that increasing c is decreasing y
        LENX = len(array[0])
        LENY = len(array)
        if xlags is None:
            xlags = np.arange(-1*LENX+1, LENX)
        if ylags is None:
            ylags = np.arange(-1*LENY+1, LENY)
        retval = np.zeros((len(ylags), len(xlags)))
        for i, xlag in enumerate(xlags):
            print(xlag)
            for j, ylag in enumerate(ylags):
                if ylag > 0 and xlag > 0:
                    A = array[:-1*ylag, xlag:]  # the "stationary" array
                    B = array[ylag:, :-1*xlag]
                elif ylag < 0 and xlag > 0:
                    A = array[-1*ylag:, xlag:]
                    B = array[:ylag, :-1*xlag]
                elif ylag > 0 and xlag < 0:  # optimize later via symmetries
                    A = array[:-1*ylag, :xlag]
                    B = array[ylag:, -1*xlag:]
                elif ylag < 0 and xlag < 0:
                    A = array[-1*ylag:, :xlag]
                    B = array[:ylag, -1*xlag:]
                else:  # one of the lags is zero
                    if ylag == 0 and xlag > 0:
                        A = array[-1*ylag:, xlag:]
                        B = array[:, :-1*xlag]
                    elif ylag == 0 and xlag < 0:
                        A = array[-1*ylag:, :xlag]
                        B = array[:, -1*xlag:]
                    elif ylag > 0 and xlag == 0:
                        A = array[:-1*ylag, :]
                        B = array[ylag:, -1*xlag:]
                    elif ylag < 0 and xlag == 0:
                        A = array[-1*ylag:, :]
                        B = array[:ylag, -1*xlag:]
                    else:
                        A = array[:, :]
                        B = array[:, :]
                        #print xlag,ylag,A,B
                C = A*B
                C = C.flatten()
                goodinds = np.where(np.isfinite(C))[0]  # check for good values
                retval[j, i] = np.mean(C[goodinds])
        return retval


def text_search(search_list, header_values, filepath, header_line=0,
                file_type='txt'):
    """ Method for pulling value from  a txt file.
    search_list = list of string-type values that demarcate the line in a txt
                  file from which to pull values
    header_values = string of column headers or array of column numbers
                    (in Python numbering) the values from which to pull
    filepath = file path of txt file. (string)
    header_line = line with headers for values.
    file_type = 'txt' or 'csv'

    returns: tuple of values matching header values for the search terms given.
    """
    #TODO Make work for other file types.
    #if file_type == 'txt':
    #    delimiter = ''
    #elif file_type == 'csv':
    #    delimiter = ','

    check = 0
    output_values = list()

    with open(filepath, 'r') as f:  # read file to local memory
        searchfile = f.readlines()

    # Find Column Numbers from column names
    if any(isinstance(elem, str) for elem in header_values):
        column_num = []
        parsed_header = searchfile[header_line].split()
        for ii, header in enumerate(header_values):
            column_num.append(parsed_header.index(header))
    else:
        column_num = np.array(header_values)

    # Find Values using search keys and column numbers.
    for line in searchfile:
        if all(ii in line for ii in search_list):

            info = line.split()
            for jj, value in enumerate(column_num):
                output_values.append(info[value])
            check += 1

    if check == 0:
        raise ValueError('Combination {0} '.format(search_list)+' not found in \
                            same line of text file.')
    if check > 1:
        raise ValueError('Combination {0} '.format(search_list)+' returned \
                            multiple results in txt file.')

    return tuple([float(i) for i in output_values])


def make_quant(param, default_unit):
    """Convenience function to intialize a parameter as an astropy quantity.
    param == parameter to initialize.
    default_unit == string that matches an astropy unit, set as
                    default for this parameter.

    returns:
        an astropy quantity

    example:
        self.f0 = make_quant(f0,'MHz')
    """
    if hasattr(param, 'unit'):
        try:
            param.to(getattr(u, default_unit))
        except u.UnitConversionError:
            raise ValueError("Frequency for {0} with incompatible unit {1}"
                             .format(param, default_unit))
        quantity = param
    else:
        quantity = param * getattr(u, default_unit)

    return quantity

def get_pint_models(psr_name, psr_file_path):
        """Function that returns pint model given a specific pulsar"""
        # will need to add section for J1713 T2 file. gls is not file wanted for this specfic pulsar.
        model_name = "{0}{1}_NANOGrav_11yv1.gls.par".format(psr_file_path,psr_name)
        par_model = models.get_model(model_name) 

        return par_model

"""
BIG BRENT HACK:
---------------------------------------------------------------------------
We will define a new function here that, when called, will write out a psrfits
file from some template. For now this will be a very specific template with
very specific parameters, but hopefully this can be generalized.

The input required will be:
signal: either the signal array in simulate or the hdf5 signal file, not sure yet
template: the template psrfits file that will be loaded. Currently this will 
        default to one in particular
the parameters of the array will be adjustable, but will have some default values
NOTE: We will assume 10 second subints after how our template file is set up,
    but this can be changed
setMJD will allow us to change the initial starting MJD and times, etc, so that
we can write psrfits files that are phase connected. The input for 'setMJD'
should be list of input values for the 'nextMJD' function.
It will also need the period of the pulsar. The initial list should be in the order:
setMJD = [obslen, period, initMJD, initSMJD, initSOFFS, initOFFSUB, increment_length]

    obslen - the length of the observation in seconds (automatically determined)
    period - period of the pulsar in seconds (automatically determined)
    initMJD - reference MJD for the first file (optional input)
    initSMJD - initial start second for the first file (optional input)
    initSOFF - initial start fractional second of the first file (optional input)
    initOFFSUB - initial subintegration center in seconds. Automatically determined
                 to be the center of the first subintegration in the file, but
                 is an optional input. NOTE: Should not be longer than half the
                 length of the shortest file being generated.
    increment length - this is how much time to add to the observation

NOTE: The nextMJD function is meant to work in corrdination with the save_psrfits
to rewrite date metadata to phase connect simulated data.
"""

def nextMJD(obslen, period, initMJD, initSMJD, initSOFFS, initOFFSUB = 42.32063, \
            nsubint = 1, increment_length = 0.0):
    """
    The purpose of this function is to rewrite the initial date of observation metadata in
    the fits file headers. It will take in the length of the current observation in seconds, 
    some initial MJD, SMJD (seconds from start of MJD), and initSOFFS (initial seconds 
    offset) and then calculate what the next set of these values, as well as the dates of 
    observation and file creation in the fits UTC format, and then return these values. This 
    will also need to be given the pulse period (in seconds) as an additional correction 
    to SOFFS.

    The increment_length argument will tell the function how much to add in between 
    observations. If '0' then the next returned values will be the initial input values.
    Otherwise the increment_length should be a number
    in units of 'days' (hours are percentages of days, etc.) and the function will return
    phase connected times as close to those incriments as possible.
    
    nsubint is the number of subintegrations there are in the file.
    
    This is currently implimented only in the save_psrfits function in PSS_utils and is 
    designed to be called only if multiple files are being created in a row so that this will
    hopefully phase connect all files. 
    
    This seems to work for now but it always needs to be referenced to the first file times and 
    centers. This makes most of the inputs useless for now which is unfortunate but I don't want
    to change it too much right now...
    """
    if increment_length == 0.0:
        SMJD = initSMJD
        SOFFS = initSOFFS
        MJD = initMJD+np.float64((initSMJD+initSOFFS)/86400.0)
        saveMJD = str(int(np.floor(MJD)))
        # Get the list of offsubs for multiple subintegrations
        if nsubint == 1:
            newOFFSUBs = [initOFFSUB] # assumption->referenced to current L-band template
        else:
            #initialize list of offsubs
            newOFFSUBs  = []
            # get length of a subint
            tsubint = obslen/nsubint # seconds
            # Get the first subint
            newOFFSUBs.append(np.float64(tsubint/2.0))
            first_offsub = np.float64(tsubint/2.0)
            # Get the subintlength in numer of periods
            psubint = tsubint/period # seconds/seconds -> periods per subint
            # Now round up for an integer number
            tsub_up = np.ceil(psubint)*period # seconds
            # Now loop through to get the rest
            for i in range(nsubint-1):
                # This is the next one
                nextoffsub = first_offsub + tsub_up
                newOFFSUBs.append(nextoffsub)
                first_offsub = nextoffsub
                
        #print(SMJD, SOFFS, MJD, saveMJD, saveDATE, saveDATEOBS, newOFFSUB)
        return  SMJD, SOFFS, MJD, saveMJD, newOFFSUBs
    
    else:
        # Determine what the MJD increase is
        MJD = initMJD+np.float64((initSMJD+initSOFFS)/86400.0) + increment_length # days
        # Get the save values
        saveMJD = str(int(np.floor(MJD)))
        seconds = np.float64("0."+str(MJD).split('.')[-1])*86400.0
        SMJD = int(np.floor(seconds))
        SOFFS = np.float64('0.'+str(seconds).split('.')[-1])#+arb_correct
        # Get the list of offsubs for multiple subintegrations
        if nsubint == 1:
            # Now we figure out what the center of the subint should be to phase connect
            initCenter = np.float64(initMJD)+np.float64((initOFFSUB+initSMJD+initSOFFS)/86400.0) # days
            newCenter = np.float64(MJD+np.float64(obslen/2.0/86400.0)) # days
            centerDiff = np.float64((newCenter-initCenter)*86400.0) # seconds
            period_to_add = np.float64(period - (centerDiff % period)) # seconds
            newOFFSUBs = [np.float64(obslen/2.0+period_to_add)]
        else:
            #initialize list of offsubs
            newOFFSUBs  = []
            # get length of a subint
            tsubint = obslen/nsubint # seconds
            # Get the first subint - center is referenced to first subint here
            initCenter = np.float64(initMJD)+np.float64((initOFFSUB+initSMJD+initSOFFS)/86400.0) # days
            newCenter = np.float64(MJD+np.float64((tsubint/2.0)/86400.0)) # days
            centerDiff = np.float64((newCenter-initCenter)*86400.0) # seconds
            period_to_add = np.float64(period - (centerDiff % period)) # seconds
            first_offsub = np.float64(tsubint/2.0+period_to_add)
            newOFFSUBs.append(first_offsub)
            # Get the subintlength in numer of periods
            psubint = tsubint/period # seconds/seconds -> periods per subint
            # Now round up for an integer number
            tsub_up = np.ceil(psubint)*period # seconds
            # Now loop through to get the rest
            for i in range(nsubint-1):
                # This is the next one
                nextoffsub = first_offsub + tsub_up
                newOFFSUBs.append(nextoffsub)
                first_offsub = nextoffsub
        
        #print(SMJD, SOFFS, MJD, saveMJD, saveDATE, saveDATEOBS, newOFFSUB)
        return SMJD, SOFFS, MJD, saveMJD, newOFFSUBs

# import some new packages
import h5py
import pdat
import astropy.io.fits as F
# now define the big function
def save_psrfits(signal, template=None, nbin = 2048, nsubint = 64, npols = 1, \
    nf = 512, tsubint = 10.0, nsubintcorr = False, check = False, DM = None, \
    freqbins = None, setMJD = None):
    #print("Attempting to save signal as psrfits")
    # NEW HACK: Must save 64 subints for now so just add zeros to the data if not correct nsubint
    # May not want to reassign the number of subints, so we add a flag for that...
    if nsubint != 64 and nsubintcorr == True:
        print("Reassigning to 64 subintegrations")
        extra_nsub = 64-nsubint
        extra_zeros = np.zeros((nf, extra_nsub*nbin))
        signal = np.concatenate((signal, extra_zeros), axis = 1)
        nsubint = 64
    # Figure out what the signal file is;
    if isinstance(signal, str) and ".hdf5" in signal:
        # read the hdf5 file
        signal_data = h5py.File(signal,mode='r')
        signal = signal_data['subint_signal'][:]
    # Otherwise we will assume that this a signal object and is just an array
    else:
        print("Signal is the input array")
    # Get the template psrfits file
    if template == None:
        # assumes we are running on bowser
        #template = "/hyrule/data/users/bjs0024/SigSim_Project1/Example_J1918/guppi_57162_J1918-0642_0026_0001.fits"
        # assumes we are running on Brent's local machine
        print("Assigning template")
        template = str("/home/brent/Desktop/Signal_Simulator_Project/guppi_57162_J1918-0642_0026_0001.fits")
    # We need to reshape the array(?) a la Jeff's code
    stop = nbin*nsubint
    signal = signal[:,:stop].astype('>i2')
    # out arrays
    Out = np.zeros((nsubint,npols,nf,nbin))
    # This loop does a thing that's important
    for ii in range(nsubint):
        idx0 = 0 + ii*2048
        idxF = idx0 + 2048
        Out[ii,0,:,:] = signal[:,idx0:idxF]
    
    # Now we get the new MJD values if necessary
    # setMJD = [obslen, period, initMJD, initSMJD, initSOFFS, initOFFSUB, increment_length]
    if setMJD:
         SMJD, SOFFS, MJD, saveMJD, saveOFFSUB = \
            nextMJD(setMJD[0], setMJD[1], initMJD = setMJD[2], initSMJD = setMJD[3], \
                    initSOFFS = setMJD[4], initOFFSUB = setMJD[5],\
                    nsubint = nsubint, increment_length = setMJD[6])
    # define the new psrfits file name
    new_psrfits = "full_signal.fits"
    # define the file(?)
    psrfits1=pdat.psrfits(new_psrfits,from_template=template,obs_mode='PSR')
    # Check `dtype`
    #print(psrfits1.fits_template[4]['DATA'][:].dtype)
    # use template file dimensions with minor changes
    psrfits1.set_subint_dims(nbin=nbin, nsblk=1, nchan=nf, \
        nsubint=nsubint, npol=npols, \
        obs_mode='PSR', data_dtype='>i2')
    # copy over the values that are not the simulated subint data array from template
    for ky in psrfits1.draft_hdr_keys[1:]:
        if ky!='SUBINT':
            psrfits1.copy_template_BinTable(ky)        
    # Make a new subint draft header
    psrfits1.HDU_drafts['SUBINT'] = psrfits1.make_HDU_rec_array(nsubint, psrfits1.subint_dtype)
    #Check that there is something there for all of the headers now. 
    #print([val is not None for val in psrfits1.HDU_drafts.values()])
    
        # Now if we need to change the date metadata we want to do that all in here
    if setMJD:
        psrfits1.set_draft_header('PRIMARY',{'STT_IMJD':int(saveMJD), \
                                             'STT_SMJD':int(SMJD),\
                                             'STT_OFFS':np.float64(SOFFS)})
        try:
            # If we have a normal fits file this will work fine
            psrfits1.HDU_drafts['POLYCO'][0][8] = MJD
        except:
            """If we have the file put together by psradd, it doesn't have a 
            'POLYCO' header, instead it has 'T2PREDICT' which has a time range
            parameter to be replaced instead."""
            #print("No POLYCO header, replacing T2PREDICT TIME_RANGE instead")
            trange_start = np.float64(MJD - (setMJD[0]/86400.0))
            trange_stop = np.float64(MJD + 2*(setMJD[0]/86400.0))
            predict_replace = "TIME_RANGE %s %s" % (trange_start, trange_stop)
            #print(predict_replace)
            psrfits1.HDU_drafts['T2PREDICT'][4][0] = predict_replace
            
        # change the subintegration offset
        
        subint = psrfits1.draft_hdrs['SUBINT']
        for ky in subint.keys():
            if subint[ky] == "OFFS_SUB":
                offsubidx = int(ky.split("E")[-1])-1
        for i in range(nsubint):
            psrfits1.HDU_drafts['SUBINT'][i][offsubidx] = saveOFFSUB[i]
        
    
    # Change polarization type
    psrfits1.set_draft_header('SUBINT',{'POL_TYPE':'AA+BB'})
    # Change the DM value if necessary
    if DM != None:
        psrfits1.set_draft_header('SUBINT',{'DM': DM})
    #single_subint_floats is a list of all the BinTable parameter names that only have one value per subint (or row).
    #print("Debugging from here(ish)")
    cols = psrfits1.single_subint_floats
    #print(cols)
    copy_cols = psrfits1.fits_template[4].read(columns=cols)
    #print(copy_cols)
    # assign copied values into draft
    for col in cols:
        #print(col, np.shape(psrfits1.HDU_drafts['SUBINT'][col][:]), np.shape(copy_cols[col][:]))
        #print(copy_cols[col][:])
        psrfits1.HDU_drafts['SUBINT'][col][:] = copy_cols[col][:]
    
    # Check to see that they've been copied
    #print(psrfits1.HDU_drafts['SUBINT']['RA_SUB'])
    
    #Reassign the template values for shorter name
    templ_subint = psrfits1.fits_template[4]
    #Assign new tsubint array and make new offset values. 
    offs_sub_init = tsubint/2
    offs_sub = np.zeros((nsubint))
    psrfits1.HDU_drafts['SUBINT']['TSUBINT'] = np.ones((nsubint))*tsubint
    #Make offset array
    for jj in range(nsubint):
        offs_sub[jj] = offs_sub_init + (jj * tsubint)
    #Assign all of the arrays to the draft SUBINT
    #The first two assignments are new, the rest are just copies (of the pertinent parts of the the old file)
    for ii in range(nsubint):
        # WE HAVE COMMENTED THIS FOR POLYCO EDITING TESTING PURPOSES!
        #psrfits1.HDU_drafts['SUBINT'][ii]['OFFS_SUB'] = offs_sub[ii]
        
        psrfits1.HDU_drafts['SUBINT'][ii]['DATA'] = Out[ii,0,:,:]
        psrfits1.HDU_drafts['SUBINT'][ii]['DAT_SCL'] = templ_subint[ii]['DAT_SCL'][:,:nf*npols]
        psrfits1.HDU_drafts['SUBINT'][ii]['DAT_OFFS'] = templ_subint[ii]['DAT_OFFS'][:,:nf*npols]
        #if freqbins == None:
        #    psrfits1.HDU_drafts['SUBINT'][ii]['DAT_FREQ'] = templ_subint[ii]['DAT_FREQ']
        #else:   
        psrfits1.HDU_drafts['SUBINT'][ii]['DAT_FREQ'] = freqbins
        psrfits1.HDU_drafts['SUBINT'][ii]['DAT_WTS'] = templ_subint[ii]['DAT_WTS']
    # Check dtype
    #print(psrfits1.HDU_drafts['SUBINT'][0]['DAT_OFFS'].dtype)
    """
    #A different list of the floating points
    one_off_floats = ['TSUBINT','OFFS_SUB','LST_SUB','RA_SUB',\
                  'DEC_SUB','GLON_SUB','GLAT_SUB','FD_ANG',\
                  'POS_ANG','PAR_ANG','TEL_AZ','TEL_ZEN']
    #Just printing to have a look at the new values
    for param in one_off_floats:
        print(param)
        print(psrfits1.HDU_drafts['SUBINT'][param])
    """
    # Check the new number of rows
    #print(psrfits1.draft_hdrs['SUBINT']['NAXIS2'])
    # Since PSRFITS can't be edited, need to make drafts and then write them all at the same time
    # Hopefully we can ignore any output errors as suggested
    psrfits1.write_psrfits(hdr_from_draft=True)
    # Close the file so it doesn't take up memory or get confused with another file. 
    psrfits1.close()
    # Now we can add a check to make sure that it worked
    if check:
        FITS = F.open(new_psrfits)
        print(FITS.info())
        subint=FITS[4]
        print(subint.data[0][19].shape)
