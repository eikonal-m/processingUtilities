# -*- coding: utf-8 -*-
"""

Useful signal processing utilities to assist in the processing and analysis of 
time domain signals

@author: michael@eikonal.uk

versions of libraries:
----------------------
Python          3.8.5
Numpy           1.21.0
scipy           1.7.0
Matplotlib      3.4.2
---------------------

"""

import numpy as np
import numexpr as ne
import scipy as sp
import scipy.signal as sps
import sys
import scipy.interpolate
import scipy.signal
import matplotlib.pyplot as plt



def makeTestData(num_traces, num_samples, fs, freq_modes, amplitudes, ring_factor, t0, noise):
    """Create test data:
        num_traces = number  of traces, 
        num_samples = number of samples, 
        fs = sampling rate
        freq_modes = list of frequencies (modes)
        amplitudes = equal length list of amplitudes
        t0 is onset time of modes
        ring_factor = list  of  ringing  factors.
        This  means  a mode  with frequency f[k] will have an exponential 
        damping factor f[k]/r[k]. t0 are  the  onset times  (in  seconds) of 
        the  modes. noise is  the relative noise level.  
        Returns (data,fs) data matrix has columns of traces, fs sampling rate. 
        If the number of time samples is odd, the last one is discarded.
    """
    
    if num_samples%2 == 1:
        num_samples -= 1
    t = np.arange(num_samples)/fs
    data = np.zeros((num_samples, num_traces))
    yy = np.zeros(num_samples)
    nmodes = len(freq_modes)
    
    for k in range(0,nmodes):
        y = np.zeros(num_samples)
        it0 = round(fs*t0[k])
        tt0 = it0/fs
        yy[it0:] += amplitudes[k] * np.sin(2*np.pi * freq_modes[k] * (t[it0:]-tt0)) \
            * np.exp(-freq_modes[k] * (t[it0:]-tt0) / ring_factor[k])
    yy = yy/nmodes
    for k in range(0,num_traces):
        data[:,k] = yy + noise * np.random.randn(num_samples)
        
    return data, fs



def zeroMean(data):
    """Subtracts a constant of the trace amplitudes in data to make the mean zero.
        This is useful to re-calibrate vertical equipment offset
        Returns a shifted matrix - data
    """
    
    return data - np.mean(np.mean(data))



def filterData(data, fs, freq_low, freq_high, freq_stop=None):
    """High- and/or  lowpass  filter  applied to  data. Filter out
    frequencies below freq_Low and above freq_high.  A 0 value (freq_low/high)
    means not to apply this filter.  fStop is size 2  array with low and high 
    freq. of stop band, None if no bandstop is applied. 
    Returns filtered data matrix.
    """
    
    filter_degree = 3
    data = zeroMean(data)
    
    # bandstop filter
    if freq_stop is not None:
        Wn = freq_stop * 2 / fs
        b, a = sps.butter(filter_degree, Wn, btype='stop')
        data = sps.lfilter(b, a, data, axis=0)
        
    # lowpass filter
    Wn = freq_high * 2 / fs
    if Wn < 1 and freq_high > 0:
        b, a = sps.butter(filter_degree, Wn, btype='low')
        data = sps.lfilter(b, a, data, axis=0)
        
    # highpass filter
    Wn = freq_low * 2 / fs
    if Wn < 1 and freq_low > 0:
        b,a = sps.butter(filter_degree, Wn, btype='high')
        data = sps.lfilter(b,a,data,axis=0) 
        
    return data



def _howManyWindowsFit(num_samples, fs, time_win, dt):
    """(Helper function) to calculate how many time windows fit over signal 
    with number samples at sample rate,fs. Use windows of size time_win, 
    shifted by dt.  Compute how many windows will result (num_wins), 
    number of samples of window (numSamps_win), number of samples of increment (numSamps_inc).
    """    
    numSamps_inc = int(round(dt * fs))
    numSamps_win = int(2*round(time_win*fs/2)) # even
    
    #kwin = np.floor((num_samples-numSamps_win+1)/numSamps_inc)+1
    num_wins = int(np.floor((num_samples - numSamps_win + 1) / numSamps_inc) + 1)
    #kwin = np.floor((num_samples-numSamps_win+1)/numSamps_inc)+1

    return num_wins, numSamps_win, numSamps_inc



def s2nstack(data, fs, time_win, dt):
    """Estimate signal to noise ratio of data over selected time windows.
    time_win is time window size, 
    dt is time increment of sliding window. 
    Returns tuple (s2n,times) with s2n array of signal to noise ratio in dB 
    at times given in array  "times". 
    "None" values indicate the s2n ratio was too low to estimate.
    """
    num_samples, num_traces = data.shape
    
    # calc size of output
    num_wins, numSamps_win, numSamps_inc = _howManyWindowsFit(num_samples,fs, time_win, dt)
    
    print ('number of windows is:...', num_wins)
    
    s2n = np.zeros(num_wins)
    times = np.zeros(num_wins)
    #print('times=', times)

    data = zeroMean(data)
    t = np.arange(num_samples)/fs
    it0 = 0
    it1 = it0 + numSamps_win - 1
    itmid = int(it0 + (numSamps_win-1) / 2)
    
    print ('time to middle of window is:...',itmid)
    print ('Time array length: %s' % len(times))
    
    k = int(0)
    while (it1 < num_samples):
        times[k] = t[itmid]
        data_s2n = data[it0:it1,:]
        s = np.mean(data_s2n, axis=1)
        s2 = np.sum(np.square(s))
        (slen,) = s.shape
        s = s.reshape(slen,1)
        q = data_s2n - s*np.ones((1,num_traces))
        q2 = np.mean(np.sum(np.square(q),0))
        q2 = np.sqrt(num_traces)*q2/(np.sqrt(num_traces)-1)
        rat = (s2-q2/num_traces)/q2
        
        if rat>0:
            s2n[k] = 10*np.log10(rat*num_traces)
        else:
            s2n[k] = None
        
        k = k+1
        it0 = it0 + numSamps_inc
        it1 = it1 + numSamps_inc
        itmid = itmid + numSamps_inc
        
    return s2n, times



def relativeEntropyWindowed(data_ref, data, fs, freq_max, time_win, dt):
    """For each  trace in data compute relative entropy to dref based on PSD 
    up to freq_max, over windows of size time_win incremented by dt. 
    Return tuple (ent,times)  with matrix ent containing columns of entropy at
    times indexed by variable "times".
    """
    num_samples, num_traces = data.shape
     
    num_wins, numSamps_win, numSamps_inc = _howManyWindowsFit(num_samples,fs, time_win, dt)
    
    entropy = np.zeros((num_wins, num_traces))
    times = np.zeros(num_wins)
    t = np.arange(num_samples)/fs
    
    it0 = 0
    it1 = it0 + numSamps_win - 1
    itmid = int(it0+(numSamps_win-1)/2)
    
    k = 0
    while(it1 < num_samples):
        times[k] = t[itmid]
        data_ent = data[it0:it1,:]
        data_ref_ent = data_ref[it0:it1]
        
        ps, tmp = psd(data_ent, fs, freq_max)
        psref, tmp = psd(data_ref_ent, fs, freq_max)
        ps = 10**(ps/20)
        psref = 10**(psref/20)
        nf, nt = ps.shape
        
        for i in range(0,num_traces):
            entropy[k,i] = (sum(ps[:,i] * np.log(ps[:,i]/psref)) + sum(psref * np.log(psref/ps[:,i])))/(2*nf)
        k = k+1
        it0 = it0 + numSamps_inc
        it1 = it1 + numSamps_inc
        itmid = itmid + numSamps_inc
        
    return entropy, times


def correlateWithReference(data_ref, data, fs, time_win, dt):
    """Compute time domain correlation over time windows  of the traces
    in data with a reference trace data_ref (must have  same number of samples
    as data).  Compute correlation of each trace in data with data_ref over time
    windows of size time_win.  Window slides with increments dt.  
    Return tuple (corr_ref,times) with matrix corr_ref containing columns of 
    correlation at times indexed by variable "times".
    """
    num_samples, num_traces = data.shape
    
    num_wins, numSamps_win, numSamps_inc = _howManyWindowsFit(num_samples, fs, time_win, dt)
    
    corr_ref = np.zeros((num_wins, num_traces))
    corrTotal = np.zeros(num_traces)
    times = np.zeros(num_wins)
    t = np.arange(num_samples)/fs
    
    it0 = 0
    it1 = it0 + numSamps_win - 1
    itmid = int(it0 + (numSamps_win - 1) / 2)
    
    k = 0
    while(it1 < num_samples):
        times[k] = t[itmid]
        data_corr = data[it0:it1,:]
        data_ref_corr = data_ref[it0:it1]
        for i in range(0,num_traces):
            cc = np.corrcoef(data_ref_corr,data_corr[:,i])
            corr_ref[k,i] = cc[0,1]
        k = k+1
        it0 = it0 + numSamps_inc
        it1 = it1 + numSamps_inc
        itmid = itmid + numSamps_inc
        
        for i in range(0, num_traces):
            cc = np.corrcoef(data_ref, data[:,i])
            corrTotal[i] = cc[0,1]
            
    return corr_ref, times, corrTotal



def psd(data, fs, freq_max):
    """Compute power spectrum up to frequency fmax (0 for all).
    Return tuple (psd,f) with psd array of PSD in dB and f array of corresponding frequencies.
    """
    
    data = zeroMean(data)
    num_samples = data.shape[0]
    data_freqs = np.fft.rfft(data,axis=0)
    f = np.fft.rfftfreq(num_samples, d=1./fs)
    
    if freq_max > 0:
        imax = np.argmax(f > freq_max)
        f = f[0:imax]
        if data_freqs.ndim > 1:
            data_freqs = data_freqs[0:imax,:]
        else:
            data_freqs = data_freqs[0:imax]
    psd = abs(data_freqs)
    psd = 20*np.log10(psd)
    return (psd,f)



def cepstrum(data):
    """Compute cepstrum of data matrix.
    Return cepstrum in dB. It has the same structure as the data.
    """
    data = zeroMean(data)
    
    data_freqs = np.fft.rfft(data, axis=0)
    ps = 20*np.log(abs(data_freqs))
    
    data_cepstrum = abs(np.fft.irfft(ps,axis=0))
    data_cepstrum = 20*np.log10(np.abs(data_cepstrum))
    
    return data_cepstrum


def emd(data,max_modes=10):
    """Calculate the  Emprical Mode Decomposition  of a single  trace in
    array data.  Return array with the modes in the columns.
    """
    # initialize modes
    modes=[]
    # perform sifts until we have all modes
    residue=data
    while not _done_sifting(residue):
        # perform a sift
        imf,residue = _do_sift(residue)
        # append the imf
        modes.append(imf)
        # see if achieved max
        if len(modes) == max_modes:
            # we have all we wanted
            break
    # append the residue
    modes.append(residue)
    # return an array of modes
    return np.transpose(np.asarray(modes))


def _done_sifting(d):
    """We are done sifting is there a monotonic function."""
    return np.sum(_localmax(d))+np.sum(_localmax(-d))<=2

def _do_sift(data):
    """
    This function is modified to use the sifting-stopping criteria
    from Huang et al (2003) (this is the suggestion of Peel et al.,
    2005).  Briefly, we sift until the number of extrema and
    zerocrossings differ by at most one, then we continue sifting
    until the number of extrema and ZCs both remain constant for at
    least five sifts.
    """

    # save the data (may have to copy)
    imf=data

    # sift until num extrema and ZC differ by at most 1
    while True:
        imf=_do_one_sift(imf)
        numExtrema,numZC = _analyze_imf(imf)
        #print 'numextrema=%d, numZC=%d' %  (numExtrema, numZC) 
        if abs(numExtrema-numZC)<=1:
            break

    # then continue until numExtrema and ZCs are constant for at least
    # 5 sifts (Huang et al., 2003)
    numConstant = 0
    desiredNumConstant = 5
    lastNumExtrema = numExtrema
    lastNumZC = numZC
    while numConstant < desiredNumConstant:
        imf=_do_one_sift(imf)
        numExtrema,numZC = _analyze_imf(imf)
        if numExtrema == lastNumExtrema and \
                numZC == lastNumZC:
            # is the same so increment
            numConstant+=1
        else:
            # different, so reset
            numConstant = 0
        # save the last extrema and ZC
        lastNumExtrema = numExtrema
        lastNumZC = numZC
        

    # calc the residue
    residue=data-imf

    # return the imf and residue
    return imf, residue


def _do_one_sift(data):

    upper=_get_upper_spline(data)
    lower=-_get_upper_spline(-data)
    #upper=jinterp(find(maxes),data(maxes),xs);
    #lower=jinterp(find(mins),data(mins),xs);

    #imf=mean([upper;lower],1)
    imf = (upper+lower)*.5

    detail=data-imf

    # plot(xs,data,'b-',xs,upper,'r--',xs,lower,'r--',xs,imf,'k-')

    return detail # imf


def _get_upper_spline(data):
    """Get the upper spline using the Mirroring algoirthm from Rilling et
    al. (2003).
    """

    maxInds = np.nonzero(_localmax(data))[0]

    if len(maxInds) == 1:
        # Special case: if there is just one max, then entire spline
        # is that number
        #s=repmat(data(maxInds),size(data));
        s = np.ones(len(data))*data[maxInds]
        return s

    # Start points
    if maxInds[0]==0:
        # first point is a local max
        preTimes=1-maxInds[1]
        preData=data[maxInds[1]]
    else:
        # first point is NOT local max
        preTimes=1-maxInds[[1,0]]
        preData=data[maxInds[[1,0]]]

    # end points
    if maxInds[-1]==len(data)-1:
        # last point is a local max
        postTimes=2*len(data)-maxInds[-2]-1;
        postData=data[maxInds[-2]];
    else:
        # last point is NOT a local max
        postTimes=2*len(data)-maxInds[[-1,-2]];
        postData=data[maxInds[[-1,-2]]]

    # perform the spline fit
    t=np.r_[preTimes,maxInds,postTimes];
    d2=np.r_[preData, data[maxInds], postData];
    #s=interp1(t,d2,1:length(data),'spline');
    # XXX verify the 's' argument
    # needed to change so that fMRI dat would work
    rep = scipy.interpolate.splrep(t,d2,s=.0)
    s = scipy.interpolate.splev(range(len(data)),rep)
    # plot(1:length(data),data,'b-',1:length(data),s,'k-',t,d2,'r--');  

    return s


def _analyze_imf(d):
    numExtrema = np.sum(_localmax(d))+np.sum(_localmax(-d))
    numZC = np.sum(np.diff(np.sign(d))!=0)
    return numExtrema,numZC



def _localmax(d):
    """Calculate the local maxima of a vector.
    """

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = np.r_[-np.inf,d,-np.inf]
    diffScore=np.diff(np.sign(np.diff(diffvec)))
    
    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions=np.r_[np.nonzero(d[0:-1]!=d[1:])[0],len(d)-1]
    runLengths = np.diff(np.r_[-1, runEndingPositions])
    runStarts=runEndingPositions-runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths>1]
    realRunStops = runEndingPositions[runLengths>1]
    realRunLengths = runLengths[runLengths>1]

    # save only the runs that are local maxima
    maxRuns=(diffScore[realRunStarts]==-1) & (diffScore[realRunStops]==-1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    # CHECK THIS
    maxRunMiddles=np.round(realRunStarts[maxRuns]+realRunLengths[maxRuns]/2.)-1

    # get all the maxima
    maxima=(diffScore==-2)
    maxima[maxRunMiddles.astype(np.int32)] = True

    return maxima




def main():
    # generate 100 traces at 30,000 points
    num_traces = 100
    num_samples = 30000
    fs = 5e9
    freq_modes = [1e6, 23e6]
    amplitudes = [1., 1]
    ring_factor = [8., 8]
    t0 = [1e-6, 0]
    noise = 0.01
    
    #bandpass setup
    freq_low = 5e6
    freq_high = 20e6
    freq_max = freq_high
    
    #window set up 
    time_win=200e-9
    dt=20e-9
    
    
    # plot data create
    (data, fs) = makeTestData(num_traces, num_samples, fs, freq_modes, amplitudes, ring_factor, t0, noise)
    
    (data_ref, fs) = makeTestData(num_traces, num_samples, fs, [2e6, 15e6, 25e6], [0.9, 0.8, 0.7], [10, 9, 9], [0.75e-6, 0.05e-6, 1e-6], 0.02)
    data_ref_sng = np.mean(data_ref,axis=1)
    
    #plot1 = plt.figure(1)
    plt.pcolormesh(data)
    plt.gca().invert_yaxis()
    plt.title('Raw data')
    plt.show()
    
    # filter and plot filtered data
    data_filtered = filterData(data, fs, freq_low, freq_high)
    #data_ref = np.mean(data_filtered,axis=1)
    
    #plot2 = plt.figure(2)
    #plt.pcolormesh(data_filtered)
    plt.pcolormesh(data_ref)
    plt.gca().invert_yaxis()
    plt.title('Filtered data')
    plt.show()
    
    
    # calculate and plot signal to noise
    s2n, times_s2n = s2nstack(data, fs, time_win, dt)
    plt.plot(s2n)
    plt.show()
    
        
    # calculate and plot entropy
    entropy, times = relativeEntropyWindowed(data_ref_sng, data, fs, freq_max, time_win, dt)
    plt.plot(entropy)
    plt.show()
    
    # calculate and plot windowed correlation
    corr_ref, times, corrTotal = correlateWithReference(data_ref_sng, data, fs, time_win, dt)
    plt.plot(corr_ref)
    plt.show()
    
     # calculate and plot cepstrum
    data_cepstrum = cepstrum(data)
    plt.plot(np.mean(data_cepstrum, axis=1))
    plt.show()
    
    
    # calculate and plot Empirical Mode Decomposition (EMD)
    t = np.arange(num_samples)/fs
    max_modes = 5
    data_emd = emd(data_ref_sng, max_modes)
    nsamples, nmodes = data_emd.shape
    
    plt.plot(t*1e9, data_ref_sng)
    plt.xlabel('t(ns)')
    plt.ylabel('data')
    plt.title('EMD')
    
    for k in range(0, nmodes-1):
        plt.subplot(nmodes, 1, k+2)
        plt.subplot(nmodes,1,k+2)
        plt.plot(t*1e9,data_emd[:,k])
        plt.xlabel('time (ns)')
        plt.ylabel('emd(%d)' %(k+1))
        plt.show(block=False)
    
    
    
if __name__ == '__main__':   
    
    main()


#------------------------------------------------------------
    __author__ = "Michael Robinson"
    __copyright__ = "Copyright 2021, M Robinson"
    __license__ = "GPL"
    __version__ = "0.0.1"
    __maintainer__ = "M Robinson"
    __status__ = "Prototype"















