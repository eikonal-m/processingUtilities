# Signal processing utilities

Repository of some useful signal processing utilities

## Utilities used: 
 - make test data: generates data to test processing utilities on 
 - zero mean: subtracts a constant to ampltidues to make means zero.  Useful to re-calibrate vertical offset in data  
 - filter data: apply low/high pass filter to data using butterworth
 - s2n stack: estimate signal to noise ratio over time windowed data
 - relative entropy windowed: calculate relative entropy to a reference signal, using PSD
 - correlate with reference: time domain windowed correlation with reference signal
 - psd: calculate power spectral density using fft.rfft
 - ceptrstrum: compute ceptrstrum on data matrix
 - emd: emprical mode decomposition of single (stacked) trace


