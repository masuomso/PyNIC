import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import detrend as scipy_detrend

''' set the following parameters '''
ifile = '../data/olr.day.mean.nc'
ofile = '../data/olr.day.mean.25-90bpfil.nc'
cut_off_frq = [1./90, 1./25] # cut-off low and high-frequencies. np.nan can be used for a high- or low-pass filter
window_length = 141


''' python codes '''

def time_filter_gfunc( data, wgts ):
    weight = xr.DataArray(wgts,dims=['window'])
    number_of_weights=len(weight)
    dataxr=xr.DataArray(data,dims=['time'])
    return dataxr.rolling(time=number_of_weights,center=True).construct('window').dot(weight)

def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.
    Args:
    window: The length of the filter window. [int]
    cutoff: The cutoff frequency in inverse time steps. [float]
    """
    order       = ((window - 1) // 2 ) + 1
    nwts        = 2 * order + 1
    if np.isnan(cutoff):
        w = np.zeros([nwts])
        return w[1:-1]
    w           = np.zeros([nwts])
    n           = nwts // 2
    w[n]        = 2 * cutoff
    k           = np.arange(1., n)
    sigma       = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1]   = firstfactor * sigma
    return w[1:-1]

def detrend(data):
    if np.any(np.isnan(data)):
        print('Error: Data contains NaN values', file=sys.stderr)
        sys.exit(1)
    return scipy_detrend(data)

def main():

    ''' read input data '''
    #data = xr.open_dataset(ifile).olr
    data = xr.open_dataset(ifile)

    ''' fill NaN by linear interpolation (for the time being) '''
    data_interpolate = data.interpolate_na( "time" )

    ''' detrend the data in time '''
    data_detrend = xr.apply_ufunc(
        detrend,
        data_interpolate.chunk(),
        input_core_dims=[["time"]],
        output_core_dims=[["time"]], 
        exclude_dims=set(("time",)),  
        vectorize=True,  
        dask="parallelized",
        output_dtypes=[data.dtype],  
    )
    data_detrend["time"] = data.time

    ''' apply Lanczos Band-pass filter '''
    wgts_high = low_pass_weights( window_length, cut_off_frq[1] )
    wgts_low  = low_pass_weights( window_length, cut_off_frq[0] )
    wgts_bp   = wgts_high - wgts_low
    
    filtered = xr.apply_ufunc(
        time_filter_gfunc,
        data_detrend.chunk(), 
        wgts_bp,
        input_core_dims=[["time"], [""]],  
        output_core_dims=[["time"]],  
        exclude_dims=set(("time",)),  
        vectorize=True,  
        dask="parallelized",
        output_dtypes=[data.dtype],  
    )
    filtered["time"] = data.time

    ''' output to netcdf file '''

    filtered.to_netcdf( ofile )

if __name__ == '__main__':
    main()
