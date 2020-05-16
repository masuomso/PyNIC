''' 
Time filter based on the Lanczos filtering
<Input parameters>
ifile:         input file name
ofile:         output file name
cut_off_frq:   cut-off low and high-frequencies. np.nan can be used for a high- or low-pass filter
               e.g., [np.nan,1./25] for 25-day high-pass filter
window_length: window length (2*n-1 in Duchon, 1979) [must be an odd integer]

[HIS] 2020/03/16 (kazu)
'''

import pandas as pd
import xarray as xr
import numpy as np
import netCDF4
from scipy.signal import detrend as scipy_detrend

''' set the following parameters '''
ifile         = '../data/olr.day.mean.nc'
ofile         = '../data/olr.day.mean.25-90bpfil.nc'
cut_off_frq   = [1./90, 1./25]   # [1/day]
window_length = 141


''' python codes '''

def time_filter_gfunc( data, wgts ):
    weight = xr.DataArray(wgts,dims=['window'])
    number_of_weights=len(weight)
    dataxr=xr.DataArray(data,dims=['time'])
    return dataxr.rolling(time=number_of_weights,center=True).construct('window').dot(weight)

def low_pass_weights(window, cutoff, dt):
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
    w[n]        = 2 * cutoff * dt
    k           = np.arange(1., n)
    sigma       = np.sin(np.pi*k/n) * n/(np.pi*k)
    firstfactor = np.sin(2.*np.pi*cutoff*k*dt) / (np.pi*k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1]   = firstfactor * sigma
    return w[1:-1]

def detrend(data):
    if np.any(np.isnan(data)):
        print('Error: Data contains NaN values', file=sys.stderr)
        sys.exit(1)
    return scipy_detrend(data)

def get_var_names( data_nc ):
    var_names_in = list(data_nc.variables.keys())
    var_names = []
    for var_name in var_names_in:
        if var_name not in ['lon', 'lat', 'time', 'info']:
            var_names.append( var_name )
    return var_names

def time_filter_main( data ):
        ''' fill NaN by linear interpolation (for the time being) '''
        data_interpolate = data.interpolate_na( "time" )

        ''' detrend the data in time '''
        data_detrend = xr.apply_ufunc(
            detrend,
            data_interpolate.chunk(),
            input_core_dims  = [["time"]],
            output_core_dims = [["time"]], 
            exclude_dims=set(("time",)),  
            vectorize=True,  
            dask="parallelized",
            output_dtypes=[data.dtype],  
        )
        data_detrend["time"] = data.time

        ''' apply Lanczos Band-pass filter '''
        dt  =  float( ( data.time[1]-data.time[0] ) / np.timedelta64(1,'D')  )
        wgts_high = low_pass_weights( window_length, cut_off_frq[1], dt )
        wgts_low  = low_pass_weights( window_length, cut_off_frq[0], dt )
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
        return filtered

def main():

    ''' read input data '''
    data_nc = netCDF4.Dataset( ifile ) # read input data with netCDF4 to obtain variable names
    data_in = xr.open_dataset(xr.backends.NetCDF4DataStore(data_nc)) # convert to xarray dataset
   
    ''' obtain variable names '''
    var_names = get_var_names( data_nc )

    for var_name in var_names:
        data = xr.DataArray(data_in[var_name])
        filtered = xr.DataArray( time_filter_main( data ) )
        if var_name in [var_names[0]]:
            filtered_out = filtered.to_dataset(name=var_name)
        else:
            filtered_out[var_name] = filtered

    ''' output to netcdf file '''

    filtered_out.to_netcdf( ofile )

if __name__ == '__main__':
    main()
