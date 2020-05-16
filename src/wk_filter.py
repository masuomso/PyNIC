'''
space-time filter based on the zonal wavenumber-frequency approach developed by Wheer and Kiladis (1999) 
<Input parameters>
ifile:      input file name
ofile:      output file name
filter_in:  filter info., ascii file consist of a set of (k,f_min,f_max)
time_range: time range. if you do not specify, the entire period is assumed. 
            Also empty string can be used such as ['1979-01-01',''].

[HIS] 2020/05/12 (kazu)
'''

import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
import scipy.signal as signal
import time_filter as tf

ifile      = '../data/olr.day.mean.nc'
ofile      = '../data/olr.day.mean.K_filter_Kiladis09.nc'
filter_in  = '../data/K_filter_Kiladis09.k-f.txt'
time_range = ['1979-01-01','']

def read_filter():
    f = open(filter_in)
    lines = f.readlines() 
    f.close()
    k = []
    f_min = {}
    f_max = {}
    for line in lines:
        line_split = line.split()
        k_read = int( line_split[0] )
        k.append( k_read )
        f_min.update( {k_read:float(line_split[1])} )
        f_max.update( {k_read:float(line_split[2])} )
    k = np.array( k )
    return k, f_min, f_max

def get_var_names( data_nc ):
    var_names_in = list(data_nc.variables.keys())
    var_names = []
    for var_name in var_names_in:
        if var_name not in ['lon', 'lat', 'time', 'info']:
            var_names.append( var_name )
    return var_names

def space_time_filter_gfunc( data, dt, ks, f_min=None, f_max=None ):
    fourier_fft = np.fft.fft2(data)
    nt = data.shape[1]
    nx = data.shape[0]
    freq = np.fft.fftfreq(nt,dt)
    fourier_fft_fil = np.zeros(data.shape,dtype=np.complex)
    for k in ks:
        if k>=0:
            for n in range(0,nt-1):
                if  f_min[k] <= freq[n] <=  f_max[k]:
                    fourier_fft_fil[nx-k+1-1,n] = fourier_fft[nx-k+1-1,n]
                if -f_max[k] <= freq[n] <= -f_min[k]:
                    fourier_fft_fil[k+1-1,n]    = fourier_fft[k+1-1,n]
        else:
            kabs = abs(k)
            for n in range(0,nt-1):
                if f_min[k]<=freq[n]<=f_max[k]:
                    fourier_fft_fil[kabs+1-1,n]    = fourier_fft[kabs+1-1,n]
                if -f_max[k]<=freq[n]<=f_min[k]:
                    fourier_fft_fil[nx-kabs+1-1,n] = fourier_fft[nx-kabs+1-1,n]
    return np.fft.ifft2(fourier_fft_fil).real

def space_time_filter( data, ks, f_min, f_max ):
    dt   = float( ( data.time[1]-data.time[0] ) / np.timedelta64(1,'D')  )
    data_filtered = xr.apply_ufunc(
        space_time_filter_gfunc,
        data.chunk({'lat':10}),
        dt,
        ks,
        kwargs=dict(f_min=f_min,f_max=f_max), 
        input_core_dims = [["lon", "time"], [], ["ks_dim"]],
        output_core_dims = [["lon", "time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype], )
    data_filtered["time"] = data.time
    data_filtered["lon"]  = data.lon
    return data_filtered

def main():

    ''' read filter info '''
    ks,f_min,f_max = read_filter()

    ''' read input data '''
    data_nc = netCDF4.Dataset( ifile ) # read input data with netCDF4 to obtain variable names
    data_in = xr.open_dataset(xr.backends.NetCDF4DataStore(data_nc)) # convert to xarray dataset

    ''' obtain variable names '''
    var_names = get_var_names( data_nc )

    ''' determine time range '''

    time_start_org = data_in.indexes['time'].normalize()[0]
    time_end_org   = data_in.indexes['time'].normalize()[-1]

    if 'time_range' in globals():
        time_start = time_range[0]
        time_end   = time_range[1]
        if not time_start:
            time_start = time_start_org
        if not time_end:
            time_end   = time_end_org
    else:
        time_start = time_start_org
        time_end   = time_end_org

    for var_name in var_names:
        data = xr.DataArray(data_in[var_name].sel(time=slice(time_start,time_end)))
        data_filtered = space_time_filter( data, ks, f_min, f_max ) 
        data_filtered.to_netcdf( ofile )
                
if __name__ == '__main__':
    main()

