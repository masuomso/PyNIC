''' 
Space-time spectra developed by Wheeler and Kialdis (1999).
Utilize wk_analysis.py developed by Alejandro Jaramillo
Input parameters are ifile, ofile, spd, nDayWin, nDaySkip
ifile:      input file name
ofile:      oupput file name
var_name:   variable name
lat_range:  latitudinal range 
spd:        sample per day
nDayWin:    window size (default is 96)
nDaySkip:   Number of samples to skip between window segements (default is 30)
time_range: time range. if you do not specify, the entire period is assumed. 
            Also empty string can be used such as ['1979-01-01',''].

[HIS] 2020/03/16 (kazu)
'''

import wk_analysis as wk
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ifile     = '../data/olr.day.mean.nc'
ofile     = '../data/olr.wk_spectra.nc'
var_name  = 'olr'
lat_range = [-15, 15]
spd       = 1
nDayWin   = 96
nDaySkip  = 30
time_range = ['1979-01-01','']

def main():

    ''' read input data '''
    data = xr.open_dataset(ifile).olr
    lats = lat_range[0]
    latn = lat_range[1]

    ''' determine time range '''
    time_start_org = data.indexes['time'].normalize()[0]
    time_end_org   = data.indexes['time'].normalize()[-1]
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

    ''' extract data '''
    data = data.sel(time=slice(time_start,time_end),lat=slice(latn,lats))
    data = data.transpose( "time", "lat", "lon" )

    x = wk.wk_analysis()
    x.import_array(data.values,varname='olr')
    x.wheeler_kiladis_spectra( spd,nDayWin,nDaySkip )

    """ convert it into xr array with (lat,z) coordinates so that grads can handle """
    k = np.linspace( -x.max_wn, x.max_wn )
    f = np.linspace( 0., x.max_freq,  )

    x.plot_background_removed()
    plt.show()

if __name__ == '__main__':
    main()
