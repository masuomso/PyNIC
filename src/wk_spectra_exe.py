''' 
Space-time spectra developed by Wheeler and Kialdis (1999).
Utilize wk_analysis.py developed by Alejandro Jaramillo
Input parameters are ifile, ofile, spd, nDayWin, nDaySkip
spd:      sample per day
nDayWin:  window size (default is 96)
nDaySkip: Number of samples to skip between window segements (default is 30)
[HIS] 2020/03/16 (kazu)
'''

import wk_analysis as wk
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ifile    = '../data/olr.day.mean.nc'
ofile    = '../data/olr.wk_spectra.nc'
var_name = 'olr'
spd      = 1
nDayWin  = 96
nDaySkip = 30


def main():

    ''' read input data '''
    data = xr.open_dataset(ifile).olr
    data = data.sel(time=slice('1979-01-01','2019-12-31'),lat=slice(15,-15))
    data = data.transpose( "time", "lat", "lon" )

    x = wk.wk_analysis()
    x.import_array(data.values,varname='olr')
    x.wheeler_kiladis_spectra( spd,nDayWin,nDaySkip )

    """ convert it into xr array with (lat,z) coordinates so that grads can handle """
    k = np.linspace( -x.max_wn, x.max_wn )
    f = np.linspace( 0., x.max_freq,  )
    print(k)

    x.plot_background_removed()
    plt.show()

if __name__ == '__main__':
    main()
