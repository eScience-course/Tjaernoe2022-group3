# import s3fs
import xarray as xr
import urllib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as cy
from matplotlib.colors import LogNorm
import matplotlib.path as mpath
import numpy as np
import cartopy.mpl.ticker as cticker
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats import linregress
from scipy import stats
import datetime as dt
import warnings
from shapely.errors import ShapelyDeprecationWarning
import scipy
from matplotlib.ticker import MaxNLocator
import kei_functions as func

def ice_one_year(months, year, title, figname):
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    ds = xr.open_dataset(f'data/sea_ice_conc/ice_conc_{year}.nc')
    months_bool = ds['time'].dt.month.isin(months)
    ds_months_limited = ds.sel(time=months_bool).mean(dim='time')
    # ds_months_limited_array = np.array(ds_months_limited['ice_conc'][:,:].where(ds_months_limited.lat>60, drop=True))
    #        ds_months_limited = ds.isel(time=months_bool).mean(dim='time')
    data_array = np.array(ds_months_limited['ice_conc'].where(ds_months_limited.lat>60, drop=True))
    sea_ice_above_60 = ds['ice_conc'][0,:,:].where(ds.lat>60, drop=True)

    gridlons = sea_ice_above_60['lon'].values
    gridlats = sea_ice_above_60['lat'].values 

    cmap = plt.get_cmap('Blues', 10)
    ax.gridlines(linestyle='--',color='black')  
    mapped_grid_summer = ax.pcolormesh(gridlons, gridlats, data_array, transform=ccrs.PlateCarree(), shading='auto',
                                cmap=cmap, vmin=0, vmax=100)

    #ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())

    ax.coastlines()
    #XR_all_years['ice_conc'].isel(time=0).plot.pcolormesh(ax=ax, transform=ccrs.epsg(6931))
    
    cb = plt.colorbar(mapped_grid_summer, orientation="vertical", pad=0.02, 
                      aspect=16, shrink=0.8)
    cb.set_label('Sea ice concentration [%]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=12)

    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    

# One year
def chl_one_year(xr_chl, lon_min, lon_max, lat_min, lat_max, title, months, year, figname):
    fig = plt.figure(1, figsize=[7,5])
    #ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=0))

    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))

    ax.gridlines(linestyle='--',color='black')  

    gridlons = xr_chl_coords_limited['lon'].values
    gridlats = xr_chl_coords_limited['lat'].values

    # print(len(gridlons))
    # print(len(gridlats))
    cmap = plt.get_cmap('Greens', 10)
    xr_chl_one_year = xr_chl.sel(time=str(year))
    xr_chl_one_year_array = xr_chl_one_year['chl'].mean(dim='time').sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))
    mapped_grid = ax.pcolormesh(gridlons, gridlats, xr_chl_one_year_array, transform=ccrs.PlateCarree(),
                 cmap=cmap, vmin=0, vmax=10**(-6))

    ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title, size=15)

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, format=ticker.FuncFormatter(fmt),
                      aspect=16, shrink=0.8)
    cb.set_label('Chl-a concentration [kg m$^{-3}$]', rotation=270, size=12,labelpad=15)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(f'fig/{figname}')
    plt.show()
    
    
def get_xr_from_url(url, variables_to_drop):
    local_filename, headers = urllib.request.urlretrieve(url)
    html = open(local_filename)
    html.close()
    data = xr.open_dataset(local_filename, drop_variables=variables_to_drop)
    return data

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def ice_trend(months, years, title, figname):
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    #ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ds = xr.open_dataset('data/sea_ice_conc/ice_conc_2000.nc')
    sea_ice_above_60 = ds['ice_conc'][0,:,:].where(ds.lat>60, drop=True)

    gridlons = sea_ice_above_60['lon'].values
    gridlats = sea_ice_above_60['lat'].values 
    sea_ice_array = ice_slope_2d(months, years)
    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('seismic', 9)
    
    mapped_grid = ax.pcolormesh(gridlons, gridlats, sea_ice_array, transform=ccrs.PlateCarree(), shading='nearest',
                                cmap=cmap, vmin=-3, vmax=3)

    #ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())

    ax.coastlines()
    #XR_all_years['ice_conc'].isel(time=0).plot.pcolormesh(ax=ax, transform=ccrs.epsg(6931))

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, 
                      aspect=16, shrink=0.8)
    cb.set_label('Trend in sea ice concentration [pp yr$^{-1}$]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    
def ice_slope_2d(months, years):
    years = np.arange(years[0], years[-1]+1)
    array_years = []
    for y in years:
        ds = xr.open_dataset(f'data/sea_ice_conc/ice_conc_{y}.nc')
        months_bool = ds['time'].dt.month.isin(months)
        ds_months_limited = ds.isel(time=months_bool).mean(dim='time')
        data_array = np.array(ds_months_limited['ice_conc'].where(ds_months_limited.lat>60, drop=True))
        array_years.append(data_array)   
    array_years_np = np.array(array_years)
    df = pd.DataFrame(array_years_np.reshape(len(array_years_np), -1), index=years.tolist())
    x_order = np.arange(1, len(years)+1, 1)
    slopes = []
    for i in np.arange(264*264):
        slope = mstats.theilslopes(y=df.iloc[:,i], x=x_order)[0]
        slopes.append(slope)
    np_slopes = np.array(slopes)
    np_slopes_2d = np_slopes.reshape(264, 264)
    return np_slopes_2d

# chl-a relative trend
def relative_trend(xr_chl, lon_min, lon_max, lat_min, lat_max, title, years, months, figname):
    fig = plt.figure(1, figsize=[7,5])
    #ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    xr_chl_years_limited = xr_chl.sel(time=slice(str(years[0]),str(years[-1])))

    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))

    ax.gridlines(linestyle='--',color='black')  

    gridlons = xr_chl_coords_limited['lon'].values
    gridlats = xr_chl_coords_limited['lat'].values

    # print(len(gridlons))
    # print(len(gridlats))
    
    xr_chl_relative_trend = chl_relative_trend_data(xr_chl_years_limited, lon_min, lon_max, lat_min, lat_max, months)
    cmap = plt.get_cmap('seismic', 9)
    mapped_grid = ax.pcolormesh(gridlons, gridlats, xr_chl_relative_trend, transform=ccrs.PlateCarree(),
                 cmap=cmap, vmin=-0.05, vmax=0.05)

    ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title, size=15)

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, format=ticker.PercentFormatter(1.0),
                      aspect=16, shrink=0.8)
    cb.set_label('Relative trend in chl-a concentration [yr$^{-1}$]', rotation=270, size=12,labelpad=15)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(f'fig/{figname}')
    plt.show()
    
# Plot chl-a absolute trend
def absolute_trend(xr_chl, lon_min, lon_max, lat_min, lat_max, title, years, months, figname):
    fig = plt.figure(1, figsize=[7,5])
    #ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    xr_chl_years_limited = xr_chl.sel(time=slice(str(years[0]),str(years[-1])))
    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))

    ax.gridlines(linestyle='--',color='black')  

    gridlons = xr_chl_coords_limited['lon'].values
    gridlats = xr_chl_coords_limited['lat'].values

    # print(len(gridlons))
    # print(len(gridlats))
    
    xr_chl_trend_data = chl_trend_data(xr_chl_years_limited, lon_min, lon_max, lat_min, lat_max, months)
    cmap = plt.get_cmap('seismic', 9)
    mapped_grid = ax.pcolormesh(gridlons, gridlats, xr_chl_trend_data, transform=ccrs.PlateCarree(),
                 cmap=cmap, vmin=-5*10**(-8), vmax=5*10**(-8))

    ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title, size=15)

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, format=ticker.FuncFormatter(fmt),
                      aspect=16, shrink=0.8)
    cb.set_label('Trend in chl-a concentration [kg m$^{-3}$yr$^{-1}$]', rotation=270, size=12,labelpad=15)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(f'fig/{figname}')
    plt.show()
    

# Calculate relative trend 2d array
def chl_relative_trend_data(xr_chl, lon_min, lon_max, lat_min, lat_max, months):
    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))
    months_bool = xr_chl_coords_limited['time'].dt.month.isin(months)
    xr_chl_month_coords_limited = xr_chl_coords_limited.isel(time=months_bool)
    
    chl_years = np.unique(xr_chl.time.dt.year)
    chl_years_limited = []
    for i, y in enumerate(chl_years):
        try:
            chl_year_limited = xr_chl_month_coords_limited.sel(time=str(y)).mean(dim='time').variables['chl'][:,:]
            chl_years_limited.append(chl_year_limited)
        except:
            chl_years = np.delete(chl_years, i)

    chl_years_array_limited = np.array(chl_years_limited)
    df_limited = pd.DataFrame(chl_years_array_limited.reshape(len(chl_years_array_limited), -1), index=chl_years.tolist())
    average_df = df_limited.mean(axis=0)
    average_array = np.array(average_df).reshape(120,160)
    
    final_df_limited = df_limited.apply(pd.Series)
    final_df_limited[np.isnan(final_df_limited)] = 0
    
    x_order_limited = np.arange(1, len(chl_years)+1, 1)
    slopes_limited = []
    for i in np.arange(19200):
        slope_limited = stats.theilslopes(y=df_limited.iloc[:,i], x=x_order_limited)[0]
        
        slopes_limited.append(slope_limited)
        np_slopes_limited = np.array(slopes_limited)
    
    np_slopes_2d_limited = np_slopes_limited.reshape(120, 160)
    relative_trend_2d = np_slopes_2d_limited/average_array
    
    return relative_trend_2d

# Plot absolute trend
def absolute_trend(xr_chl, lon_min, lon_max, lat_min, lat_max, title, years, months, figname):
    fig = plt.figure(1, figsize=[7,5])
    #ax = plt.subplot(projection=ccrs.LambertAzimuthalEqualArea(central_latitude=90.0))
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    xr_chl_years_limited = xr_chl.sel(time=slice(str(years[0]),str(years[-1])))
    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))

    ax.gridlines(linestyle='--',color='black')  

    gridlons = xr_chl_coords_limited['lon'].values
    gridlats = xr_chl_coords_limited['lat'].values

    # print(len(gridlons))
    # print(len(gridlats))
    
    xr_chl_trend_data = chl_trend_data(xr_chl_years_limited, lon_min, lon_max, lat_min, lat_max, months)
    cmap = plt.get_cmap('seismic', 9)
    mapped_grid = ax.pcolormesh(gridlons, gridlats, xr_chl_trend_data, transform=ccrs.PlateCarree(),
                 cmap=cmap, vmin=-5*10**(-8), vmax=5*10**(-8))

    ax.set_extent([-45, 45, 60, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title, size=15)

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, format=ticker.FuncFormatter(fmt),
                      aspect=16, shrink=0.8)
    cb.set_label('Trend in chl-a concentration [kg m$^{-3}$yr$^{-1}$]', rotation=270, size=12,labelpad=15)
    cb.ax.tick_params(labelsize=10)
    fig.savefig(f'fig/{figname}')
    plt.show()
    
# Get absolute trend 2d array
def chl_trend_data(xr_chl, lon_min, lon_max, lat_min, lat_max, months):
    xr_chl_coords_limited = xr_chl.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))
    months_bool = xr_chl_coords_limited['time'].dt.month.isin(months)
    xr_chl_month_coords_limited = xr_chl_coords_limited.isel(time=months_bool)
    
    chl_years = np.unique(xr_chl.time.dt.year)
    chl_years_limited = []
    for i, y in enumerate(chl_years):
        try:
            chl_year_limited = xr_chl_month_coords_limited.sel(time=str(y)).mean(dim='time').variables['chl'][:,:]
            chl_years_limited.append(chl_year_limited)
        except:
            chl_years = np.delete(chl_years, i)

    chl_years_array_limited = np.array(chl_years_limited)
    df_limited = pd.DataFrame(chl_years_array_limited.reshape(len(chl_years_array_limited), -1), index=chl_years.tolist())
    # average_df = df_limited.mean(axis=0)
    # average_array = np.array(average_df).reshape(120,160)
    
    final_df_limited = df_limited.apply(pd.Series)
    final_df_limited[np.isnan(final_df_limited)] = 0
    
    x_order_limited = np.arange(1, len(chl_years)+1, 1)
    slopes_limited = []
    for i in np.arange(19200):
        slope_limited = stats.theilslopes(y=df_limited.iloc[:,i], x=x_order_limited)[0]
        
        slopes_limited.append(slope_limited)
        np_slopes_limited = np.array(slopes_limited)
    
    np_slopes_2d_limited = np_slopes_limited.reshape(120, 160)
    # relative_trend_2d = np_slopes_2d_limited/average_array
    
    return np_slopes_2d_limited

# Plot time series of slopes
def plot_chl_time_series(xr_chl, lon_min, lon_max, lat_min, lat_max, months, years, ylim, title):
    xr_years_limited = xr_chl.sel(time=slice(str(years[0]),str(years[-1])))
    xr_chl_coords_limited = xr_years_limited.sel(lat=slice(lat_min, lat_max)).sel(lon=slice(lon_min, lon_max))
    months_bool = xr_chl_coords_limited['time'].dt.month.isin(months)
    xr_chl_month_coords_limited = xr_chl_coords_limited.isel(time=months_bool)
    chl_years = np.unique(xr_years_limited.time.dt.year)
    average_years = []
    for i, y in enumerate(chl_years):
        try:
            chl_year_limited = xr_chl_month_coords_limited.sel(time=str(y)).mean(dim='time').variables['chl'][:,:]
            average_year = np.nanmean(np.array(chl_year_limited))
            average_years.append(average_year)
        except:
            chl_years = np.delete(chl_years, i)
    
    fig, ax = plt.subplots()
    x = chl_years
    y = average_years
    lsq_res = linregress(x, y)
    ax.plot(x, y, 'b.')
    res = stats.theilslopes(y, x, 0.95)
    ax.plot(x, res[1] + res[0] * x, '-', color='blue', alpha=0.4)
    ax.plot(x, res[1] + res[2] * x, 'r--')
    ax.plot(x, res[1] + res[3] * x, 'r--')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')
    ax.set_xlabel('Year')
    ax.set_ylabel('Chl-a concentration [kg m$^{-3}$]')
    ax.set_ylim(ylim)
    ax.set_title(title)

# Plot the trend line in area A and B to compare
def df_to_plot(df_above, df_below, confidence, title, label_below, label_above):
    fig, ax = plt.subplots()
    # Above
    x1 = df_above.mean(axis=1).index.values
    y1 = df_above.mean(axis=1).values
    lsq_res = linregress(x1, y1)
    ax.plot(x1, y1, '.', label=label_above, c='red')
    res = stats.theilslopes(y1, x1, confidence)
    ax.plot(x1, res[1] + res[0] * x1, '-', c='red', alpha=0.4)
    # ax.plot(x, res[1] + res[2] * x, 'r--')
    # ax.plot(x, res[1] + res[3] * x, 'r--')
    ax.text(2000, 8*10**(-7), 'Slope: {:.3g}'.format(res[0]))
    
    # Below
    x2 = df_below.mean(axis=1).index.values
    y2 = df_below.mean(axis=1)
    lsq_res = linregress(x2, y2)
    ax.plot(x2, y2, '.', label=label_below, color='blue')
    res = stats.theilslopes(y2, x2, confidence)
    ax.plot(x2, res[1] + res[0] * x2, '-', c='blue', alpha=0.4)
    # ax.plot(x, res[1] + res[2] * x, 'r--')
    # ax.plot(x, res[1] + res[3] * x, 'r--')
    ax.text(2006, 7*10**(-7), 'Slope: {:.3g}'.format(res[0]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Year')
    ax.set_ylabel('Chl-a Concentration [kg m$^{-3}$]')
    ax.set_title(title)
    ax.legend()
    
# Create df to calculate and plot the trends both in area A and area B
def xr_to_trend_df(xr_chl, lon_min, lon_max, lat_min, lat_max, months, years, decline_coords):
    xr_chl_coords_limited = xr_chl.sel(lon=slice(lon_min, lon_max)).sel(lat=slice(lat_min, lat_max))
    xr_chl_year_coords_limited = xr_chl_coords_limited.sel(time=slice(str(years[0]),str(years[-1])))
    months_bool = xr_chl_year_coords_limited['time'].dt.month.isin(months)
    xr_chl_time_coords_limited = xr_chl_year_coords_limited.isel(time=months_bool)
    
    # Get trend dataset
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)   
        list_above = []
        for i in np.arange(len(decline_coords['a_lat'])):
            one_coord_above = xr_chl_time_coords_limited.sel(lon=decline_coords['a_lon'][i], method='nearest').sel(lat=decline_coords['a_lat'][i], method='nearest')
            list_above.append(one_coord_above)
            # if i % 1000 == 0:
            #     print(i)
        # print(list_above)
        ds_trend_above = xr.concat(list_above, dim='coords')
        # print(ds_trend_above)
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        list_below = []
        for j in np.arange(len(decline_coords['b_lat'])):
            one_coord_below = xr_chl_time_coords_limited.sel(lon=decline_coords['b_lon'][j], method='nearest').sel(lat=decline_coords['b_lat'][j], method='nearest')
            list_below.append(one_coord_below)
            # if j % 1000 == 0:
            #     print(j)
        ds_trend_below = xr.concat(list_below, dim='coords')
        
    # Get trend dataframe    
    array_years_below = []
    for y in np.arange(years[0], years[-1]+1):
        array_year_below = np.array(ds_trend_below.chl.sel(time=slice(str(y))).mean(dim='time'))
        array_years_below.append(array_year_below)
    array_years_merged_below = np.concatenate(array_years_below)
    df_below = pd.DataFrame(array_years_merged_below.reshape(len(array_years_below), -1), index=np.arange(years[0], years[-1]+1))

    array_years_above = []
    for y in np.arange(years[0], years[-1]+1):
        array_year_above = np.array(ds_trend_above.chl.sel(time=slice(str(y))).mean(dim='time'))
        array_years_above.append(array_year_above)
    array_years_merged_above = np.concatenate(array_years_above)
    df_above = pd.DataFrame(array_years_merged_above.reshape(len(array_years_above), -1), index=np.arange(years[0], years[-1]+1))
    
    return [df_below, df_above]

# Get coordinates of the areas depending on the sea ice trend
def get_coords_from_trend(slope_2d, threshold):
    ds = xr.open_dataset(f'data/sea_ice_conc/ice_conc_2017.nc')
    ds_trend = ds.drop_dims('time')
    ds_trend['trend'] = (('yc','xc'), slope_2d)
    # l = []
    a_list_lat = []
    a_list_lon = []
    b_list_lat = []
    b_list_lon = []
    for i in np.arange(432):
        a = ds_trend.trend[i][ds_trend.trend[i]>=-1]
        b = ds_trend.trend[i][ds_trend.trend[i]<-1]
        # l.append(a)
        try:
            a_lat_np = np.array(a.lat)
            a_lon_np = np.array(a.lon)
            b_lat_np = np.array(b.lat)
            b_lon_np = np.array(b.lon)
            a_list_lat.append(a_lat_np)
            a_list_lon.append(a_lon_np)
            b_list_lat.append(b_lat_np)
            b_list_lon.append(b_lon_np)
        except:
            pass
    a_lat_all = np.concatenate(a_list_lat)
    a_lon_all = np.concatenate(a_list_lon)
    b_lat_all = np.concatenate(b_list_lat)
    b_lon_all = np.concatenate(b_list_lon)
    return {'a_lat': a_lat_all, 'a_lon': a_lon_all, 'b_lat': b_lat_all, 'b_lon': b_lon_all}

# Get sea ice trend for 432*432 grid cells
def ice_slope_2d_432(months, years):
    years = np.arange(years[0], years[-1]+1)
    array_years = []
    for y in years:
        ds = xr.open_dataset(f'data/sea_ice_conc/ice_conc_{y}.nc')
        months_bool = ds['time'].dt.month.isin(months)
        ds_months_limited = ds.isel(time=months_bool).mean(dim='time')
        # print(ds_months_limited)
        data_array = np.array(ds_months_limited['ice_conc'])
        # print(data_array)
        # plt.pcolormesh(data_array)
        # plt.show()
        array_years.append(data_array)   
    array_years_np = np.array(array_years)
    df = pd.DataFrame(array_years_np.reshape(len(array_years_np), -1), index=years.tolist())
    x_order = np.arange(1, len(years)+1, 1)
    slopes = []
    for i in np.arange(432*432):
        slope = stats.theilslopes(y=df.iloc[:,i], x=x_order)[0]
        slopes.append(slope)
        # if i % 1000 == 0:
        #     print(i)
    np_slopes = np.array(slopes)
    np_slopes_2d = np_slopes.reshape(432, 432)
    return np_slopes_2d

# Model sea ice one year
def ice_model_one_year(ds, year, title):
    ds_year = ds.sel(time=str(year)).mean(dim='time')
    ds_60 = ds_year.where(ds_year.lat>60, drop=True).where(ds_year.lon>-40, drop=True).where(ds_year.lon<50, drop=True)
    # ds_final = ds_60.assign_coords(lat=(['y', 'x'], lat_array)).assign_coords(lon=(['y', 'x'], lon_array))
    # ds_final = ds_final.rename({"i": "x", "j": "y"})
    
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.NorthPolarStereo())

    array = np.array(ds_60.siconc)
    gridlons = ds_60.lon.values
    gridlats = ds_60.lat.values

    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('Blues', 9)
    mapped_grid = ax.pcolormesh(gridlons, gridlats, array, shading='nearest',transform=ccrs.PlateCarree(),
                                cmap=cmap)
    ax.coastlines()

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, 
                      aspect=16, shrink=0.8)
    cb.set_label('Sea ice concentration (model) [%]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    
# Model sea ice absolute trend    
def ice_model_trend(ds, years, months, title, figname):
    ice_60 = ds.where(ds.lat>60, drop=True).where(ds.lon>-40, drop=True).where(ds.lon<50, drop=True)
    
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.NorthPolarStereo())
    gridlons = ice_60.lon.values
    gridlats = ice_60.lat.values
    
    sea_ice_array = ice_model_slope_2d(ds, months, years)

    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('seismic')
    mapped_grid = ax.pcolormesh(gridlons, gridlats, sea_ice_array, shading='nearest',transform=ccrs.PlateCarree(), cmap=cmap, vmin=-3, vmax=3)
    ax.coastlines()

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label('Trend in sea ice concentration (model) [pp yr$^{-1}$]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    
# Model sea ice slope
def ice_model_slope_2d(ds, months, years):
    years = np.arange(years[0], years[-1]+1)
    array_years = []
    for y in years:
        ds_year = ds.sel(time=str(y))
        months_bool = ds_year['time'].dt.month.isin(months)
        ds_months_limited = ds_year.isel(time=months_bool).mean(dim='time')
        data_array = np.array(ds_months_limited['siconc'].where(ds_months_limited.lat>60, drop=True).where(ds_months_limited.lon>-40, drop=True).where(ds_months_limited.lon<50, drop=True))
        array_years.append(data_array)   
    array_years_np = np.array(array_years)
    df = pd.DataFrame(array_years_np.reshape(len(array_years_np), -1), index=years.tolist())
    x_order = np.arange(1, len(years)+1, 1)
    slopes = []
    for i in np.arange(73*168):
        slope = stats.theilslopes(x=x_order, y=df.iloc[:,i])[0]
        slopes.append(slope)
    np_slopes = np.array(slopes)
    np_slopes_2d = np_slopes.reshape(73, 168)
    return np_slopes_2d

def chl_model_one_year(ds, year, title, figname):
    ds_year = ds.sel(time=str(year)).mean(dim='time')
    ds_final = ds_year.where(ds_year.latitude>60, drop=True).where(ds_year.longitude>-40, drop=True).where(ds_year.longitude<50, drop=True)
    
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.NorthPolarStereo())

    array = np.array(ds_final.isel(lev=0).chl)
    gridlons = ds_final.longitude.values
    gridlats = ds_final.latitude.values

    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('Greens')
    mapped_grid = ax.pcolormesh(gridlons, gridlats, array, shading='nearest',transform=ccrs.PlateCarree(),
                                cmap=cmap)
    ax.coastlines()
    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label('Chlorophyll concentration (model) [kg m$^{-3}$]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    
# Absolute trend in chl-a
def chl_model_trend(ds, years, months, title, figname):
    ds_final = ds.where(ds.latitude>60, drop=True).where(ds.longitude>-40, drop=True).where(ds.longitude<50, drop=True)
    # chl_60 = ice_60.rename({"i": "x", "j": "y"})
    # ds_final = chl_60.assign_coords(lat=(['y', 'x'], lat_array)).assign_coords(lon=(['y', 'x'], lon_array))
    
    
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.NorthPolarStereo())
    gridlons = ds_final.longitude.values
    gridlats = ds_final.latitude.values
    
    chl_array = chl_model_slope_2d(ds, months, years)

    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('seismic')
    mapped_grid = ax.pcolormesh(gridlons, gridlats, chl_array, shading='nearest',transform=ccrs.PlateCarree(),
                                cmap=cmap)
    ax.coastlines()

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label('Trend in chlorophyll concentration [kg m$^{-3}$ yr$^{-1}$]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    

# Model chl-a slope    
def chl_model_slope_2d(ds, months, years):
    years = np.arange(years[0], years[-1]+1)
    array_years = []
    for y in years:
        ds_year = ds.sel(time=str(y))
        months_bool = ds_year['time'].dt.month.isin(months)
        ds_months_limited = ds_year.isel(time=months_bool).mean(dim='time')
        data_array = np.array(ds_months_limited.isel(lev=0)['chl'].where(ds_months_limited.latitude>60, drop=True).where(ds_months_limited.longitude>-40, drop=True).where(ds_months_limited.longitude<50, drop=True))
        array_years.append(data_array)   
    array_years_np = np.array(array_years)
    df = pd.DataFrame(array_years_np.reshape(len(array_years_np), -1), index=years.tolist())
    x_order = np.arange(1, len(years)+1, 1)
    slopes = []
    for i in np.arange(73*168):
        slope = stats.theilslopes(x=x_order, y=df.iloc[:,i])[0]
        slopes.append(slope)
    np_slopes = np.array(slopes)
    np_slopes_2d = np_slopes.reshape(73, 168)
    return np_slopes_2d


# Chlorophyll relative trend (model)
def chl_model_relative_trend(ds, years, months, title, figname):
    ds_final = ds.where(ds.latitude>60, drop=True).where(ds.longitude>-40, drop=True).where(ds.longitude<50, drop=True)
    # chl_60 = ice_60.rename({"i": "x", "j": "y"})
    # ds_final = chl_60.assign_coords(lat=(['y', 'x'], lat_array)).assign_coords(lon=(['y', 'x'], lon_array))
    
    
    fig = plt.figure(1, figsize=[15,10])
    ax = plt.subplot(projection=ccrs.NorthPolarStereo())
    gridlons = ds_final.longitude.values
    gridlats = ds_final.latitude.values
    
    chl_array = chl_model_slope_2d(ds, months, years)

    ax.gridlines(linestyle='--',color='black')  
    cmap = plt.get_cmap('seismic')
    mapped_grid = ax.pcolormesh(gridlons, gridlats, chl_array, shading='nearest',transform=ccrs.PlateCarree(),
                                cmap=cmap)
    ax.coastlines()

    cb = plt.colorbar(mapped_grid, orientation="vertical", pad=0.02, aspect=16, shrink=0.8, format=ticker.PercentFormatter(1.0))
    cb.set_label('Relative trend in chlorophyll concentration [yr $^{-1}$]',size=12,rotation=270,labelpad=15)
    cb.ax.tick_params(labelsize=15)
    ax.set_title(title, size=15)
    plt.show()
    fig.savefig(f'fig/{figname}.png')
    
# Chl-a model slope    
def chl_model_slope_2d(ds, months, years):
    years = np.arange(years[0], years[-1]+1)
    array_years = []
    for y in years:
        ds_year = ds.sel(time=str(y))
        months_bool = ds_year['time'].dt.month.isin(months)
        ds_months_limited = ds_year.isel(time=months_bool).mean(dim='time')
        data_array = np.array(ds_months_limited.isel(lev=0)['chl'].where(ds_months_limited.latitude>60, drop=True).where(ds_months_limited.longitude>-40, drop=True).where(ds_months_limited.longitude<50, drop=True))
        array_years.append(data_array)   
    array_years_np = np.array(array_years)
    df = pd.DataFrame(array_years_np.reshape(len(array_years_np), -1), index=years.tolist())
    average_df = df.mean(axis=0)
    average_array = np.array(average_df).reshape(73,168)
    
    x_order = np.arange(1, len(years)+1, 1)
    slopes = []
    for i in np.arange(73*168):
        slope = stats.theilslopes(x=x_order, y=df.iloc[:,i])[0]
        slopes.append(slope)
    np_slopes = np.array(slopes)
    np_slopes_2d = np_slopes.reshape(73, 168)
    relative_trend_2d = np_slopes_2d/average_array
    return relative_trend_2d