from __future__ import print_function
import xarray as xr
import numpy as np
import intake
#import s3fs
import requests
import xml.etree.ElementTree as ET
import cftime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cmocean
from shapely.geometry.polygon import LinearRing
plt.rcParams.update({'font.size': 22})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Functions for solving some of the special tasks in this report
# The author is Astrid Bragstad Gjelsvik if not otherwise specified


# ----------------------------------------------------------------------
# Access data
# ----------------------------------------------------------------------
def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    """
    Returns data from ESGF server
    author: Unknown, provided through ESGF and Anne Fouilloux
    """
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = [] 
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)

# ------------------------------------------------------------
# Select and compute data
# ------------------------------------------------------------

def choose_period(model_data, START_YEAR, END_YEAR,ProlepticGregorian=False):
    
    # This function picks out a specified period consisting of one or more years.
    # The default calender is the DatetimeNoLeap. if the model uses a
    # proleptic gregorian calender, this must be specified.
    
    if ProlepticGregorian:
        START = cftime.DatetimeProlepticGregorian(START_YEAR, 1, 15)
        END = cftime.DatetimeProlepticGregorian(END_YEAR, 1, 15)
    else:
        START = cftime.DatetimeNoLeap(START_YEAR, 1, 15)
        END = cftime.DatetimeNoLeap(END_YEAR, 1, 15)
    
    return model_data.sel(time=slice(START,END))


def get_summer(model_data):
    
    # This function picks out the summer (June, July, August) 
    # seasonal average for model data of a one year period or longer.
    
    model_data = model_data.groupby('time.season').mean('time')
    
    return model_data.sel(season='JJA')


def convert360_180(_ds):
    """
    convert longitude from 0-360 to -180 -- 180 deg
    author: Sara M. Blichner
    """
    # check if already 
    attrs = _ds['lon'].attrs
    if _ds['lon'].min() >= 0:
        with xr.set_options(keep_attrs=True): 
            _ds.coords['lon'] = (_ds['lon'] + 180) % 360 - 180
        _ds = _ds.sortby('lon')
    return _ds


def computeWeightedMean(ds, MODIS=False, grid_label='gr',CMCC=False,var=None):
    """
    author: Anne Fouilloux
    modified by: Astrid Bragstad Gjelsvik
    """
    # Compute weights based on the xarray you pass
    if MODIS:  
        weights = np.cos(np.deg2rad(ds.latitude))
    else:
        if grid_label=='gr':
            weights = np.cos(np.deg2rad(ds.lat))
        else:
            weights = np.cos(np.deg2rad(ds.latitude))
    weights.name = "weights"
    # Compute weighted mean
    air_weighted = ds.weighted(weights)
    if MODIS:
        weighted_mean = air_weighted.mean(("longitude", "latitude"))
    else:
        if grid_label=='gr':
            weighted_mean = air_weighted.mean(("lon", "lat"))
        else:
            if CMCC:
                weighted_sum = air_weighted.sum(("i", "j"))
                weighted_mean = weighted_sum[var]*1./(len(ds.i)*len(ds.j))
            else: 
                weighted_mean = air_weighted.mean(("i", "j"))
    return weighted_mean


def choose_square(lats,lons,data,data180=None):
    
    # Selects a square given by latitude and longitude coordinates.
    # Handy if the model does not have langitudes and latitudes as dimensions
    # If the given coordinates are known to be close to longitude 0 or 360 on a model 
    # with longitude range from 0 to 360 you risk getting nans or strange values 
    # when indexing. In that case, provide the dataset with converted longitude 
    # to -180-180 range as well.

    if -170 < lons[0] < 10:
        square = data180.where((data180.latitude > lats[0]) & (data180.latitude < lats[1]) 
                                   & (data180.longitude > lons[0]) & (data180.longitude < lons[1]),drop=True ).squeeze()
    elif lons[1] > 350:
        lons[0] -= 360
        lons[1] -= 360
        square = data180.where((data180.latitude > lats[0]) & (data180.latitude < lats[1]) 
                                   & (data180.longitude > lons[0]) & (data180.longitude < lons[1]),drop=True ).squeeze()
    elif lons[0] < -170:
        lons[0] += 360
        lons[1] += 360
        square = data.where((data.latitude > lats[0]) & (data.latitude < lats[1]) 
                                   & (data.longitude > lons[0]) & (data.longitude < lons[1]),drop=True ).squeeze()
    else:
        square = data.where((data.latitude > lats[0]) & (data.latitude < lats[1]) 
                                & (data.longitude > lons[0]) & (data.longitude < lons[1]),drop=True ).squeeze()
    return square


def computeTimeseries(data, grid_label='gr',MODIS=False,CMCC=False,ProlepticGregorian=False):
    
    # Averages satellite data over areas Fram Strait and Bering Strait
    
    # Bering Strait
    LON_MIN_bs = -174
    LON_MAX_bs = -166
    LAT_MIN_bs = 70
    LAT_MAX_bs = 73

    # Fram strait
    LAT_MIN_fs = 75
    LON_MIN_fs = -5
    LAT_MAX_fs = 78
    LON_MAX_fs = 5
    
    # Outputs a pandas dataframe timeseries over the areas.
    
    
    
    if MODIS:
        # Get copy of data
        ds1 = data.copy()
        ds2 = data.copy()
        # Select Bering Strait area and compute weighted mean
        ds1 = ds1.sel(latitude=slice(LAT_MIN_bs,LAT_MAX_bs),
                longitude=slice(LON_MIN_bs,LON_MAX_bs))
        # Select Fram Strait area and compute weighted mean
        ds2 = ds2.sel(latitude=slice(LAT_MIN_fs,LAT_MAX_fs),
                longitude=slice(LON_MIN_fs,LON_MAX_fs))
        # Compute weighted mean
        ds1 = computeWeightedMean(ds1,MODIS=True)
        ds2 = computeWeightedMean(ds2,MODIS=True)
        # Rename timeseries after area
        ds1['chl_fs'] = ds2['chlor_a']
        ds1 = ds1.rename({'chlor_a':'chl_bs'})
        df = ds1.to_dataframe()
    else:
        if grid_label=='gr':
            # Get copy of data
            ds1 = data.copy()
            # Convert latitude grid of model data
            ds1 = convert360_180(ds1)
            # Extract time period from models
            ds1 = choose_period(ds1,2002,2100)
            ds2 = ds1.copy()
            # Select Bering Strait area 
            ds1 = ds1.sel(lat=slice(LAT_MIN_bs,LAT_MAX_bs),
                    lon=slice(LON_MIN_bs,LON_MAX_bs))
            # Select Fram Strait area
            ds2 = ds2.sel(lat=slice(LAT_MIN_fs,LAT_MAX_fs),
                    lon=slice(LON_MIN_fs,LON_MAX_fs))
            # Compute mean
            ds1 = computeWeightedMean(ds1)
            ds2 = computeWeightedMean(ds2)
            ds1['chl_fs'] = ds2['chl']*1e+6
            ds1['chl'] = ds1['chl']*1e+6
            # Rename timeseries after area
            ds = ds1.rename({'chl':'chl_bs'})
        else:
            # Get copy of data
            ds1 = data.copy()
            # Extract time period from model
            ds1 = choose_period(ds1,2002,2100,ProlepticGregorian)
            ds2 = ds1.copy()
            data180 = ds1.copy()
            data180.coords['longitude'] = (data180['longitude'] + 180) % 360 - 180
            # Select Bering Strait area 
            ds1 = choose_square(lats=[LAT_MIN_bs,LAT_MAX_bs],
                    lons=[LON_MIN_bs,LON_MAX_bs],data=ds1,data180=data180)
            # Select Fram Strait area
            ds2 = choose_square(lats=[LAT_MIN_fs,LAT_MAX_fs],
                    lons=[LON_MIN_fs,LON_MAX_fs],data=ds2,data180=data180)
            # Compute weighted mean
            if CMCC:
                ds1 = computeWeightedMean(ds1,grid_label='gn',CMCC=True,var='chl')
                ds2 = computeWeightedMean(ds2,grid_label='gn',CMCC=True,var='chl')
                # Convert model data to mg/m³
                ds1 = ds1*1e+6
                ds2 = ds2*1e+6
                # Create new dataset
                ds = xr.Dataset(data_vars=dict(
                    chl_bs=(["time"], ds1),
                    chl_fs=(["time"], ds2),),
                            coords=dict(
                    time=ds1.time,),)
            else: 
                ds1 = computeWeightedMean(ds1,grid_label='gn')
                ds2 = computeWeightedMean(ds2,grid_label='gn')
                ds1['chl_fs'] = ds2['chl']*1e+6
                ds1['chl'] = ds1['chl']*1e+6
                # Rename timeseries after area
                ds = ds1.rename({'chl':'chl_bs'})
                
        df = ds.to_dataframe()
        datetimeindex = ds.indexes['time'].to_datetimeindex()
        df.index = datetimeindex
        
    return df


def computeCorr(model_chl, model_siconc, START_YEAR, NYEARS,convert360,grid_label='gr',ProlepticGregorian=False):
   
    # Computes sea ice and chlorophyll average values in a specified grid covering
    # the central Arctic (around the North Pole) and above the Bering Strait,
    # for the summers in each year for as long as you specify.
    
    chlTiles = []
    siconcTiles = []
    year = START_YEAR

    if grid_label=='gr':
    
        if convert360:
            model_chl = convert360_180(model_chl)
            model_siconc = convert360_180(model_siconc)
        
        for nyear in range(NYEARS):
    
            model_chl_period = choose_period(model_chl,year,year+1,ProlepticGregorian)
            model_chl_period_JJA = get_summer(model_chl_period)

            model_siconc_period = choose_period(model_siconc,year,year+1,ProlepticGregorian)
            model_siconc_period_JJA = get_summer(model_siconc_period)
    
            # Around the North Pole

            LONS = [-180, -170]
            LATS = [84, 86]
            lats = LATS
            for i in range(2):
                lons = LONS
                chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                chlTile = computeWeightedMean(chlTile)
                siconcTile = computeWeightedMean(siconcTile)
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(35):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                    siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                    chlTile = computeWeightedMean(chlTile)
                    siconcTile = computeWeightedMean(siconcTile)
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                lats = np.add(lats, [2,2])

            # Above Bering Strait
        
            LONS = [-180, -170]
            LATS = [73, 75]
            lats = LATS
            for i in range(5):
                lons = LONS
                chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                chlTile = computeWeightedMean(chlTile)
                siconcTile = computeWeightedMean(siconcTile)
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(4):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                    siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                    chlTile = computeWeightedMean(chlTile)
                    siconcTile = computeWeightedMean(siconcTile)
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                lats = np.add(lats, [2,2])
        
            LONS = [160, 170]
            LATS = [73, 75]
            lats = LATS
            for i in range(5):
                lons = LONS
                chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                chlTile = computeWeightedMean(chlTile)
                siconcTile = computeWeightedMean(siconcTile)
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(2):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = model_chl_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                          lon=slice(lons[0],lons[1]))
                    siconcTile = model_siconc_period_JJA.sel(lat=slice(lats[0],lats[1]),
                                                lon=slice(lons[0],lons[1]))
                    chlTile = computeWeightedMean(chlTile)
                    siconcTile = computeWeightedMean(siconcTile)
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                lats = np.add(lats, [2,2])
     
            year += 1
 
    
    else:
        
        for nyear in range(NYEARS):
    
            model_chl_period = choose_period(model_chl,year,year+1,ProlepticGregorian)
            model_chl_period_JJA = get_summer(model_chl_period)

            model_siconc_period = choose_period(model_siconc,year,year+1,ProlepticGregorian)
            model_siconc_period_JJA = get_summer(model_siconc_period)
            
            model_chl180 = model_chl_period_JJA.copy()
            model_siconc180 = model_siconc_period_JJA.copy()
            model_chl180.coords['longitude'] = (model_chl180['longitude'] + 180) % 360 - 180
            model_siconc180.coords['longitude'] = (model_siconc180['longitude'] + 180) % 360 - 180

            # Around the North Pole

            LONS = [180, 190]
            LATS = [84, 86]
            lats = LATS
            for i in range(2):
                lons = LONS
                chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                chlTile = computeWeightedMean(chlTile,grid_label='gn')
                siconcTile = computeWeightedMean(siconcTile,grid_label='gn')
                
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(35):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                    siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                    chlTile = computeWeightedMean(chlTile,grid_label='gn')
                    siconcTile = computeWeightedMean(siconcTile,grid_label='gn')
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                
                lats = np.add(lats, [2,2])

            # Above Bering Strait
        
            LONS = [-180, -170]
            LATS = [73, 75]
            lats = LATS
            for i in range(5):
                lons = LONS
                chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                chlTile = computeWeightedMean(chlTile,grid_label='gn')
                siconcTile = computeWeightedMean(siconcTile,grid_label='gn')            
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(4):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                    siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                    chlTile = computeWeightedMean(chlTile,grid_label='gn')
                    siconcTile = computeWeightedMean(siconcTile,grid_label='gn')
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                    
                lats = np.add(lats, [2,2])
        
            LONS = [160, 170]
            LATS = [73, 75]
            lats = LATS
            for i in range(5):
                lons = LONS
                chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                chlTile = computeWeightedMean(chlTile,grid_label='gn')
                siconcTile = computeWeightedMean(siconcTile,grid_label='gn')
                chlTiles.append(chlTile['chl'].values)
                siconcTiles.append(siconcTile['siconc'].values)
                for j in range(2):                 
                    lons = np.add(lons, [10, 10])
                    chlTile = choose_square(lats=lats,lons=lons,data=model_chl_period_JJA,data180=model_chl180)
                    siconcTile = choose_square(lats=lats,lons=lons,data=model_siconc_period_JJA,data180=model_siconc180)
                    chlTile = computeWeightedMean(chlTile,grid_label='gn')
                    siconcTile = computeWeightedMean(siconcTile,grid_label='gn')                
                    chlTiles.append(chlTile['chl'].values)
                    siconcTiles.append(siconcTile['siconc'].values)
                    
                lats = np.add(lats, [2,2])
     
            year += 1
    
    return chlTiles, siconcTiles


# -------------------------------------------------------------------------------
# Visualize data
# -------------------------------------------------------------------------------

def polarCentral_set_latlim(lat_lims, ax):
    """
    author: Anne Fouilloux
    """
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

def plot_ArcticSpatialData(ax, data, title, label, levels, colormap,
                           grid_label='gr', white_coastlines=False,
                           add_colorbar=True,add_labels=True):
    
    # Plots Arctic Spatial data for data with and without 
    # longitude and latitude as dimensions.
    
    if white_coastlines:
        ax.coastlines(color='white')
    else:
        ax.coastlines()
    ax.gridlines()
    polarCentral_set_latlim([60,90], ax)
    if grid_label=='gr':
        data.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(),
                    cmap=colormap,levels=levels,
                    add_colorbar=True,add_labels=True, 
                    cbar_kwargs=dict(label=label))
    else:
        ds1 = data.where((data.latitude > 60) & (data.longitude > 20) & (data.longitude < 340) ).squeeze()
        ds2 = data.copy()
        ds2.coords['longitude'] = (ds2['longitude'] + 180) % 360 - 180
        ds2 = ds2.where((ds2.latitude > 60) & (ds2.longitude > -160) & (ds2.longitude < 160) ).squeeze()
        ds1.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(),
                    x = 'longitude',y='latitude',
                    cmap=colormap,levels=levels,
                    add_colorbar=False)
        ds2.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(),
                    x = 'longitude',y='latitude',
                    cmap=colormap,levels=levels,
                    add_colorbar=add_colorbar,add_labels=add_labels, 
                    cbar_kwargs=dict(label=label))
    ax.set_title(title)


def plot_RedSquare(axes, lats, lons):
    
    # Plots a red square for given latitude and longitude
    # coordinates on a map on a given axis
    
    LONS = [lons[0], lons[0], lons[1], lons[1]]
    LATS = [lats[0], lats[1], lats[1], lats[0]]
    ring = LinearRing(list(zip(LONS, LATS)))
    for ax in axes:
        ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='red',linewidth=2)

def plot_RedSquares_SIarea(axes):
    
    # Plots red square in specified grid area around the North Pole
    # and above the Bering Strait
    
    LONS = [-180, -170]
    LATS = [84, 86]
    lats = LATS
    for i in range(2):
        lons = LONS
        plot_RedSquare(axes=axes,lats=lats,lons=lons)
        for j in range(35):                 
            lons = np.add(lons, [10, 10])
            plot_RedSquare(axes=axes,lats=lats,lons=lons)
        lats = np.add(lats, [2,2])

    LONS = [-180, -170]
    LATS = [73, 75]    
    lats = LATS
    for i in range(5):
        lons = LONS
        plot_RedSquare(axes=axes,lats=lats,lons=lons)
        for j in range(4):                 
            lons = np.add(lons, [10, 10])
            plot_RedSquare(axes=axes,lats=lats,lons=lons)  
        lats = np.add(lats, [2,2])
    
    LONS = [160, 170]
    LATS = [73, 75]
    lats = LATS
    for i in range(5):
        lons = LONS
        plot_RedSquare(axes=axes,lats=lats,lons=lons)
    
        for j in range(2):                 
            lons = np.add(lons, [10, 10])
            plot_RedSquare(axes=axes,lats=lats,lons=lons)
        lats = np.add(lats, [2,2])

def plotCorr(chlTiles,siconcaTiles,NYEARS):
    
    # Plots the output of the correlation between chlorophyll and sea ice.
    # Gives color according to location: 
    # green above Bering Strait, blue around north pole, lighter color further south
    
    for nyear in range(NYEARS):
    
        for j in range(36):
            if nyear==0 and j==0:
                plt.scatter(siconcaTiles[nyear*112+j],chlTiles[nyear*112+j], 
                            color='cornflowerblue', alpha = 0.5,label='Around North Pole (southward)')
                plt.scatter(siconcaTiles[nyear*112+36+j],chlTiles[nyear*112+36+j], 
                            color='cornflowerblue', alpha = 1,label='Around North Pole (northward)')
            else:
                plt.scatter(siconcaTiles[nyear*112+j],chlTiles[nyear*112+j], 
                            color='cornflowerblue', alpha = 0.5)
                plt.scatter(siconcaTiles[nyear*112+36+j],chlTiles[nyear*112+36+j], 
                            color='cornflowerblue', alpha = 1)
        
        for j in range(8):
            if nyear==0 and j==0:
                plt.scatter(siconcaTiles[nyear*112+72+j],chlTiles[nyear*112+72+j], color='darkcyan', alpha = 0.25)
                plt.scatter(siconcaTiles[nyear*112+72+8+j],chlTiles[nyear*112+72+8+j], 
                            color='darkcyan', alpha = 0.5,label='Above Bering Strait (southward)')
                plt.scatter(siconcaTiles[nyear*112+72+8*2+j],chlTiles[nyear*112+72+8*2+j], color='darkcyan', alpha = 0.75)
                plt.scatter(siconcaTiles[nyear*112+72+8*3+j],chlTiles[nyear*112+72+8*3+j], 
                            color='darkcyan', alpha = 1, label='Above Bering Strait (northward)')
                plt.scatter(siconcaTiles[nyear*112+72+8*4+j],chlTiles[nyear*112+72+8*4+j], color='darkcyan', alpha = 1)
            else:
                plt.scatter(siconcaTiles[nyear*112+72+j],chlTiles[nyear*112+72+j], color='darkcyan', alpha = 0.25)
                plt.scatter(siconcaTiles[nyear*112+72+8+j],chlTiles[nyear*112+72+8+j], color='darkcyan', alpha = 0.5)
                plt.scatter(siconcaTiles[nyear*112+72+8*2+j],chlTiles[nyear*112+72+8*2+j], color='darkcyan', alpha = 0.75)
                plt.scatter(siconcaTiles[nyear*112+72+8*3+j],chlTiles[nyear*112+72+8*3+j], color='darkcyan', alpha = 1)
                plt.scatter(siconcaTiles[nyear*112+72+8*4+j],chlTiles[nyear*112+72+8*4+j], color='darkcyan', alpha = 1)
        