import glob
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib as mpl
import cmocean
from matplotlib import cm
import warnings

bin_col_names_2010_2020 = ['5.0118723e-09', '5.6234133e-09', '6.3095734e-09',
       '7.0794578e-09', '7.9432823e-09', '8.9125094e-09', '1.0000000e-08',
       '1.1220185e-08', '1.2589254e-08', '1.4125375e-08', '1.5848932e-08',
       '1.7782794e-08', '1.9952623e-08', '2.2387211e-08', '2.5118864e-08',
       '2.8183829e-08', '3.1622777e-08', '3.5481339e-08', '3.9810717e-08',
       '4.4668359e-08', '5.0118723e-08', '5.6234133e-08', '6.3095734e-08',
       '7.0794578e-08', '7.9432823e-08', '8.9125094e-08', '1.0000000e-07',
       '1.1220185e-07', '1.2589254e-07', '1.4125375e-07', '1.5848932e-07',
       '1.7782794e-07', '1.9952623e-07', '2.2387211e-07', '2.5118864e-07',
       '2.8183829e-07', '3.1622777e-07', '3.5481339e-07', '3.9810717e-07',
       '4.4668359e-07', '5.0118723e-07', '5.6234133e-07', '6.3095734e-07',
       '7.0794578e-07']
additional_2010_2020 = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'UFCPC','CPC3010','Ntot','unknown4','unknown5', 'unknown6',
              'unknown7','unknown8']  

dict_season_to_season_long_name = {'Summer':'Summer (JJAS)', 'Slow build up': 'Slow build up  (ONDJ)', 
                                   'Arctic Haze':'Arctic Haze (FMAM)','':''}               
          
       
def get_columns_2010_2020(additional_2010_2020, bin_col_names_2010_2020):
    """[YYYY, MM, DD, HH, mm, UF(?)CPC, CPC3010, N_int, bin1:end, numflag]. Sizes in m and dN/dlogdp in cm-3. This data is level 2."""
    columns = additional_2010_2020 + bin_col_names_2010_2020 + ['flag']
    return columns  
    
def load_and_append_2010_2020(inpath, name_in_file):
    """[YYYY, MM, DD, HH, mm, UF(?)CPC, CPC3010, N_int, bin1:end, numflag]. Sizes in m and dN/dlogdp in cm-3. This data is level 2."""    
    cols = get_columns_2010_2020(additional_2010_2020, bin_col_names_2010_2020)    
    print(cols)
    DFs = []
    folder = glob.glob(inpath+str(name_in_file)+'*.dat')
    folder.sort()
    for file in folder: 
        print(file)
        ds = pd.read_csv(file, sep='\s+',index_col=False, skiprows=1, names=cols)     
        ds[['Year', 'Month', 'Day', 'Hour', 'Minute']] = ds[['Year', 'Month', 'Day', 'Hour', 'Minute']].astype(int)
        ds['DateTime'] = ds[['Year', 'Month', 'Day', 'Hour', 'Minute']].apply(lambda s : datetime.datetime(*s),axis = 1)
        ds = ds.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
        ds = ds.set_index('DateTime')        
        print("Size without flags removed: "+str(len(ds)))
        ds = ds[ds.loc[:,'flag'] != 0.999] #remove these
        print("Size flags removed: "+str(len(ds)))  
        
        DFs.append(ds)
    return DFs
    
def concat_df_2010_2020(df_list):    
    appended_data = []
    for i in range(len(df_list)):         
        df = df_list[i]     
#         if str(df.index.dtype) != 'datetime64[ns]':                
#             datetime = pd.DataFrame({'year': df.iloc[:, 0],
#                     'month': df.iloc[:, 1],
#                     'day': df.iloc[:, 2],
#                     'hour': df.iloc[:, 3],
#                     'minute': df.iloc[:, 4]})
#             df.index = pd.to_datetime(datetime)         
#             df.drop(df.columns[[0,1,2,3,4]], axis=1, inplace=True)          
#         bin_col_names_floats = [float(i)*10**9 for i in bin_col_names_2010_2020]
#         cols = np.around(bin_col_names_floats, decimals=3)
#         cols = np.asarray(cols)         
#         df = df[bin_col_names_2010_2020]        
#         df.columns = cols
        appended_data.append(df)        
    appended_data = pd.concat(appended_data, sort=True)   
    add_cols = ['UFCPC','CPC3010','Ntot','unknown4','unknown5', 'unknown6',
              'unknown7','unknown8']
    cols = get_columns_2010_2020(add_cols, bin_col_names_2010_2020)    
    appended_data = appended_data.reindex(cols, axis = 1)
    appended_data.drop(['unknown4','unknown5', 'unknown6','unknown7','unknown8'], axis = 1, inplace=True)
    return appended_data
    
def remove_cols_with_same_value(df):    
    for col in df.columns:
            number_uniques_in_col = len(df[col].unique())
            if number_uniques_in_col == 1:
                df = df.drop(columns = [col])      
    return df
    
def resample_dfs(dict_years_to_df, name):
    df = dict_years_to_df[name]    
    df = remove_cols_with_same_value(df)
    if (name == "2000_2005") or (name == "2006_2009"):
        df = clean_df_2000_2009(df)
    df_hourly = df.resample('60T').median() #the size distribution is every 30 minutes 
    dict_years_to_df[name] = df_hourly
    return dict_years_to_df
    
def df_to_xr(dict_years_to_df, name, name_to_start_size_bin, name_to_end_size_bin):
    start_size_bin_col = name_to_start_size_bin[name] 
    end_size_bin_col = name_to_end_size_bin[name]  
    df = dict_years_to_df[name]   
    df = df.loc[:, start_size_bin_col:end_size_bin_col]    
    df.index = pd.to_datetime(df.index)  
    df.index.names = ['DateTime']
    df.columns = df.columns.astype(float)
    df.columns.name = 'dia'
    xr = df.stack().to_xarray().resample({'DateTime':'h'}).median()
    xr.name=name
    return xr
    
def create_colourmesh(xrs):
    fig, ax = plt.subplots(figsize=(20,8))
    c = xrs.plot.pcolormesh(x=xrs.dims[0],y=xrs.dims[1], ax=ax, add_labels=True,extend='max', vmin=0, vmax=400,
                      cbar_kwargs=dict(orientation='horizontal', pad=0.15, shrink=1, 
                                       label='dN/dlog$\,$D$_{\mathrm{p}}$ [cm$^{-3}$]'))
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))       
    plt.yscale('log')
    plt.ylim(5,10**3)
    plt.ylabel('Diameter [nm]')
    plt.xlabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)   
    ax.spines['bottom'].set_visible(False)       
    plt.show()
    return fig
    
def get_unique_years(xrs):
    datetimes = list(xrs['DateTime'].values)
    years = [x.astype('datetime64[Y]').astype(int) + 1970 for x in datetimes]
    unique_years = sorted(list(set(years)))
    return unique_years
    
def creat_distribution_plots(xrs, season, accumulation_min=75):
    fig, ax = plt.subplots(figsize=(8, 5))    
    years = get_unique_years(xrs)
    print("years: "+str(years))
    for year in years:
        norm = mpl.colors.Normalize(vmin=years[0], vmax=years[-1])
        cmap = cmocean.cm.deep
        m = cm.ScalarMappable(norm=norm, cmap=cmap)        
        xrs_year = xrs.loc[str(year)+'-01-01':str(year+1)+'-12-31']
        xrs_year.median(dim='DateTime').plot(label=str(year), c=m.to_rgba(year)) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            xrs_year.quantile(0.75, dim='DateTime').plot(alpha = 0.1, add_legend=False, c=m.to_rgba(year))
            xrs_year.quantile(0.25, dim='DateTime').plot(alpha = 0.1, add_legend=False, c=m.to_rgba(year))
    plt.xscale('log')
    plt.xlabel('Diameter [nm]')
    plt.ylabel('dN/dlogD [cm$^{-3}$]')
    plt.legend(title=str(dict_season_to_season_long_name[season])+'\nmedians', frameon=False)    
    plt.xlim(1,10**3)
    plt.ylim(0,300) #650 if means
    ax.axvline(x=10.0, ls='--',c='k',alpha=0.5)
    ax.axvline(x=398.107, ls='--',c='k',alpha=0.5)    
    ax.axvline(x=accumulation_min, ls='-', c='k')    
    plt.title('')
    plt.show()
    return fig