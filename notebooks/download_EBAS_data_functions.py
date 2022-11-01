from pydap.client import open_dods, open_url
from netCDF4 import num2date
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def read_EBAS_data(ST_STATION_CODE, FT_TYPE, RE_REGIME_CODE, MA_MATRIX_NAME, CO_COMP_NAME, DS_RESCODE, FI_REF,
                   ME_REF, DL_DATA_LEVEL):
    open_string = 'http://dev-ebas-pydap.nilu.no/'+ST_STATION_CODE+'.'+FT_TYPE+'.'+RE_REGIME_CODE+'.'+MA_MATRIX_NAME+'.'+CO_COMP_NAME+'.'+DS_RESCODE+'.'+FI_REF+'.'+ME_REF+'.'+DL_DATA_LEVEL+'.dods'
    print("open string: "+str(open_string))    
    ds = open_dods(open_string) 
    print("ds: "+str(ds))
    print("ds type: "+str(type(ds)))
    print("ds keys: "+str(list(ds.keys())))
    return ds
	
def convert_EBAS_ds(ds, var, given_name=None, calendar='gregorian'):    
    EBAS_data = ds[var]
    print("Dimensions of the datasettype: "+str(EBAS_data.dimensions))
    print("Shape of the dataset type: "+str(EBAS_data.shape))
    print("Number of dimensions: "+str(EBAS_data.ndim))
    
    dimensions_with_size_greater_than_1 = []
    for ndim in range(EBAS_data.ndim):
        dim = EBAS_data.shape[ndim]
        print("dimension number "+str(ndim+1)+": size "+str(dim))
        if int(dim) > 1:
            dimensions_with_size_greater_than_1.append(dim)
    print("dimensions with size greatetr than 1: "+str(dimensions_with_size_greater_than_1))    
    
    if len(dimensions_with_size_greater_than_1) == 1:
        variables = EBAS_data.data
        print("Example data: "+str(variables[0].flatten()))      
        time = num2date(EBAS_data.time, units='days since 1900-01-01 00:00:00',
        calendar=calendar)    
        df_NILU = pd.DataFrame(variables[0].flatten(), index=time)
        df_NILU.columns = [given_name]
        df_NILU[given_name] = pd.to_numeric(df_NILU[given_name], errors='ignore')    
        df_NILU.index = [datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') for x in df_NILU.index] 
        
    if len(dimensions_with_size_greater_than_1) == 2:  
        dNdlogDp= EBAS_data[var].data #get normalised size distribution in dNdlogDp    
        tim_dmps = num2date(EBAS_data.time.data,units='days since 1900-01-01 00:00:00', calendar=calendar) #get time in datatime format using function from netCDF4 package
        dp_NILU = EBAS_data.D.data # get diameter vector
        df_NILU = pd.DataFrame(dNdlogDp.T, index=tim_dmps, columns=dp_NILU) # make DataFrame to simplify the handling of data        
        df_NILU.index = [datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') for x in df_NILU.index] 
        df_NILU = df_NILU.sort_index()
        
    return df_NILU
		
def make_plot(df_NILU, var, ymax):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_NILU.index, df_NILU[var].values, 'o')
    ax.set_ylim(0,ymax)
    plt.show()
    return fig
