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
import scipy as sc
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy.stats as st
import seaborn as sns
from sklearn.metrics import davies_bouldin_score


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
       
bin_col_names_DMPS = bin_col_names_2010_2020.copy()
       
additional_2010_2020 = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'UFCPC','CPC3010','Ntot','unknown4','unknown5', 'unknown6',
              'unknown7','unknown8']  

dict_season_to_season_long_name = {'Summer':'Summer (JJAS)', 'Slow build up': 'Slow build up  (ONDJ)', 
                                   'Arctic Haze':'Arctic Haze (FMAM)','':''}               
          
       
def get_columns_2010_2020(additional_2010_2020, bin_col_names_2010_2020):
    """[YYYY, MM, DD, HH, mm, UF(?)CPC, CPC3010, N_int, bin1:end, numflag]. Sizes in m and dN/dlogdp in cm-3. This data is level 2."""
    columns = additional_2010_2020 + bin_col_names_2010_2020 + ['flag']
    return columns  
    
def get_bin_column_string_list():
    '''Return a list of strings with bin column namnes, i.e. sizes in DMPS data.'''
    global bin_col_names_DMPS
    bin_col_list = bin_col_names_DMPS.copy()
    return bin_col_list    
    
def load_and_append_DMPS(inpath, name_in_file):
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
    
def get_bins(bin_col_list):
    '''A function that takes in a list of non-rounded bin midpoint diameters as strings, \
    and rounds the values. A list of mid point diameteters with fewer decimals is returned.'''
    
    # Turn string items in list to floats and round to 3 decimals
    bin_col_list_floats = [float(i)*10**9 for i in bin_col_list]
    bin_cols = np.around(bin_col_list_floats,decimals=3)
    bin_cols = np.asarray(bin_cols)
    
    # Loop over all list items and change them back to string items.
    bin_cols = [str(x) for x in bin_cols] 
    
    return bin_cols
    
def renameDpColumns(df_DMPS, bin_col_list):
    '''Rename the midpoint bin column headings in bin_col_list \
    in DMPS dataframe to rounded values.'''
    
    # Get the list of rounded bin midpoint diameters 
    bin_cols = get_bins(bin_col_list)

    # Rename all columns 
    dict_cols_to_goodnames = dict(zip(bin_col_list, bin_cols))
   
    df_DMPS = df_DMPS.rename(dict_cols_to_goodnames, axis=1)
    
    return df_DMPS
    
def concat_df_DMPS(df_list):
    '''Drop columns that are not used and make one large dataframe containg all data from list of dataframes.'''
    appended_data = []
    for i in range(len(df_list)):         
        df = df_list[i]     
        appended_data.append(df)  
        
    appended_data = pd.concat(appended_data, sort=True)   
    add_cols = ['UFCPC','CPC3010','Ntot','unknown4','unknown5', 'unknown6',
              'unknown7','unknown8']
    cols = get_columns_2010_2020(add_cols, bin_col_names_2010_2020)    
    appended_data = appended_data.reindex(cols, axis = 1)
    appended_data.drop(['unknown4','unknown5', 'unknown6','unknown7','unknown8'], axis = 1, inplace=True)
    return appended_data
    

def getFloatDiameterListAndArray():
    '''Take in list of non-rounded midpoint diameters as strings. \
    Return list and array with numeric values.'''
    global bin_col_names_DMPS
    bin_col_list = bin_col_names_DMPS.copy()
    
    diameterList = [float(i) for i in bin_col_list]
    diameters = np.asarray(diameterList)
    
    return diameterList, diameters 

def calcNtot(diameters, df, GMDs):
    '''Integrate the log-normal ditribution for given diameters in dataframe df. \
    GMS´Ds is the list of all diameters as strings'''
    
    # Create array to store upper bin boudaries
    upperBoundaries = np.empty(0)
    diameter_list = list(diameters)

    # Create array to store the number concentration in each bin
    dNs = np.empty(0)
    upperLimits = []

    for Dp in range(len(diameter_list)-1):

        # Calulate the upper bin from the geo mean of the midpoint diamters as they are equally spaced on a log scale
        upperLimits.append(np.sqrt( diameter_list[Dp] * diameter_list[Dp+1] ) )

    upperLimits = np.array(upperLimits)

    # Calulate the endpoints, ie the first lower limit and the last upper limit
    firstLimit = diameter_list[0]**2 / upperLimits[0] # This is actually the first lower limit, but its needed for the first binwidth
    lastLimit = diameter_list[-1]**2 / upperLimits[-1]

    upperBoundaries = np.insert(upperLimits, 0, firstLimit) 
    upperBoundaries = np.append(upperBoundaries, lastLimit)

    # Calculate dlogDp from the boundaries
    dlogDp = np.log10(upperBoundaries[1:]) - np.log10(upperBoundaries[:-1])

    # Calculate the particle concentration in each bin (dN) by multiplying dNdlogD with dlogD

    lenDiam = len(diameters)
    #idx = len(diameter_list)-lenDiam+3
    idx = len(GMDs)-lenDiam+3
    
    dNdlogDp = df.iloc[:,idx:-1]
    
    dNs = dNdlogDp*(dlogDp)
    ntotCalc = dNs.sum(axis=1)    

    df_ntotCalc = df.copy(deep = True)
    
    # Add column containing the calulated N_tot in dataframe
    df_ntotCalc['NtotCalc'] = ntotCalc
    return df_ntotCalc

def compareIntegration(N_calc,N_meas):
    '''Make sure that calculated and measured total \
    particle number concentrations agrees, i.e. that the\
    function calcNtot works.'''
    
    varx = N_calc.copy()
    vary = N_meas.copy()
    
    mask = ~np.isnan(varx) & ~np.isnan(vary)
    res = sc.stats.linregress(varx[mask], vary[mask])

    print(f"R-squared: {res.rvalue**2:.6f}")

    plt.plot(varx,
             vary,
             'o', label='original data')
    plt.plot(varx,
             res.intercept + res.slope*varx,
             'r-', label='fitted line')
    plt.legend()  
    plt.ylabel('Measured $N_{tot}$ [#/$cm^3]$')
    plt.xlabel('Calculated $N_{tot}$ [#/$cm^3]$')
    print('Intercept:',res.intercept)
    print('Slope:',res.slope)
    
    return 

def create_normalised_df(dataFrame, start_size_bin_col='5.012', end_size_bin_col='707.946'):
    '''Normalize the size distributions.'''
    df = dataFrame.copy()
    
    # Make column headings for the normalized size distributions
    n_vars = ['norm'+str(df.loc[:, start_size_bin_col:end_size_bin_col].columns.tolist()[i]) for i in range(0, df.loc[:, start_size_bin_col:end_size_bin_col].shape[1])]

    
    # Divide by maximum to normaize 
    df[n_vars] = df.loc[:, start_size_bin_col:end_size_bin_col].div(df.loc[:, start_size_bin_col:end_size_bin_col].max(axis=1), axis=0)
    
    df = df.loc[df.loc[:,start_size_bin_col:end_size_bin_col].dropna().index]
    
    Datetime_index = df.index    
    df.reset_index(drop=True, inplace=True)
    
    start_size_normbin_col = 'norm'+str(start_size_bin_col)
    end_size_normbin_col = 'norm'+str(end_size_bin_col)    
    
    df_norm = df.loc[:, start_size_normbin_col:end_size_normbin_col].copy()
    df_norm.index = Datetime_index
    df.index = Datetime_index
    
    return df, df_norm
    
def perform_clustering(df_normarlised, no_clusters):
    ''' K-means'''
    kmeans = KMeans(init="k-means++", n_clusters=no_clusters).fit(df_normarlised) #Compute k-means clustering.
    labels = kmeans.labels_
    centres = kmeans.cluster_centers_

    # Predict the closest cluster each sample in X belongs to and add a column in the dataframe called clusters
    df_normarlised['clusters'] = kmeans.predict(df_normarlised) 
    
    # #-----Test
    # x = df_normarlised.copy()
    # x = df_normarlised.drop('clusters', inplace=True, axis=1)
    # silhouette_avg = skm.silhouette_score(x.values, labels)
    
    # clusters = np.unique(df_normarlised['clusters'].values)
    # print("For n_clusters =", no_clusters,'and unique no of clusters:',clusters,
    # "The average silhouette_score is :", silhouette_avg)
    # https://medium.com/@haidarlina4/k-means-clustering-with-scikit-learn-70c805230646
    # #---------------
    
    df_normalized_copy = df_normarlised.copy()
    #print(df_normalized_copy['clusters'].unique())    
    
    # Start cluster numbering from 1
    df_normalized_copy['clusters'] = df_normalized_copy['clusters']+1
    #print(df_normalized_copy['clusters'].unique())
    
    dict_max_columns_unordered, dict_max_columns = produce_dicts_for_sorting(df_normalized_copy)
    final_mapping = connect_dicts(dict_max_columns_unordered, dict_max_columns)
    
    #print(df_normalized_copy['clusters'].unique())
    #print('Final mapping =',final_mapping)
    
    df_normalized_copy['clusters'] = df_normalized_copy['clusters'].map(final_mapping)
    
    #print(df_normalized_copy['clusters'].unique())

    return df_normalized_copy

def computeAvergeSilScoreAndDunnIndex(df_norm_clustered_1h_mean,clusters):
    df = df_norm_clustered_1h_mean.copy()
    df.drop('clusters', inplace=True, axis=1)
    X = df.values
    
    #Compute k-means clustering. Clusterer is the kmeans model.
    clusterer = KMeans(init="k-means++", n_clusters=len(clusters)).fit(df) 
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg = skm.silhouette_score(X, cluster_labels)
    print("For n_clusters =", len(clusters),
    "The average silhouette_score is :", silhouette_avg)
    
    # Compute the silhouette scores for each sample (very time consuming)
    #sample_silhouette_values = skm.silhouette_samples(X, cluster_labels)  
    #Centroids:
    #print(clusterer.cluster_centers_)
    centers = np.array(clusterer.cluster_centers_)

    #plt.plot()
    #plt.title('k means centroids')
    
    #-----------Compute dunn index-------------------------------------
    # https://www.geeksforgeeks.org/dunn-index-and-db-index-cluster-validity-indices-set-1/

    #-------------------------------------------------------------
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
 

    return silhouette_avg


def produce_dicts_for_sorting(df):
    df_ = df.groupby('clusters').mean()
    
    #print(np.unique(df['clusters'].values,return_counts=True))
    
    # Applying the anonymous lambda function
    max_col_index = df_.apply(lambda x: x.argmax(), axis=1)
    
    # Find the column which contains the max of the clusters
    dict_max_columns_unordered = dict(zip(df_.index, max_col_index))
    
    #print('dict_max_columns_unordered',dict_max_columns_unordered)
    
    # Sort so that cluster 1 has peak for the smallest diameter and so on
    max_col_index_sorted = sorted(max_col_index.values)
    dict_max_columns = dict(zip(df_.index, max_col_index_sorted))
   
    #print('dict_max_columns',dict_max_columns)
    
    #print('dict_max_columns:',dict_max_columns)
    return dict_max_columns_unordered, dict_max_columns
    
def connect_dicts(dict_max_columns_unordered, dict_max_columns):
    final_mapping = {}
    
    for k,v in dict_max_columns_unordered.items():
        ordered_v = dict_max_columns[k]
        ordered_v_list = list(dict_max_columns.values())
        new_key = ordered_v_list.index(v)+1    
        final_mapping[k] = new_key
    #print("the mapping from clustering to ordered clusters (by mode): "+str(final_mapping))
    
    return final_mapping
    
def remove_cols_with_same_value(df):    
    for col in df.columns:
            number_uniques_in_col = len(df[col].unique())
            if number_uniques_in_col == 1:
                df = df.drop(columns = [col])      
    return df

def checkUniqueModeDiam(df_norm_clustered,n_clusters):
    '''Check if number of n_clusters = the resulting number of clusters'''   
    assignedClusters = len(np.unique(df_norm_clustered['clusters'].values))
    if assignedClusters != n_clusters:
        print('Some clusters peak for the same diameter when number of clusters = ',
              n_clusters,'. Consider choosing other number of clusters.')
    if assignedClusters == n_clusters:
        print('OK! Clusters peak for different diameter when number of clusters = ',
              n_clusters)
    
    return

def plotClustersNormalized(df_norm_clustered_1h_mean, diameters,
                           df_norm_clustered_1h_mean_mean, df_norm_clustered_1h_std,
                          df_norm_clustered_1h_mean_median):
    '''Plot the reults from the cluster analysis, median and mean + one std.'''
    fig, ax = plt.subplots(figsize=(10,4))

    # Defining the number of clusters unique in the sence that peak diameter is different
    clusters = np.unique(df_norm_clustered_1h_mean['clusters'].values)

    for i in range(len(clusters)):
        cluster = clusters[i]
        ax.plot(diameters[1:-2]*10**9, df_norm_clustered_1h_mean_mean.iloc[i,:].values,
                '-', label='cluster: '+str(cluster))
        ax.fill_between(diameters[1:-2]*10**9, 
                        df_norm_clustered_1h_mean_mean.iloc[i,:].values + df_norm_clustered_1h_std.iloc[i,:].values,                    
                        df_norm_clustered_1h_mean_mean.iloc[i,:].values - df_norm_clustered_1h_std.iloc[i,:].values, alpha=0.2)


        # Plot the median to see similarity
        ax.plot(diameters[1:-2]*10**9, df_norm_clustered_1h_mean_median.iloc[i,:].values, ':')
        #ax.set_xticks(bin_cols[::5])
        ax.set_xscale('log')
        ax.set_ylim(0,1.1)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_linewidth(3)


    plt.legend(frameon=False,bbox_to_anchor=(.9, 0.1))    
    plt.xlabel('Dp [nm]')
    plt.title('1h mean clusters (mean+ 1std)')
    plt.ylabel('Normalised concentration')
    plt.show()
    return

def makeTrendPlotsMean(dfMean, dfStd, xL, yL, tL):
    fig, axs = plt.subplots(1, figsize=(8, 5))
    axs.plot(dfMean.index, 
             dfMean.values,
             'o-', label= 'Mean')
    min_std = dfMean.values - dfStd.values
    #min_std[min_std<0]=0
    
    axs.fill_between(dfMean.index,
                dfMean.values + dfStd.values,
                min_std,
                alpha=0.2,label= '+/-1$\sigma$')
    axs.legend(frameon=False)
    axs.set_xlabel(xL)
    axs.set_ylabel(yL)
    axs.set_title(tL) 
    return fig, axs

def makeTrendPlotsMedian(dfMedian, dfUpperQ, dfLowerQ, UQ, LQ, xL, yL, tL):
    fig, axs = plt.subplots(1, figsize=(8, 5))
    axs.plot(dfMedian.index, 
             dfMedian.values,
             'ro-', label= 'Median')
    axs.fill_between(dfMedian.index,
                dfUpperQ,
                dfLowerQ,
                color ='r',alpha=0.2,label= str(LQ)+'-'+str(UQ)+' percentiles')
    axs.legend(frameon=False)
    axs.set_xlabel(xL)
    axs.set_ylabel(yL)
    axs.set_title(tL) 
    return #fig, axs

def makeTrendPlotsMedian2(dfMedian, dfUpperQ, dfLowerQ, UQ, LQ, xL, yL, tL):
    fig, axs = plt.subplots(1, figsize=(8, 5))
    axs.plot(dfMedian.index, 
             dfMedian.values,
             'ro-', label= 'Median $|CPC_{UF}-CPC$|')
    axs.fill_between(dfMedian.index,
                dfUpperQ,
                dfLowerQ,
                color ='r',alpha=0.2,label= str(LQ)+'-'+str(UQ)+' percentiles')
    axs.legend(frameon=False)
    axs.set_xlabel(xL)
    axs.set_ylabel(yL)
    axs.set_title(tL) 
    return fig, axs


 
def calcFriedmanDiaconisNo(x):
    '''Calculate the optimal number of bins in histogram
    x is np.array'''
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    fd_no_of_bins = round((x.max() - x.min()) / bin_width)
    #print("Freedman–Diaconis number of bins:", fd_no_of_bins)
    return fd_no_of_bins
    
    
def plotHistWithKDE(x,label):
    '''Plot a histogram with KDE'''
    # Calculate optimum bin number with Friedman-Diaconis Number
    fd_no_of_bins = calcFriedmanDiaconisNo(x)
    # Plot histogram with PDF  => area under graph =1
    # Add a kde = True give a kernel density estimate to smooth the histogram, 
    # providing complementary information about the shape of the distribution:
    h1 = sns.histplot(x, bins = fd_no_of_bins, kde=True,label=label,fill=False)
    plt.legend(frameon = False,bbox_to_anchor=(1.02, 1), borderaxespad=0)

    return

 
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