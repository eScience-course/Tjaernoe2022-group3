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

def importData(path):
    glob.glob(path+'\\*')
    return    
       
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
    # print(cols)
    DFs = []
    folder = glob.glob(inpath+str(name_in_file)+'*.dat')
    folder.sort()
    for file in folder: 
        #print(file)
        ds = pd.read_csv(file, sep='\s+',index_col=False, skiprows=1, names=cols)     
        ds[['Year', 'Month', 'Day', 'Hour', 'Minute']] = ds[['Year', 'Month', 'Day', 'Hour', 'Minute']].astype(int)
        ds['DateTime'] = ds[['Year', 'Month', 'Day', 'Hour', 'Minute']].apply(lambda s : datetime.datetime(*s),axis = 1)
        ds = ds.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
        ds = ds.set_index('DateTime')        
        #print("Size without flags removed: "+str(len(ds)))
        ds = ds[ds.loc[:,'flag'] != 0.999] #remove these
        #print("Size flags removed: "+str(len(ds)))  
        
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
    fig = plt.figure(figsize=(4, 4))
    plt.plot(varx[mask],
             vary[mask],
             'o', label='original data')
    plt.plot(varx[mask],
             res.intercept + res.slope*varx[mask],
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
    kmeans = KMeans(init="k-means++", n_clusters=no_clusters,random_state = 100).fit(df_normarlised) #Compute k-means clustering.
    # Get inertia
    #  = Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    # Something weighting for sample size is not considered here, see inertia attribute sk.learn.Kmeans
    inertia = kmeans.inertia_
    #print('Intertia =', inertia)
    # Get labels of which cluster each sample belongs to. 
    labels = kmeans.labels_
    
    # Compute silhouette average score
    silhouette_avg = skm.silhouette_score(df_normarlised, labels)
    #print('silhouette score', silhouette_avg)
    # Predict the closest cluster each sample in X belongs to and add a column in the dataframe called clusters
    df_normarlised['clusters'] = kmeans.predict(df_normarlised) 
    

    df_normalized_copy = df_normarlised.copy()
    #print(df_normalized_copy['clusters'].unique())    
    
    # Start cluster numbering from 1 and add column with assigned cluster
    df_normalized_copy['clusters'] = df_normalized_copy['clusters']+1
    #print(df_normalized_copy['clusters'].unique())
    
    dict_max_columns_unordered, dict_max_columns = produce_dicts_for_sorting(df_normalized_copy)
    final_mapping = connect_dicts(dict_max_columns_unordered, dict_max_columns)
    
    #print(df_normalized_copy['clusters'].unique())
    #print('Final mapping =',final_mapping)
    
    df_normalized_copy['clusters'] = df_normalized_copy['clusters'].map(final_mapping)
    
    #print(df_normalized_copy['clusters'].unique())

    return silhouette_avg,inertia, df_normalized_copy
    
def optimizeClusters(df_hourly_norm_dropped):
    '''Compute the inertia (elbow method) and the average silouette score
    to choose optimal number of clusters.'''

    test_n_of_cluster = [2,3,4,5,6,7,8,9,10,11,12,13]
    av_sil_score_list = []
    inertia_list = []

       
    for n_clusters in test_n_of_cluster:
        
        df = df_hourly_norm_dropped.copy()
        #Compute the average silhouette score, inertia and cluster data
        av_sil_score, inertia, df_norm_clustered = perform_clustering(df, n_clusters)
        
        # Define the unique number of size distribution clusters in the sence that peak diameter is different.
        clusters = np.unique(df_norm_clustered['clusters'].values)

        # Add average siloutte scores to list for plotting later
        av_sil_score_list.append(av_sil_score)

        # Add inertia to intertia list
        inertia_list.append(inertia)
        
        # Check that the number of clusters resulting from the clustering procedure is equal to the variable ``n_clusters``, 
        # i.e. that the peak diameters of the clustered size distributions are unique. 
        #fu.checkUniqueModeDiam(df_norm_clustered,n_clusters)
        
    #Plot the inertia and average silhouette score
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot of interia
    ax.plot(test_n_of_cluster,
            inertia_list,
            color="red", 
            marker="o",
            label = 'Inertia')
    # set x-axis label
    ax.set_xlabel("Number of clusters", fontsize = 16)
    # set y-axis label
    ax.set_ylabel("Inertia",
                  color="red",
                  fontsize=16)

    # Add second y-axis for average Siloutte score
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(test_n_of_cluster,
             av_sil_score_list,
             color="blue",
             marker="x",
             label = 'Average Siloutte Score')
    ax2.set_ylabel("Average Silhoutette Score",color="blue",fontsize=16)
    
    plt.show()     
        
    return

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
                           df_norm_clustered_1h_mean_mean,
                           df_norm_clustered_1h_10q,
                           df_norm_clustered_1h_90q,
                           df_norm_clustered_1h_mean_median):
    '''Plot the reults from the cluster analysis, median and mean + percentiles.'''
    fig, ax = plt.subplots(figsize=(10,4))
    
    
    # Defining the number of clusters unique in the sence that peak diameter is different
    clusters = np.unique(df_norm_clustered_1h_mean['clusters'].values)
    
    # Define colormap
    n = len(clusters)
    colors = cm.Set2(np.linspace(0,1,n))
    
    # Plot the clusters and shaded spread in the data
    for i in range(len(clusters)):
        cluster = clusters[i]
        ax.plot(diameters[1:-2]*10**9, df_norm_clustered_1h_mean_median.iloc[i,:].values,
                '-', label='cluster: '+str(cluster),
                color=colors[i],
                linewidth = 4)
        ax.fill_between(diameters[1:-2]*10**9, 
                        #Plot percentiles
                        df_norm_clustered_1h_90q.iloc[i,:].values,
                        df_norm_clustered_1h_10q.iloc[i,:].values,
                        #For plotting std+mean
                        #df_norm_clustered_1h_mean_mean.iloc[i,:].values + df_norm_clustered_1h_std.iloc[i,:].values,                    
                        #df_norm_clustered_1h_mean_mean.iloc[i,:].values - df_norm_clustered_1h_std.iloc[i,:].values,
                        alpha=0.2,color=colors[i])


        # Plot the median to see similarity
        #ax.plot(diameters[1:-2]*10**9, df_norm_clustered_1h_mean_median.iloc[i,:].values, ':',color=colors[i])
        
        # Plot mean to see similarity
        ax.plot(diameters[1:-2]*10**9, df_norm_clustered_1h_mean_mean.iloc[i,:].values, ':',color=colors[i],linewidth = 4)
        #ax.set_xticks(bin_cols[::5])
        ax.set_xscale('log')
        ax.set_ylim(0,1.1)
        
        # Remove the top and right graph window
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_linewidth(3)


    plt.legend(frameon=False,bbox_to_anchor=(1, 0.75))    
    plt.xlabel('Dp [nm]')
    plt.title('1h mean clusters (median + 10-90 percentiles)')
    plt.ylabel('Normalised particle concentration \n [relative]')
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
    return fig, axs

def makeTrendPlotsMedian2(dfMedian, dfUpperQ, dfLowerQ, UQ, LQ, xL, yL):
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
    return fig, axs

def groupDFs(df_norm_clustered_1h_mean,cluster_ID):
    '''Compute the annual, monthly and month-yearly occurences and return dataframes.'''
    # All values are the same row wise (feature of using group-by count)
    
    # Create a temporary copy of cluster-assigned data (normalized)
    df_norm_clustered_1h_mean_copy = df_norm_clustered_1h_mean.copy()
    
    # Define NPF clusters
    clusterID = cluster_ID

    # Choosing only data assigne to cluster ID = cluster
    df_cluster_sliced = df_norm_clustered_1h_mean_copy[df_norm_clustered_1h_mean_copy['clusters'] == clusterID]
    df_cluster_mc     = df_cluster_sliced.copy(deep = True)
    df_cluster_my     = df_cluster_sliced.copy(deep = True)
    df_cluster_y      = df_cluster_sliced.copy(deep = True)
    
    # Fixing the dates so month-year can be extracted
    dt_array = df_cluster_my.index.values # Datetime array
    df_dummy = df_cluster_sliced.copy(deep = True)
    df_dummy['dtObjects'] = dt_array
    df_dummy['month_year'] = pd.to_datetime(df_dummy['dtObjects']).dt.to_period('M')
    df_dummy = df_dummy.drop('dtObjects', axis=1)
    
    
        
    # Create a cloumn called month and year in dataframe
    df_cluster_mc.loc[:,'month']      =  df_cluster_mc.index.month
    df_cluster_my['month_year']       = df_dummy['month_year'].values
    df_cluster_y.loc[:,'year']        =  df_cluster_y.index.year
    
    # Compute month count (irrespective of year)
    df_cluster_count_mc = df_cluster_mc.groupby('month').count()
    
    # Compute the month-year count
    df_cluster_count_my         = df_cluster_my.groupby('month_year').count()
    
    # Comoute the year total count
    df_cluster_count_y          = df_cluster_y.groupby('year').count()     

    df_cluster_count_mc_copy    = df_cluster_count_mc.copy()
    df_cluster_count_my_copy    = df_cluster_count_my.copy()
    df_cluster_count_y_copy     = df_cluster_count_y.copy()
    
        
    return df_cluster_count_mc_copy, df_cluster_count_my_copy, df_cluster_count_y_copy

def calcNPFOccurence(df_mc,df_my,df_y):
    '''Compute the monthly occurence from gropued dataframes and return list and datetimeobjects'''
    
    # We can thake the vlues from any column as input are grouped
    mcOcc    = df_mc.iloc[:,0].values
    mcOcc_dt = df_mc.index.values
    
    myOcc    = df_my.iloc[:,0].values
    myOcc_dt = df_my.index.values
    
    yOcc     = df_y.iloc[:,0].values
    yOcc_dt  = df_y.index.values  
       
    return mcOcc, mcOcc_dt, myOcc, myOcc_dt, yOcc, yOcc_dt
 
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
    
    
def plotNPFproxys(df_hourly_2010_2020_mean,df_norm_clustered_1h_mean,clusterIDs,bin_cols_,threshold,diameters,diameters_as_strings):
    '''Plot different NPF proxies'''
    
    df_tmp = df_hourly_2010_2020_mean.copy(deep = True)
    
    bin_cols = bin_cols_.copy()
    
    # Create Nx/Ntot for 1 h mean data----------------------------------------------------------
    bin_cols_LTnm = [x for x in bin_cols if x < threshold]

    df_tmp = calcNtot(diameters[:len(bin_cols_LTnm)+1], df_tmp,diameters_as_strings)
    df_tmp_nxntot = df_tmp.copy(deep = True)

    df_tmp_nxntot['NxNtot'] = df_tmp_nxntot['NtotCalc']/df_tmp_nxntot['Ntot']

    # Drop NaN:s
    df_tmp_nxntot = df_tmp_nxntot.dropna(subset =['NxNtot'])

    # Look at annual cycle for NxNtot
    df_tmp_nxntot_mean = df_tmp_nxntot['NxNtot'].groupby(df_tmp_nxntot.index.month).mean()
    df_tmp_nxntot_std = df_tmp_nxntot['NxNtot'].groupby(df_tmp_nxntot.index.month).std()

    df_tmp_nxntot_median = df_tmp_nxntot['NxNtot'].groupby(df_tmp_nxntot.index.month).median()
    df_tmp_nxntot_10q = df_tmp_nxntot['NxNtot'].groupby(df_tmp_nxntot.index.month).quantile(0.1)
    df_tmp_nxntot_90q = df_tmp_nxntot['NxNtot'].groupby(df_tmp_nxntot.index.month).quantile(0.9)
    
    # Create absolute diff Uf cpc - cpc----------------------------------------------------------
    
    
    df_tmp['abs_diff'] = np.absolute(df_tmp['UFCPC']-df_tmp['CPC3010'])

    df_tmp_adiff = df_tmp.copy(deep = True)

    # Drop NaN's
    df_tmp_adiff = df_tmp_adiff.dropna(subset =['abs_diff'])
        
    # we dont want of the measuring zero or less?
    # a bad mask!
    #maskcpc = (df_tmp['UFCPC'].values !=0 ) & (df_tmp['CPC3010'].values !=0 )
    
    #Better to drop rows like this
    df_tmp_adiff = df_tmp_adiff.drop(df_tmp_adiff[(df_tmp_adiff.UFCPC == 0) | (df_tmp_adiff.CPC3010 == 0)].index)
    

    df_1h_annual_cycle_adiff_mean  = df_tmp_adiff['abs_diff'].groupby(df_tmp_adiff.index.month).mean()
    df_1h_annual_cycle_adiff_std   = df_tmp_adiff['abs_diff'].groupby(df_tmp_adiff.index.month).std()

    df_1h_annual_cycle_adiff_median = df_tmp_adiff['abs_diff'].groupby(df_tmp_adiff.index.month).median()
    df_1h_annual_cycle_adiff_10q    = df_tmp_adiff['abs_diff'].groupby(df_tmp_adiff.index.month).quantile(0.1)
    df_1h_annual_cycle_adiff_90q    = df_tmp_adiff['abs_diff'].groupby(df_tmp_adiff.index.month).quantile(0.9)
    
    ## Create diff UFcpc and cpc--------------------------------------------------------------------------
    df_tmp['diff_cpcs'] = df_tmp['UFCPC'].values - df_tmp['CPC3010'].values  
    # Have to drop nans!!

    df_tmp_diff = df_tmp.copy(deep = True)
    df_tmp_diff = df_tmp_diff.dropna(subset =['diff_cpcs'])

    # Drop colmuns where mask is
    df_tmp_diff = df_tmp_diff.drop(df_tmp_diff[(df_tmp_diff.UFCPC == 0) | (df_tmp_diff.CPC3010 == 0)].index)

    df_1h_annual_cycle_diff_mean   = df_tmp_diff['diff_cpcs'].groupby(df_tmp_diff.index.month).mean()
    df_1h_annual_cycle_diff_std    = df_tmp_diff['diff_cpcs'].groupby(df_tmp_diff.index.month).std()

    df_1h_annual_cycle_diff_median = df_tmp_diff['diff_cpcs'].groupby(df_tmp_diff.index.month).median()
    df_1h_annual_cycle_diff_10q    = df_tmp_diff['diff_cpcs'].groupby(df_tmp_diff.index.month).quantile(0.1)
    df_1h_annual_cycle_diff_90q    = df_tmp_diff['diff_cpcs'].groupby(df_tmp_diff.index.month).quantile(0.9)

    #--------------------------------------------------------------------------------------------------
    
    # Plot
    fig = plt.figure(figsize=(8, 8))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    ax = plt.subplot(3, 1, 1) #----------------------------------------------------
    
    
    # Plot abs diff median
    ax.plot(df_1h_annual_cycle_adiff_median.index, 
               df_1h_annual_cycle_adiff_median.values,
               label = 'Median $|CPC_{UF}-CPC|$',color ='r')
    ax.fill_between(df_1h_annual_cycle_adiff_median.index,
                df_1h_annual_cycle_adiff_90q,
                df_1h_annual_cycle_adiff_10q,
                color ='r',alpha=0.1,label= str(10)+'-'+str(90)+' percentiles')

        
    # Plot abs diff mean
    ax.plot(df_1h_annual_cycle_adiff_mean.index, 
               df_1h_annual_cycle_adiff_mean.values,
               label = 'Mean $|CPC_{UF}-CPC|$',color ='b',linestyle = ':')
    
    # It is unreasonable for std to be less than zero (plotting artefact)
    neg_std = df_1h_annual_cycle_adiff_mean.values - df_1h_annual_cycle_adiff_std
    neg_std[neg_std < 0] = 0
    ax.fill_between(df_1h_annual_cycle_adiff_mean.index,
                df_1h_annual_cycle_adiff_mean.values + df_1h_annual_cycle_adiff_std.values,
                neg_std,
                color ='b',alpha=0.1,label = '+/-1$\sigma$') 
    plt.legend(frameon=False,bbox_to_anchor=(1.6, 1))
    plt.xticks(rotation = 45)
    # xticks color white
    plt.xticks(color='w')
    ax.set_ylabel('$|CPC_{UF}-CPC|$ \n [#/cm3]')

    # Remove the top and right graph window
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        ax.spines[spine].set_linewidth(3)    
    
    
    ax = plt.subplot(3, 1, 2) #--------------------------------------------------------------------
    ax.axhline(y=0, color='k', linestyle='-',alpha = 0.7)
    # Plot diff median
    ax.plot(df_1h_annual_cycle_diff_median.index, 
               df_1h_annual_cycle_diff_median.values,
               label = 'Median $CPC_{UF}-CPC$',color ='g')
    ax.fill_between(df_1h_annual_cycle_diff_median.index,
                df_1h_annual_cycle_diff_90q,
                df_1h_annual_cycle_diff_10q,
                color ='g',alpha=0.2,
                label= str(10)+'-'+str(90)+' percentiles')
        
    # Plot  diff mean
    
    ax.plot(df_1h_annual_cycle_diff_mean.index, 
               df_1h_annual_cycle_diff_mean.values,
               label = 'Mean $CPC_{UF}-CPC$',color ='m',linestyle = ':')
    
    ax.fill_between(df_1h_annual_cycle_diff_mean.index,
                df_1h_annual_cycle_diff_mean.values + df_1h_annual_cycle_adiff_std.values,
                df_1h_annual_cycle_diff_mean.values - df_1h_annual_cycle_adiff_std.values,
                color ='m',alpha=0.1,label = '+/-1$\sigma$',hatch = 'x')
    ax.set_ylabel('$CPC_{UF}-CPC$ \n [#/cm3]')
      
    
    plt.legend(frameon=False,bbox_to_anchor=(1.6, 1))
    plt.xticks(rotation = 45)
    # xticks color white
    plt.xticks(color='w')
    
    #------------
    
    # Plot Nx/Ntot
    axt = ax.twinx()
    axt.plot(df_tmp_nxntot_median.index, 
            df_tmp_nxntot_median.values,
            color ='k',linestyle = '--',
            label = '$N_x/N_{tot}$')
    axt.fill_between(df_tmp_nxntot_median.index,
                df_tmp_nxntot_90q,
                df_tmp_nxntot_10q,
                color ='k',alpha=0.1,label= str(10)+'-'+str(90)+' percentiles')
    axt.set_ylabel('$N_x/N_{tot}$ [a.u.]')

    # Remove the top and right graph window
    for spine in ['top']:
        axt.spines[spine].set_visible(False)
        axt.spines[spine].set_linewidth(3)        
    
    plt.xticks(rotation = 45)
    plt.xticks(np.arange(1, 13, 1), month_names)
    plt.legend(frameon=False,bbox_to_anchor=(1.56, 0.4))
    
    #----------------------------------------------------------------------------------------------
    # Next subplot for clusters
    ax2 = plt.subplot(3, 1, 3)
    
    # Define colormap
    n = len(clusterIDs)
    colors = cm.Set2(np.linspace(0,1,n))
    i = 0
    for clusterID in clusterIDs:
        
        # Get clusters annual cycle 
        # For one cluster:
        dfmc,dfmy,dfy = groupDFs(df_norm_clustered_1h_mean,clusterID)
        mcOcc, mcOcc_dt, myOcc, myOcc_dt, yOcc, yOcc_dt = calcNPFOccurence(dfmc,dfmy,dfy)
        
        if clusterID < 3:
            ax2.plot(mcOcc_dt,mcOcc,'-o',label='cluster: '+str(clusterID),
            color = colors[i],linewidth=3,markersize = 8)
        
        if clusterID > 2:
            ax2.plot(mcOcc_dt,mcOcc,'-x',label='cluster: '+str(clusterID),
            alpha = 0.5,color = colors[i])
        i = i+1
    ax2.set_ylabel('Occurence \n [hours]')
    
    # Remove the top and right graph window
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
        ax2.spines[spine].set_linewidth(3)    
    
    
    plt.legend(frameon=False,bbox_to_anchor=(1.45, 1))    
    plt.xticks(np.arange(1, 13, 1), month_names)
    plt.xticks(rotation = 45)
    plt.show()
    
    return


def makeDFforStackedPlot(df_norm_clustered_1h_mean, clusters):
    '''Make DF for the stacked plot'''

    df_norm_clustered_1h_mean_copy = df_norm_clustered_1h_mean.copy(deep = True)
    
    # Create empty df with colmumns like the name of the clusters
    df_clusters_month = pd.DataFrame(columns=clusters)
    

    for cluster in clusters:
        df_cluster = df_norm_clustered_1h_mean_copy[df_norm_clustered_1h_mean_copy['clusters'] == cluster]
        df_cluster = df_cluster.copy() 
        

        # For cluster x add column month and set values in column month equal to the month in the index
        df_cluster.loc[:,'month'] =  df_cluster.index.month

        
        # Calculate the occurence of cluster "cluster" per month
        df_cluster_count = df_cluster.groupby('month').count()

        
        #print('df_cluster_count')
        # We can take any column's values, rowwise values are all the same = total count per monts
        monthly_occurance = df_cluster_count.iloc[:,0].values
        
        #print(monthly_occurance)
        # Set values in column cluster to the montly counts
        df_clusters_month[cluster] = monthly_occurance
    
    # Compute the sum of all observation by calulating the rowwise sum and add a new column called "total_freq"
    df_clusters_month['total_freq'] = df_clusters_month.sum(axis=1)
    # print(df_clusters_month)
    
    # Normalizing the frequency by dividing the total number of clusters by 
    df_clusters_month_ = df_clusters_month.div(df_clusters_month['total_freq'], axis=0) 

    df_clusters_month  = df_clusters_month_.copy(deep=True)
    #print(df_clusters_month)
    
    return df_clusters_month, df_cluster

def makeStackedPlot(df_norm_clustered_1h_mean, clusters):
    '''Make stacked plot'''
    
    df_clusters_month, df_cluster = makeDFforStackedPlot(df_norm_clustered_1h_mean,clusters)
    
    #Fix so y-scale is in percentages
    df = df_clusters_month.copy(deep = True)
    # Multply all number objects by, happens to total freq as well, not good
    df[df.select_dtypes(include=['number']).columns] *= 100
    
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    
    df[clusters].plot(kind='bar', 
                    stacked=True, 
                    colormap='Set2',
                    width = 0.9,            
                    figsize=(6, 4))

    plt.ylabel("Frequency [%] \n (Normalized for data cover)")
    plt.ylim(0,100)
    plt.legend(title = 'Cluster:',frameon=False,bbox_to_anchor=(1, 1))
    plt.xticks(np.arange(0, 12, 1), month_names,)
    plt.xticks(rotation = 45)
    plt.show()
    return 
    
def prepareDFforTrendPlot(df_norm_clustered_1h_mean,clusters):
    '''Prepare datafram for trend analysis'''
    
    df_trend = df_norm_clustered_1h_mean.copy(deep=True)
    dt_array = df_trend.index.values
    df_trend['dtObjects'] = dt_array
    df_trend['month_year'] = pd.to_datetime(df_trend['dtObjects']).dt.to_period('M')
    df_trend['month'] = df_trend['dtObjects'].dt.month
    df_trend['year'] = df_trend['dtObjects'].dt.year

    # Create a dataframe from dictionaries as sometimes ther might not be any cluster 3 for example in some month --> 
    # Problem that when we group by month-year we don't get the 

    list_of_dicts = []
    for cluster in clusters:
        df_cluster = df_trend[df_trend['clusters'] == cluster]
        df_cluster = df_cluster.copy()    
        #print(cluster)   
        # Calculate the occurence of cluster "cluster" per month
        df_cluster_count  = df_cluster.groupby('month_year').count()    
        monthly_occurance = df_cluster_count.iloc[:,0].values


        dict_cluster = dict(zip(df_cluster_count.index, monthly_occurance))

        #print(dict_cluster)
        list_of_dicts.append(dict_cluster)
        #print(list_of_dicts)

    # ds is a dataframe whcih contains the rows = cluster no, and rows equal to month-year. 
    df_clusters_seqMonth = pd.DataFrame(list_of_dicts)
    
    str(df_clusters_seqMonth.columns.values[0])
    datetimes = [pd.to_datetime(str(x)) for x in list(df_clusters_seqMonth.columns.values)]
       
    # The days that have zero count get a Nan values that should be replaced by 0
    df_clusters_seqMonth = df_clusters_seqMonth.replace(np.nan, 0)
    
    # Add a row with total sum of columns  
    df_clusters_seqMonth.loc['total'] = df_clusters_seqMonth.sum(axis=0)
    
    # Transform the data frame so clusters are columns 
    df_T = df_clusters_seqMonth.T
    df_T.index = datetimes
    df_T['month'] = df_T.index.month
    
    # genealize the calling for columns
    cluster_columns_str_list = [str(x) for x in clusters]
    correct_columns_str = cluster_columns_str_list + ['total', 'month']
    
    # Number the colmns from 1-5 (clusterIDs) instead of 0-4 and name total and month
    df_T.columns = correct_columns_str

    df_clusters_seqMonth_copy = df_clusters_seqMonth.copy(deep=True)
    df_T_copy = df_T.copy(deep=True)
    
    return df_clusters_seqMonth_copy, df_T
    
def makeDFforTrend(df_clusters_seqMonth_T,period_list,clusters):
    '''Choose months that you take interest in and a 
    df which is normalized for the data coverage for plotting is returned'''
    
    df_T = df_clusters_seqMonth_T.copy(deep=True)
    df_T_period = df_T[df_T.month.isin(period_list)] # Choosing the months
    df_T_period = df_T_period.sort_index()
    
    # Make a list of strings for the cluster column names
    cluster_columns_str_list = [str(x) for x in clusters]
    
    # Normalizing the count for data coverage (numbers are not comparable if this is not done)
    df_norm_period_ = df_T_period[cluster_columns_str_list].div(df_T_period['total'], axis=0)
    
    df_norm_period = df_norm_period_.copy(deep = True)
    
    return df_norm_period
    
    
def plotScatter(df_norm_period, clusters):
    
    n = len(clusters)
    df_ = df_norm_period.copy(deep=True)
    df_.loc[:,'month'] =  df_.index.month
    colors = cm.Set2(np.linspace(0,1,n))
    
    # Plot annual scatter plots for all clusters colored by cluster 
    
    
    fig = plt.figure(figsize=(8, 8))
    i = 1
    j = 1
    for cluster, n_ax in zip(clusters, range(0,5)):
        i
        ax = plt.subplot(3,2,i)
        ax.set_title('Cluster ' + str(cluster))
        # Plot normally
#         ax.plot(df_norm_period.index,
#                 df_norm_period[str(cluster)].values,
#                 'o', color = colors[cluster-1],
#                 markersize = 4)
        #Plot with color by month
        im = ax.scatter(df_norm_period.index,
                df_norm_period[str(cluster)].values,
                s = 30,
                c = df_['month'].values,
                cmap='Greens')
                
        i = i + 1

        ax.set_ylim([-0.1,1.2])
        plt.xticks(rotation = 45)
          
    
    fig.add_subplot(111, frame_on=False)
    plt.ylabel("Normalized occurence")
    
    
    #Needed to not mess up labels
    plt.tick_params(labelcolor="none", bottom=False, left=False)

   
    fig.tight_layout()
    plt.show()
    

    return

def DFAnnualCount(df_norm_all,clusters):
    '''Makes a dataframe with the normalised yearly count for each cluster'''
    
    df_norm_all_test = df_norm_all.copy(deep=True)
    df_norm_all_test['year']  = df_norm_all_test.index.year
    #print(df_norm_all_test.head())
    
    years = df_norm_all_test['year'].unique() 
    
    # Make a df with sum normalized count per year and cluster
    year_list = []
    year_count_cluster = []  
    
    for year in years:  
        
        # Do it year by year --> slice df
        df_year_sliced = df_norm_all_test[df_norm_all_test['year'] == year]
        #print(df_year_sliced.head())
        
        df_tmp = df_year_sliced.copy(deep = True)

        
        # Delete column years
        del df_tmp['year']
        #print(df_tmp.head()) 
        
        # Compute the sum columnwise, you get a series, s
        # With the sum as and array with one item per cluster 
        s_sum = df_tmp.sum()
        
        year_list.append(year)
        year_count_cluster.append(s_sum.values)
        #print(year_count_cluster) 
    
    # Make dataframe from lists with columns as clusters and years as rows
    # year_count_cluster is a list of np arrays with length year. 
    # Each np.array contains five elements, one for each cluster 
    
    lists = year_count_cluster
    
    # Make lists into a dataframe
    df = pd.concat([pd.Series(x) for x in lists], axis=1)
    df = df.T
    
    # Rename columns to cluster IDs
    
    # Turn all cluster id:s to string items
    clusterIDs_str= [str(i) for i in clusters]
    
    #Rename columns
    df.columns = clusterIDs_str
       
    # Rename indicies to years
    years_str= [str(i) for i in years]
    df.index = years_str
        
    df_year = df.copy(deep = True)
    
    return df_year
    
    
def plotThielSen(df_norm_period, clusters,title):
    
    # Not suitable for full year -then use Mann-Kendall 
    
    df = df_norm_period.copy(deep=True)
    
            
    # Fix DF for Theil SEN
    df['year'] = df.index.year
    df['year'] = df['year'] - df['year'].iloc[0]
    df['month'] = df.index.month
    df['month_ordered'] = df['year']*12 + df['month'] 
    df['month_ordered'] = df['month_ordered'] - df['month_ordered'].iloc[0] + 1
    
    # Compute the Julian day
    df['julianDay'] = df.index.map(datetime.datetime.toordinal) + 1721424.5

    n = len(clusters)
    colors = cm.Set2(np.linspace(0,1,n))
    
    
    fig = plt.figure(figsize=(10, 10))
    
    i = 1

    # Plot Theil Sen for all clusters colored by cluster ID
    
    for cluster, n_ax in zip(clusters, range(0,len(clusters))):
                
        varx = df['month_ordered'].values
        vary = df[str(cluster)].values

        # For trend analysis we use theil zen slope
        slope, intercept, lo_slope, up_slope  = sc.stats.theilslopes(vary, varx, 0.95)

#         print('---------------------------')
#         print('Theil-Sen slope')
#         print('Slope:',slope)
#         print('Intercept:',intercept)
#         print('Lower bound of the confidence interval on medslopeLower:',lo_slope)
#         print('Upper bound of the confidence interval on medslopeLower:',up_slope)
#         print('---------------------------')

        ax = plt.subplot(3,2,i)
        ax.set_title('Cluster ' + str(cluster))
        # Plot datapoints as x:es
        ax.plot(varx, vary, 'x:', color = colors[cluster-1])
        

                
        # # Plot 
        
        # # Convert the trend unit from frequency/month to frequency/year in percetage
        # # by multiplying the slope with this convertion factor
        # Then you get change in normalized frequency per year

        
        cf = 12.0*100.0

        ax.plot(varx, intercept + slope * varx, '-', 
                 #label='y = '+str( round(cf *slope, 1) )+'x +' +str(round(intercept,1) ),
                 label='Slope: '+str( round(cf *slope, 1) )+'%/year' + \
                 ' \n y = '+str( round(cf *slope, 2) )+'x +' +str(round(intercept,2)) ,
                 color = colors[cluster-1])
        ax.fill_between(varx, 
               intercept + up_slope * varx,
               intercept + lo_slope * varx,
               alpha = 0.2,color = colors[cluster-1],
               label = '95% Confidence')
        
        ax.legend(frameon=False)
        #ax2.set_xticks([])
        ax.set_ylim([-0.1,1])
        #---------------------------------------
        
        #plt.xticks(rotation = 45)
        i = i+1

        # Remove the top and right graph window
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_linewidth(3)
    
    fig.add_subplot(111, frame_on=False)
    plt.ylabel("NPF events [a.u.] \n (hours normalized to data coverage)")
    plt.xlabel('Consequtive months')
        
    #Needed to not mess up labels
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.suptitle(title)
    fig.tight_layout()
    plt.show()
    
    return 
    
def sumCluster12(df_monthly_cluster_count,df_yearly_count_clusters):
    '''Make df:s with months as indicies and on column for the total count, yearly and monthly'''
    df_m = df_monthly_cluster_count.copy(deep = True)
    df_y = df_yearly_count_clusters.copy(deep = True)
    
    #-Montly----------------------
    #display(df_m)
    
    # Delete uneessary columns in dataframes 
    del df_m['3']
    del df_m['4']
    del df_m['5']
    
    del df_y['3']
    del df_y['4']
    del df_y['5']
    
    
    #display(df_m)     

    # Add new sum column
    df_m['12'] = df_m['1'].values + df_m['2'].values
    df_y['12'] = df_y['1'].values + df_y['2'].values
    
    #display(df_m)
    
    # Return 
    df_m_copy = df_m.copy(deep = True)
    df_y_copy = df_y.copy(deep = True)
    
    return df_m_copy, df_y_copy
    
def readSeaIcetoDF():
    '''Read the annual sea ice concentration to a data frame '''
    
    # Defining the to sea ice concentration data
    path = "C:\\Users\\Lovisa\\Documents\\GitHub\\Tjaernoe2022-group3\\notebooks\\data\\ave_sea_ice.csv"
    #path = 'C:\Users\Lovisa\Documents\GitHub\Tjaernoe2022-group3\notebooks\data\ave_sea_ice.csv'
    df = pd.read_csv(path)
    
    del df['Unnamed: 0']
    years = df['Year'].values
    years_str = [str(i) for i in years]
    df.index = years_str
    
    return df
    
def plotNPFvsSeaIce(sea_ice_annual,df_yearly_cluster_count_12):
    df_sea_ice = sea_ice_annual.copy(deep = True)
    df_NPF = df_yearly_cluster_count_12.copy(deep= True)
    
    # Extract data for matching years
    start_year_NPF = df_NPF.index[0]
    end_year_NPF   = df_NPF.index[-1]
    
    start_year_sea_ice = df_sea_ice.index[0]
    end_year_sea_ice   = df_sea_ice.index[-1]
    
    start, end = find_overlap(start_year_NPF,end_year_NPF,start_year_sea_ice,end_year_sea_ice)
    #print(start,end)
    
    years_ice = df_sea_ice.index.values
    years_NPF = df_NPF.index.values
       
    mask_ice = (years_ice >= start) & (years_ice <= end)
    mask_NPF = (years_NPF >= start) & (years_NPF <= end)
    
#     print(years_ice[mask_ice])
#     print(years_NPF[mask_NPF])

    # Plot all annual counts, cluster 1,2 and 12
    cluster_list = ['1', '2', '12']
    
    fig = plt.figure(figsize=(10, 10))
    
    ax = plt.subplot(1,1,1)
    
    ax.plot(df_sea_ice['Sea ice conc (%)'].values[mask_ice],
           df_NPF['12'].values[mask_NPF],
           'ko',label = 'Normalized NPF occurence \n (sum of cluster 1 and 2)')
    ax.set_xlabel('Sea ice concentration [%] \n (Barents Sea and Greenland Sea)')
    ax.set_ylabel("NPF events [a.u.] \n (hours/year normalized to data coverage)")
    
    varx = df_sea_ice['Sea ice conc (%)'].values[mask_ice]
    vary = df_NPF['12'].values[mask_NPF]
    
    # print(df_NPF[mask_NPF])
    # Add annotations
#     for xy in zip(varx, vary):                                       
#         ax.annotate('(%s, %s)' % xy, xy=xy, textcoords=labels_str) 
#    ax.annotate(text[i], (x[i], y[i] + 0.2))

    labels = years_ice[mask_ice]
    labels_str =[str(i) for i in labels] 
    # Add annotations
    for i in range(len(varx)):
        ax.annotate(labels_str[i], (varx[i], vary[i] + 0.04))
        
    # Fit regression line 
    res = sc.stats.linregress(varx, vary)
    #print(res.pvalue)

#     print(f"R-squared: {res.rvalue**2:.6f}")
#     print('Intercept:',res.intercept)
#     print('Slope:',res.slope)


    ax.plot(varx,
             res.intercept + res.slope*varx,
             'r-', 
            label='y = '+str( round(res.slope, 3) )+'x +' +str(round(res.intercept,3))
           + ', $R^2$ = ' + str( round(res.rvalue**2,3) )
           + '\n Pearson\'s $r$ = ' + str( round(res.rvalue,3) )         
           + '\n $p$ = ' +str( round(res.pvalue,2) ) )
    ax.legend(frameon = False, bbox_to_anchor=(1.8, 1))
    
    # Remove the top and right graph window
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        ax.spines[spine].set_linewidth(3)    
        
    
    plt.xlim(14.5,21.5)
    plt.ylim(0,0.6)
    plt.show()


    return
    
def find_overlap(start_year_NPF,end_year_NPF,start_year_sea_ice,end_year_sea_ice):
    
    if start_year_NPF >= start_year_sea_ice:
        start = start_year_NPF
    else:
        start = start_year_sea_ice
            
    if end_year_NPF >= end_year_sea_ice:
        end = end_year_sea_ice
    else:
        end = end_year_NPF
        
    return start, end
    
def DFAnnualCountNorm(df_norm_clustered_1h_mean,clusters):
        
    df = df_norm_clustered_1h_mean.copy(deep=True)
    df['year']  = df.index.year
    #print(df.head())
    
    years = df['year'].unique() 
    
    # Make a df with sum normalized count per year and cluster
   
    # Make a dataframe
    df_count_norm = pd.concat([pd.Series(x) for x in years], axis=1)
    df_count_norm = df_count_norm.T
    
    # First column is just named 0 for some reason, schange to year
    df_count_norm.rename(columns = {0:'year'}, inplace = True)
    #print(df_count_norm)
    
    for cluster in clusters: 
        year_list = []
        year_count_clusters = [] 
        all_total_year_datapoints = []
        all_years_count_norm = []
        for year in years: 
            # Do it year by year --> slice df
            df_year_sliced = df[df['year'] == year]
            
            #print(df_year_sliced.head())

            df_tmp = df_year_sliced.copy(deep = True)

            # Calulated total datapoints that year
            total_year_datapoints = len(df_tmp['clusters'].values)

            # Remove rows not equal to cluster number
            df_tmp = df_tmp.drop(df_tmp[(df_tmp.clusters != cluster)].index)

            #print(df_tmp.head())

            #Total number of datapoints in year for cluster = cluster_no
            year_count_cluster = len(df_tmp['clusters'].values)

            year_count_clusters.append(year_count_cluster)
            year_list.append(year)
            all_total_year_datapoints.append(total_year_datapoints)
            
            # division of lists
            # using zip() + list comprehension
            res = [i / j for i, j in zip(year_count_clusters, all_total_year_datapoints)]
            all_years_count_norm = np.array(res)
            #all_years_count_norm.append(arr)
       
        #print('length all_years_count_norm:',len(all_years_count_norm))
        # Add column for cluster with norm counts
        df_count_norm[str(cluster)] = all_years_count_norm
        
    # Rename indicies to years
    years_str= [str(i) for i in years]
    df_count_norm.index = years_str
    del df_count_norm['year']
    
        
    df_count_norm_copy = df_count_norm.copy(deep = True)
 
    return df_count_norm_copy
    
    

def makeSeasonalSizeDistPlot(df_norm_1h,diameters,df_norm_clustered_1h_mean_median,clusters,plot_all_months,plot_seasons):
    '''Aim is to compare season with clusters in a plot, set True for plot_all_months
    if you want that otherwise False. Set True for plot_seasons if you want that, otherwise false'''
    
    df = df_norm_1h.copy(deep = True)
    
    # Add column 'month' where the month as a number 1-12 is indicated
    df['month'] = df.index.month
    #print(df)
    
    if plot_all_months == False and plot_seasons == True:
    
        # Define the seasons as lists
        DJF = [12,1,2]
        MAM = [3,4,5]
        JJA = [6,7,8]
        SON = [9,10,11]

        # Make a list of all the seasons to loop over
        seasons = [DJF,MAM,JJA,SON]
        seasons_str = ['DJF','MAM','JJA','SON']
        n1 = len(seasons)
        colors = cm.tab20(np.linspace(0,1,n1))
        
    if plot_all_months == True and plot_seasons == False:    
    
        # Look at the whole year
        seasons = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]]
        seasons_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
        n2 = len(seasons)
        colors = cm.tab20(np.linspace(0,1,n2))

    fig, ax = plt.subplots(figsize=(10,4))
    
    
    #Loop over the seasons
    i = 0
    for season in seasons:
        
        season_str = seasons_str[i] 
        
        #Pick out the months
        df_season = df[df.month.isin(season)].copy(deep=True) 
        # as we don't use sort_index months are scrambeled but it should not matter?
        
        # Add column indicating the season, 0 = DJF, 1 = MAM, JJA = 2, SON = 3
        df_season.loc[:,'season'] = i
        # print(df_season['season'].unique())
        
        # Check to see that right months are picked
        #print(df_season['month'].unique())
        
        # Drop unessary columns 
        if 'clusters' in df_season.columns:
            del df_season['clusters']
        del df_season['month']
        
        # Calculate median and 10,90 percentiles for each month 
        df_season_median = df_season.groupby('season').median()
        df_season_10q    = df_season.groupby('season').quantile(.1)
        df_season_90q    = df_season.groupby('season').quantile(.9)
        
        #print(len(df_season_median.iloc[0,:].values))
        #print(len(diameters))
        
        # Plot the median + quartiles
        ax.plot(diameters*10**9, df_season_median.iloc[0,:].values,
                '-', label=season_str,
                color=colors[i],
                linewidth = 3)
        ax.fill_between(diameters*10**9, 
                        #Plot percentiles
                        df_season_90q.iloc[0,:].values,
                        df_season_10q.iloc[0,:].values,
                        #For plotting std+mean
                        #df_norm_clustered_1h_mean_mean.iloc[i,:].values + df_norm_clustered_1h_std.iloc[i,:].values,                    
                        #df_norm_clustered_1h_mean_mean.iloc[i,:].values - df_norm_clustered_1h_std.iloc[i,:].values,
                        alpha=0.5,color=colors[i])
        ax.set_xscale('log')
        ax.set_ylim(0,1.1)
        
        # Remove the top and right graph window
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_linewidth(3)
        
        # Change i for next loop-round
        i = i+1
        
    plt.legend(frameon=False,loc = 'center left',bbox_to_anchor=(1, 0.5))      
    plt.xlabel('Dp [nm]')
    plt.title('Median + 10-90 percentiles')
    plt.ylabel('Normalised particle concentration \n [relative]')
    plt.show()
    
    
    # Make another figure with only cluster and no percentiles
    
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Plot cluster medians--------------------------------------------
    # Define colormap
    n = len(clusters)
    colors_cluster = cm.Set2(np.linspace(0,1,n)) 
    
    # Remove the top and right graph window
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        ax.spines[spine].set_linewidth(3)
        
    # Plot the clusters and shaded spread in the data
    ii = 0
    for i in range(len(clusters)):
        cluster = clusters[i]
        ax.plot(diameters*10**9, df_norm_clustered_1h_mean_median.iloc[i,:].values,
                ':',color=colors_cluster[ii],linewidth = 4,
                label='cluster: '+str(cluster))
        ii = ii+1
    
    #-----------------------------------------------------------------
    
    #Loop over the seasons
    i = 0
    for season in seasons:
        
        season_str = seasons_str[i] 
        
        #Pick out the months
        df_season = df[df.month.isin(season)].copy(deep=True) 
        # as we don't use sort_index months are scrambeled but it should not matter?
        
        # Add column indicating the season, 0 = DJF, 1 = MAM, JJA = 2, SON = 3
        df_season.loc[:,'season'] = i
        # print(df_season['season'].unique())
        
        # Check to see that right months are picked
        #print(df_season['month'].unique())
        
        # Drop unessary columns 
        if 'clusters' in df_season.columns:
            del df_season['clusters']
        del df_season['month']
        
        # Calculate median and 10,90 percentiles for each month 
        df_season_median = df_season.groupby('season').median()
        df_season_10q    = df_season.groupby('season').quantile(.1)
        df_season_90q    = df_season.groupby('season').quantile(.9)
        
        #print(len(df_season_median.iloc[0,:].values))
        #print(len(diameters))
        
        # Plot the median + quartiles
        ax.plot(diameters*10**9, df_season_median.iloc[0,:].values,
                '-', label=season_str,
                color=colors[i],
                linewidth = 3)
        # ax.fill_between(diameters*10**9, 
                        # #Plot percentiles
                        # df_season_90q.iloc[0,:].values,
                        # df_season_10q.iloc[0,:].values,
                        # #For plotting std+mean
                        # #df_norm_clustered_1h_mean_mean.iloc[i,:].values + df_norm_clustered_1h_std.iloc[i,:].values,                    
                        # #df_norm_clustered_1h_mean_mean.iloc[i,:].values - df_norm_clustered_1h_std.iloc[i,:].values,
                        # alpha=0.5,color=colors[i])
        ax.set_xscale('log')
        ax.set_ylim(0,1.1)
        

        
        # Change i for next loop-round
        i = i+1
    # Remove the top and right graph window
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        ax.spines[spine].set_linewidth(3)
    
    plt.legend(frameon=False,loc = 'center left',bbox_to_anchor=(1, 0.5))    
    plt.xlabel('Dp [nm]')
    plt.title('Median')
    plt.ylabel('Normalised particle concentration \n [relative]')
    plt.show()
        
        
    return 