import glob
import pandas as pd
import numpy as np
import datetime 
import matplotlib.pyplot as plt

def save_df(df, path, name='',  float_format='%.3f'):
    print("Save as: "+str(path+'\\'+name+'.dat'))
    df.to_csv(path+'\\'+name+'.dat', index=True, float_format=float_format)

def find_where_data_starts(infile, startwith):   
    with open(infile) as f:
        lines = f.readlines()
        for value, line in enumerate(lines):
            if line.startswith(startwith):
                value
                break
    return value
	
def find_line_with(infile, string_in_line):   
    with open(infile) as f:
        lines = f.readlines()
        for value, line in enumerate(lines):
            if line.startswith(string_in_line):
                line
                break
    return line
	
def find_line_number(infile, line_number):   
    with open(infile) as f:
        lines = f.readlines()
        for value, line in enumerate(lines):
            if value == line_number:
                line
                break
    return line
	
def load_data_to_dict_res(path, name_includes, startwith):

    list_files = glob.glob(path+'*'+name_includes+'*.nas') 
    appended_data_daily = []
    appended_data_2hourly = []
    appended_data_1hourly = []

    dict_res_to_df={}
    for file in list_files:
        print("file: "+str(file))
        digits_in_file =  ''.join(filter(str.isdigit, file))
        print(digits_in_file)
        start_year = digits_in_file[4:8]
        print(start_year)

        start_date_from_file = find_line_with(infile=file, string_in_line='Startdate')
        start_date_from_file = start_date_from_file.replace('Startdate:','')
        start_date_from_file = start_date_from_file.strip()
        print("start date from file: "+str(start_date_from_file))
        start_date_from_file_to_datetime = pd.to_datetime(start_date_from_file)
        print(start_date_from_file_to_datetime)

        resolution = find_line_with(infile=file, string_in_line='Resolution code')
        resolution = resolution.replace('Resolution code:','')
        resolution = resolution.strip()
        print("Resolution code: "+str(resolution))

        originator = find_line_number(file, line_number=1)
        print(originator)

        start_idx = find_where_data_starts(infile=file, startwith=startwith) #find the index to cut the dataframe

        df = pd.read_csv(file, sep='\s+', skiprows=start_idx, index_col=False)   

        if resolution == '1d':        
            df.starttime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.starttime, unit='D')) #convert float of year
            df.endtime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.endtime, unit='D'))
            appended_data_daily.append(df)

        if resolution == '2h':        
            df.starttime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.starttime, unit='D')) #convert float of year
            df.endtime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.endtime, unit='D'))
            appended_data_2hourly.append(df)

        if resolution == '1h':        
            df.starttime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.starttime, unit='D')) #convert float of year
            df.endtime = pd.to_datetime(datetime.datetime.strptime(str(start_date_from_file_to_datetime), '%Y-%m-%d %H:%M:%S') + pd.to_timedelta(df.endtime, unit='D'))
            appended_data_1hourly.append(df)

    if len(appended_data_daily) > 0:
        df_appened_daily=pd.concat(appended_data_daily)
        dict_res_to_df['daily'] = df_appened_daily
    if len(appended_data_2hourly) > 0:
        df_appened_2hourly=pd.concat(appended_data_2hourly)
        dict_res_to_df['2hours'] = df_appened_2hourly
    if len(appended_data_1hourly) > 0:  
        df_appened_1hourly=pd.concat(appended_data_1hourly)
        dict_res_to_df['1hour'] = df_appened_1hourly
    
    if len(dict_res_to_df) == 1:
        print("only one resolution so returning a df")
        res = list(dict_res_to_df.keys())[0]
        print(list(dict_res_to_df.keys())[0])
        df_res = dict_res_to_df[res]
        return df_res
    return dict_res_to_df
	
def select_cols_df(df, var):
    df = df.dropna(how='all', axis=0)
    cols = df.columns
    var = 'rH'
    var_cols = [x for x in cols if var in x]
    print(var_cols)
    selected_cols = ['starttime', 'endtime'] + var_cols
    df = df[selected_cols].copy()
    df = df.set_index('starttime')
    df.index = pd.to_datetime(df.index)
    return df
	
def clean_df_flags(df, var):
    cols = df.columns
    flag_cols = [x for x in cols if 'flag' in x]
    print(flag_cols)
    for flag in flag_cols:
        print(df[flag].unique())
        print(df[flag].value_counts(normalize=True))
        df_flags_removed = df[df[flag].isin([0, np.nan])]
        df_flags_removed = df_flags_removed[df_flags_removed[var] < 9999]
        df_flags_removed = df_flags_removed.sort_values('starttime')
    return df_flags_removed
    
 