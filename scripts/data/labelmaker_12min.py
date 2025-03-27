#In this python program, the flare catalog(with cme) is used as the label source.
#To create the label, log scale flare intensity is used
import glob
import argparse
import os.path, os
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None 

#In this function, to create the label, the maximum intensity of flare between midnight to midnight
#and noon to noon with respective date is used.
def sub_class_num(df: pd.DataFrame):
    # Extract the numeric part after the class (C/M/X) and convert to float
    numeric_part = df['goes_class'].str[1:].astype(float)

    # Use np.select for multiple conditions to avoid repeated operations
    conditions = [
        df['goes_class'].str.startswith('C'),
        df['goes_class'].str.startswith('M'),
        df['goes_class'].str.startswith('X')
    ]

    # Corresponding choices based on class
    choices = [
        numeric_part,            # C-class: same value
        10 * numeric_part,       # M-class: multiply by 10
        100 * numeric_part       # X-class: multiply by 100
    ]

    # Assign to sub_cls using np.select
    df['sub_cls'] = np.select(conditions, choices, default=None)

def hourly_obs(df_fl: pd.DataFrame, img_dir, start, stop, class_type = 'bin'):

    # Datetime 
    df_fl['start_time'] = pd.to_datetime(df_fl['start_time'], format = '%Y-%m-%d %H:%M:%S')

    # Create sub-class column
    sub_class_num(df_fl)

    #List to store intermediate results
    lis = []
    for year in range(start, stop + 1):
        for month in range(1, 13):
            print(f"{year}:{month:02d} processing ...")
            for day in range(1, 32):
                dir = img_dir + f'{year}/{month:02d}/{day:02d}/*.jpg'
                files = sorted(glob.glob(dir))

                for file in files:
                    # print(file)
                    window_start = pd.to_datetime(file.split('HMI.m')[1][:-4], format="%Y.%m.%d_%H.%M.%S")
                    window_end = window_start + pd.Timedelta(hours = 23, minutes = 59, seconds = 59)
                    window = df_fl[ (df_fl.start_time > window_start) & (df_fl.start_time <= window_end) ]

                    emp = window.sort_values('goes_class', ascending = False).head(1).squeeze(axis = 0)
                    cumulative_index = window['sub_cls'].sum()

                    # 1) define binary index from max flare class
                    if pd.Series(emp.goes_class).empty:
                        ins = 'FQ'
                        target = 0
                    else: 
                        ins = emp.goes_class
                        
                        if class_type == 'bin':
                            if ins >= "M1.0": # FQ and A class flares
                                target = 1
                            else:
                                target = 0
                        elif class_type == 'multi':

                            if ins >= "M1.0": # FQ and A class flares
                                target = 3
                            elif ins >= "C1.0":
                                target = 2
                            elif ins >= "B1.0":
                                target = 1
                            else:
                                target = 0

                    # 2) define binary index from cumulative flare class
                    if cumulative_index >= 10:
                        target_cumulative = 1
                    else:
                        target_cumulative = 0

                    lis.append([window_start, f"hmi/{year}/{month:02d}/{day:02d}/" + file.split('/')[-1], ins, cumulative_index, target, target_cumulative])

    cols = ['timestep', 'path', 'goes_class', 'cumulative_index', 'label_max', 'label_cum']
    df_out = pd.DataFrame(lis, columns = cols)

    # df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_out['timestep'] = df_out['timestep'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df_out


#Creating time-segmented 4 tri-monthly partitions
def split_dataset(df, savepath = '/', class_type = 'bin'):
    search_list = [['2011', '2012', '2013'], ['2014']]
    for i in range(2):
        search_for = search_list[i]
        mask = df['timestep'].apply(lambda row: row[0:4]).str.contains('|'.join(search_for))
        partition = df[mask]
        print(partition['label'].value_counts())
        
            
        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        flag = 'train' if i == 0 else 'test'
        partition.to_csv(savepath + f'/{class_type}_classification_{flag}_12min.csv',
                        index = False, 
                        header = True, 
                        columns = ['timestep', 'path', 'goes_class', 'label'])

if __name__ == "__main__":

    #Load Original source for Goes Flare X-ray Flux 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/media/jh/maxone/Research/GSU/Research1_xray_flux/", help="Path to data folder")
    parser.add_argument("--project_path", type=str, default="/home/jh/2python_pr/baseline_fulldisk/", help="Path to project folder")
    parser.add_argument("--start", type=int, default='2011', help="start time of the dataset")
    parser.add_argument("--end", type=int, default='2014', help="end time of the dataset")
    args = parser.parse_args()
    
    df_fl = pd.read_csv(args.data_path + 'sdo_era_goes_flares_integrated_all_CME_r1.csv', usecols = ['start_time', 'goes_class'])
    savepath = os.getcwd()

    #Calling functions in order
    df_res = hourly_obs(
        df_fl = df_fl, 
        img_dir = args.data_path + 'hmi_jpgs_512/', 
        start = args.start, 
        stop = args.end, 
        class_type = 'bin'
        )
    split_dataset(df_res, savepath = savepath, class_type = 'bin')