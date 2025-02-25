#In this python program, the flare catalog(with cme) is used as the label source.
#To create the label, log scale flare intensity is used
import argparse
import os.path, os
import pandas as pd
# pd.options.mode.chained_assignment = None 

#In this function, to create the label, the maximum intensity of flare between midnight to midnight
#and noon to noon with respective date is used.
def hourly_obs(df_fl: pd.DataFrame, img_dir, stop, class_type = 'bin'):

    # Datetime 
    df_fl['start_time'] = pd.to_datetime(df_fl['start_time'], format = '%Y-%m-%d %H:%M:%S')


    #List to store intermediate results
    lis = []
    cols = ['Timestamp', 'GOES_cls', 'Label']

    #Loop to check max from midnight to midnight and noon to noon
    for i in range(len(df_rec)):

        #Date with max intensity of flare with in the 24 hour window
        window_start = df_rec.iloc[i, 0] # timestamp
        window_end = window_start + pd.Timedelta(hours = 23, minutes = 59, seconds = 59)

        if window_start > stop:
            break
        
        emp = df_fl[ (df_fl.start > window_start) & (df_fl.start <= window_end) ].sort_values('fl_goescls', ascending = False).head(1).squeeze(axis = 0)
        if pd.Series(emp.fl_goescls).empty:
            ins = ''
            target = 0
        else: 
            ins = emp.fl_goescls
            
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
            

        lis.append([window_start, ins, target])
        
    df_out = pd.DataFrame(lis, columns = cols)

    # df_out['Timestamp'] = pd.to_datetime(df_out['Timestamp'], format='%Y-%m-%d %H:%M:%S')
    df_out['Timestamp'] = df_out['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df_out


#Creating time-segmented 4 tri-monthly partitions
def create_partitions(df, savepath = '/', class_type = 'bin'):
    search_list = [['01', '02', '03'], ['04', '05', '06'], ['07', '08', '09'], ['10', '11', '12']]
    for i in range(4):
        search_for = search_list[i]
        mask = df['Timestamp'].apply(lambda row: row[5:7]).str.contains('|'.join(search_for))
        partition = df[mask]
        print(partition['GOES_cls'].value_counts())
        
        # Make directory 
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
            print('Created directory:', savepath)
            
        # Dumping the dataframe into CSV with label as Date and goes_class as intensity
        partition.to_csv(savepath + f'24image_{class_type}_GOES_classification_Partition{i + 1}.csv',\
                         index = False, header = True, columns = ['Timestamp', 'GOES_cls', 'Label'])


if __name__ == "__main__":

    #Load Original source for Goes Flare X-ray Flux 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/workspace/", help="Path to data folder")
    parser.add_argument("--project_path", type=str, default="/workspace/", help="Path to project folder")
    args = parser.parse_args()
    
    
    df_fl = pd.read_csv(args.data_path + 'MultiwayIntegration_2010_to_2018_conf_rxfi.csv', usecols = ['start_time', 'goes_class'])
    savepath = os.getcwd()
    
    stop = pd.to_datetime('2024-07-31 23:59:59', format = '%Y-%m-%d %H:%M:%S')

    #Calling functions in order
    df_res = hourly_obs(df_fl = df_fl, stop = stop, class_type = 'bin')
    create_partitions(df_res, savepath = savepath, class_type = 'bin')