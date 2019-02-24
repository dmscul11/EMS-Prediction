
import pandas as pd
import numpy as np
import glob
import os


# all event files
project = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/'
files = glob.glob(project + 'event-files/E*Events.csv')

# cycle through experiemtns/participant event files
for f in files:

    # read in events
    events = pd.read_csv(f, header=0, sep=',', index_col=False)
    file = f.split('/')
    experiment = file[-1].split('_')[0]
    participant = file[-1].split('_')[1]
    print('\nExperiment ' + experiment)
    print('Participant ' + participant)

    # cycle through data types
    data_types = ['AppleWatch_Right', 'AppleWatch_Left', 'Myo_EMG', 'Myo_IMU', 'PatientSpace', 'RawXY']
    # data_types = ['C2_PatientSpace', 'C2_RawXY']
    for dt in data_types:
        data_files = glob.glob(project + 'raw-data/' + experiment + '_' + participant + '_' + dt + '*.csv')

        # cycle through each data type file read in data
        for df in data_files:
            print('\n' + df)

            # read in data
            data = pd.read_csv(df, header=0, sep=',', index_col=False)
            if (dt in ['AppleWatch_Right', 'AppleWatch_Left']) and (data.columns[0] != 'id'):
                data = pd.read_csv(df, header=0, sep=',', index_col=False, names=['id', 'session_id', \
                    'start_time', 'end_time', 'number_of_pts', 'frequency', 'vital', 'session_notes', \
                    'device_id', 'device_make_model', 'time', 'xacceleration', 'yacceleration', \
                    'zacceleration', 'xrotation', 'yrotation', 'zrotation', 'yaw pitch', 'roll', 'session_id'])

            # cycle through events
            for index, row in events.iterrows():
                procedure = row['Procedure']
                trial_number = row['Trial_Number']
                start_time = row['Cleaned_Start_Time']
                end_time = row['Cleaned_Stop_Time']

                # get info based on data type
                if dt in ['AppleWatch_Right', 'AppleWatch_Left']:
                    device = 'AppleWatch'
                    # side = str(dt.split('_')[1])
                    side = df.split('/')[-1].split('_')[-1].split('.')[0]
                    dtype = 'Acceleration'
                if dt in ['Myo_EMG', 'Myo_IMU']:
                    data = data.rename(columns={' Timestamp': 'time'})
                    data = data.rename(columns={'Timestamp': 'time'})
                    device = 'Myo'
                    dtype = str(dt.split('_')[1])
                if dt in ['C2_PatientSpace', 'C2_RawXY']:
                    data = data.rename(columns={'Timestamp': 'time'})
                    device = 'Camera'
                    side = int(dt.split('_')[0][1])
                    dtype = str(dt.split('_')[1])

                # grab data just from this event
                event_df = data.loc[(data['time'] >= start_time) & (data['time'] <= end_time)]
                if event_df.shape[0] > 1:
                    event_df.sort_values(by=['time'], inplace=True, ascending=True)
                    event_df.reset_index(inplace=True, drop=True)

                    # remove unnecessary columns based on data type and rename all to match
                    if dt in ['AppleWatch_Right', 'AppleWatch_Left']:
                        try:
                            event_df = event_df.drop(columns=['id', 'session_id', 'start_time', 'end_time', 'number_of_pts', \
                                'frequency', 'vital', 'session_notes', 'device_id', 'device_make_model', 'session_id.1'])
                        except:
                            event_df = event_df.drop(columns=['id', 'session_id'])
                    if dt == 'Myo_EMG':
                        try:
                            try:
                                try:
                                    event_df = event_df.drop(columns=['Unnamed: 0', 'Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                        ' Orientation_W', ' Orientation_X', ' Orientation_Y', ' Orientation_Z', ' Acc_X', ' Acc_Y', \
                                        ' Acc_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' RSSI', ' Roll', ' Pitch', ' Yaw '])
                                    event_df = event_df.rename(columns={' Arm': 'Arm', ' EMG_1': 'EMG_1', ' EMG_2': 'EMG_2', \
                                        ' EMG_3': 'EMG_3', ' EMG_4': 'EMG_4', ' EMG_5': 'EMG_5', ' EMG_6': 'EMG_6', \
                                        ' EMG_7': 'EMG_7', ' EMG_8': 'EMG_8'})
                                except:
                                    event_df = event_df.drop(columns=['Unnamed: 0', 'Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                        ' Orientation_W', ' Orientation_X', ' Orientation_Y', ' Orientation_Z', ' Acc_X', ' Acc_Y', \
                                        ' Acc_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' RSSI', ' Roll', ' Pitch', ' Yaw '])
                                    event_df = event_df.rename(columns={' Arm': 'Arm'})
                            except:
                                event_df = event_df.drop(columns=['DeviceID', 'Warm', 'Sync', 'Locked', 'Pose', \
                                'Orientation_W', 'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Acc_X', 'Acc_Y', \
                                'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'RSSI', 'Roll', 'Pitch', 'Yaw'])
                        except:
                            event_df = event_df.drop(columns=['Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                ' Orientation_W', ' Orientation_X', ' Orientation_Y', ' Orientation_Z', ' Acc_X', ' Acc_Y', \
                                ' Acc_Z', ' Gyro_X', ' Gyro_Y', ' Gyro_Z', ' RSSI', ' Roll', ' Pitch', ' Yaw '])
                            event_df = event_df.rename(columns={' Arm': 'Arm', ' EMG_1': 'EMG_1', ' EMG_2': 'EMG_2', \
                                ' EMG_3': 'EMG_3', ' EMG_4': 'EMG_4', ' EMG_5': 'EMG_5', ' EMG_6': 'EMG_6', \
                                ' EMG_7': 'EMG_7', ' EMG_8': 'EMG_8'})
                    if dt == 'Myo_IMU':
                        try:
                            try:
                                try:
                                    event_df = event_df.drop(columns=['Unnamed: 0', 'Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                        'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_7', 'EMG_8'])
                                    event_df = event_df.rename(columns={' Arm': 'Arm', ' Orientation_W': 'Orientation_W', \
                                        ' Orientation_X': 'Orientation_X', ' Orientation_Y': 'Orientation_Y', ' Orientation_Z': 'Orientation_Z', \
                                        ' Acc_X': 'Acc_X', ' Acc_Y': 'Acc_Y', ' Acc_Z': 'Acc_Z', ' Gyro_X': 'Gyro_X', ' Gyro_Y': 'Gyro_Y', \
                                        ' Gyro_Z': 'Gyro_Z', ' RSSI': 'RSSI', ' Roll': 'Roll', ' Pitch': 'Pitch', ' Yaw ': 'Yaw'})
                                except:
                                    event_df = event_df.drop(columns=['DeviceID', 'Warm', 'Sync', 'Locked', 'Pose', \
                                        'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_7', 'EMG_8'])
                            except:
                                event_df = event_df.drop(columns=['Unnamed: 0', 'Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                    ' EMG_1', ' EMG_2', ' EMG_3', ' EMG_4', ' EMG_5', ' EMG_6', ' EMG_7', ' EMG_8'])
                                event_df = event_df.rename(columns={' Arm': 'Arm', ' Orientation_W': 'Orientation_W', \
                                    ' Orientation_X': 'Orientation_X', ' Orientation_Y': 'Orientation_Y', ' Orientation_Z': 'Orientation_Z', \
                                    ' Acc_X': 'Acc_X', ' Acc_Y': 'Acc_Y', ' Acc_Z': 'Acc_Z', ' Gyro_X': 'Gyro_X', ' Gyro_Y': 'Gyro_Y', \
                                    ' Gyro_Z': 'Gyro_Z', ' RSSI': 'RSSI', ' Roll': 'Roll', ' Pitch': 'Pitch', ' Yaw ': 'Yaw'})
                        except:
                            event_df = event_df.drop(columns=['Device ID', ' Warm?', ' Sync', 'Locked', ' Pose', \
                                ' EMG_1', ' EMG_2', ' EMG_3', ' EMG_4', ' EMG_5', ' EMG_6', ' EMG_7', ' EMG_8'])
                            event_df = event_df.rename(columns={' Arm': 'Arm', ' Orientation_W': 'Orientation_W', \
                                ' Orientation_X': 'Orientation_X', ' Orientation_Y': 'Orientation_Y', ' Orientation_Z': 'Orientation_Z', \
                                ' Acc_X': 'Acc_X', ' Acc_Y': 'Acc_Y', ' Acc_Z': 'Acc_Z', ' Gyro_X': 'Gyro_X', ' Gyro_Y': 'Gyro_Y', \
                                ' Gyro_Z': 'Gyro_Z', ' RSSI': 'RSSI', ' Roll': 'Roll', ' Pitch': 'Pitch', ' Yaw ': 'Yaw'})
                    if dt in ['C2_PatientSpace', 'C2_RawXY']:
                        event_df = event_df.drop(columns=['Frame'])



                    # add in time difference column
                    event_df.insert(0, 'seconds', '')
                    event_df['time'] = event_df['time'].str[:-3]
                    try:
                        event_df['time'] = pd.to_datetime(event_df['time'], format="%Y-%m-%d %H:%M:%S.%f")
                    except:
                        event_df['time'] = pd.to_datetime(event_df['time'], format="%Y-%m-%d %H:%M:%S %f")
                    event_df['seconds'] = (event_df['time'] - event_df['time'][0]) / np.timedelta64(1, 's')

                    # add in additional information columns
                    event_df.insert(0, 'Data Type', dtype)
                    if dt in ['Myo_EMG', 'Myo_IMU']:
                        event_df.insert(0, 'Side', '')
                    else:
                        event_df.insert(0, 'Side', side)
                    event_df.insert(0, 'Device', device)
                    event_df.insert(0, 'Trial', trial_number)
                    event_df.insert(0, 'Participant', int(participant[1]))
                    event_df.insert(0, 'Experiment', int(experiment[1]))
                    event_df.insert(0, 'Procedure', procedure)

                    # make output folders if not exists
                    directory = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/processed-data/' + procedure
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # split myo data left and right
                    if dt in ['Myo_EMG', 'Myo_IMU']:
                        left_df = event_df.loc[(event_df['Arm'] == 'left')]
                        left_df['Side'] = 'Left'
                        right_df = event_df.loc[(event_df['Arm'] == 'right')]
                        right_df['Side'] = 'Right'
                        left_df = left_df.drop(columns=['Arm'])
                        right_df = right_df.drop(columns=['Arm'])

                        # write data to csv files
                        file_name = directory + '/' + procedure.replace(' ', '') + '_' + experiment + '_' + participant + '_T' + \
                            str(trial_number) + '_' + device + '_Left_' + dtype + '.csv'
                        left_df.to_csv(path_or_buf=file_name, sep=',', header=True)
                        file_name = directory + '/' + procedure.replace(' ', '') + '_' + experiment + '_' + participant + '_T' + \
                            str(trial_number) + '_' + device + '_Right_' + dtype + '.csv'
                        right_df.to_csv(path_or_buf=file_name, sep=',', header=True)

                    # otherwise just write to file
                    else:
                        # write data to csv files
                        file_name = directory + '/' + procedure.replace(' ', '') + '_' + experiment + '_' + participant + '_T' + \
                            str(trial_number) + '_' + device + '_' + str(side) + '_' + dtype + '.csv'
                        event_df.to_csv(path_or_buf=file_name, sep=',', header=True)
