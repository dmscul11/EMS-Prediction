
import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# count occaurances of events
def count_events(project, trials_threshold, use_data, use_exp, use_par):

    # get all event dirs
    dirs = glob.glob(project + 'processed-data/*')
    events = {}

    # count number of files in each event dir
    for d in dirs:

        # get all files in dir and event name
        files = glob.glob(d + '/*.csv')
        event = d.split('/')[-1]

        # only include files in count of the data type, exp, and par
        file_list = []
        for f in files:
            file = f.split('/')[-1]
            tmp = file.split('_')
            exp = tmp[1]
            par = tmp[2]

            if (exp in use_exp) and (par in use_par) and (use_data[0] in file):
                file_list.append(f)

        # only include events if they are above threshold
        if len(file_list) >= trials_threshold:
            events[event] = len(file_list)

    print(events)
    return events


# read in data into arrays
def read_in(project, events, use_data, use_exp, use_par, max_rows):

    # read in data by event folder
    count = 0
    data_all = []
    events_all = []
    all_rows = []
    max_tps = []
    for e in events:

        # get each experiment included
        for exp in use_exp:

            # get each participant included
            for par in use_par:
                files = glob.glob(project + 'processed-data/' + e + '/*' + exp + '_' + par + '*.csv')

                # if data exists
                if files != []:
                    # get unique instances
                    instances = []
                    for f in files:
                        instance = f.split('/')[-1].split('_')[3]
                        if instance not in instances:
                            instances.append(instance)

                    # for each instance of event
                    for i in instances:
                        count = count + 1

                        # for each data type in included get files
                        data_list = []
                        for d in use_data:
                            files = glob.glob(project + 'processed-data/' + e + '/*' + exp + '_' + \
                                par + '_' + i + '*' + d + '*.csv')
                            data_list.append(files)

                        # combine all data types and files
                        data, rows_tmp, tps_tmp = combine_data_types(use_data, data_list, count)
                        events_all.append(e)
                        all_rows.extend(rows_tmp)
                        max_tps.extend(tps_tmp)

                        # split data by 3 minutes at 60 Hz
                        if max_rows == data.shape[0]:
                            np.savetxt(project + 'combined-data/' + e + '_' + exp + '_' + par + '_' + i + '_Combined.csv', \
                                data, delimiter=',')
                        # need to split into 3 minute chuncks
                        elif max_rows < data.shape[0]:
                            wcount = 0
                            # split into 3 min chuncks until left over data
                            while max_rows < data.shape[0]:
                                wcount = wcount + 1
                                np.savetxt(project + 'combined-data/' + e + '_' + exp + '_' + par + '_' + i + str(wcount) + '_Combined.csv', \
                                    data[0:max_rows, :], delimiter=',')
                                data = np.array(data[max_rows:, :])
                            # take care of left over data
                            if max_rows > data.shape[0]:
                                diff = max_rows - data.shape[0]
                                data = np.array(np.append(data, np.zeros((diff, data.shape[1])), axis=0))
                                np.savetxt(project + 'combined-data/' + e + '_' + exp + '_' + par + '_' + i + str(wcount + 1) + '_Combined.csv', \
                                    data, delimiter=',')
                        # need to pad with 0s
                        elif max_rows > data.shape[0]:
                            diff = max_rows - data.shape[0]
                            data = np.array(np.append(data, np.zeros((diff, data.shape[1])), axis=0))
                            np.savetxt(project + 'combined-data/' + e + '_' + exp + '_' + par + '_' + i + '_Combined.csv', \
                                data, delimiter=',')

                        # combine data into one matrix - ONLY if keeping in memory to run NN here
                        # if data_all == []:
                            # data_all = np.array(data)
                        # else:
                            # data_all = np.array(np.dstack((data_all, data)))

    # create graph
    plt.clf()
    plt.close()
    plt.hist(all_rows, bins=100)
    plt.title("# of Rows")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig('hist-rows.png')
    plt.clf()
    plt.close()
    plt.boxplot(all_rows)
    plt.title("# of Rows")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Counts")
    # plt.show()
    plt.savefig('boxplot-rows.png')   
    plt.clf()
    plt.close()
    plt.boxplot(all_rows, sym='')
    plt.title("# of Rows")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Counts")
    # plt.show()
    plt.savefig('boxplot-rows-nofliers.png')
    plt.clf()
    plt.close()
    plt.hist(max_tps, bins=100)
    plt.title("Max TP")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Frequency")
    # plt.show()
    plt.savefig('hist-tps.png')
    plt.clf()
    plt.close()
    plt.boxplot(max_tps)
    plt.title("Max TP")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Counts")
    # plt.show()
    plt.savefig('boxplot-tps.png')
    plt.clf()
    plt.close()
    plt.boxplot(max_tps, sym='')
    plt.title("Max TP")
    plt.xticks(fontsize=6, rotation=45)
    plt.ylabel("Counts")
    # plt.show()
    plt.savefig('boxplot-tps-nofliers.png')

    return data_all, events_all


# combine all data types data into one
def combine_data_types(use_data, data_list, sample):

    # static variable do not change
    info_cols = 10  # number of first info cols
    all_rows = []
    max_tps = []

    # for each data type included
    count = 0
    data_combined = []
    for i, dt in enumerate(use_data):

        # make sure correct number of cols are inserted for data types, pad 0 otherwise
        if dt == 'AppleWatch':
            l = ['Left', 'Right']
            cols = 9
        elif dt == 'EMG':
            l = ['Left', 'Right']
            cols = 8
        elif dt == 'IMU':
            l = ['Left', 'Right']
            cols = 14
        elif dt == 'PatientSpace':
            l = ['Camera']
            cols = 72
        elif dt == 'RawXY':
            l = ['Camera']
            cols = 72

        # for each file of the data type
        for j, f in enumerate(l):
            try:
                # read in data, drop unnecessary cols
                data = pd.read_csv(data_list[i][j], header=0, sep=',', index_col=False)

                # create graph data
                all_rows.append(data.shape[0])
                max_tps.append(max(data['seconds']))

                # create data matrix
                if j == 0:
                    data_final = np.array(data.iloc[:, info_cols:])
                    curr_secs = data['seconds']

                # join new data with data matrix
                else:
                    data.drop(data.columns[0:info_cols], axis=1, inplace=True)

                    # make sizes match by padding with 0s
                    if data.shape[0] < data_final.shape[0]:
                        diff = data_final.shape[0] - data.shape[0]
                        data = np.array(np.append(data, np.zeros((diff, data.shape[1])), axis=0))
                        curr_secs = data_final['seconds']
                    elif data.shape[0] > data_final.shape[0]:
                        diff = data.shape[0] - data_final.shape[0]
                        data_final = np.array(np.append(data_final, np.zeros((diff, data_final.shape[1])), axis=0))
                        curr_secs = data['seconds']
                    data_final = np.array(np.append(data_final, data, axis=1))

            # if no file for type pad with 0s
            except:
                # add zero data of some size, will be combined anyways
                if j == 0:
                    data_final = np.zeros((3, cols))
                    curr_secs = np.zeros((3, 1))

                # add zero data of same size
                else:
                    data = np.zeros(data_final.shape)
                    data_final = np.array(np.append(data_final, data, axis=1))

            count = count + 1

        # combine data types
        if data_combined == []:
            data_combined = np.array(data_final)
            secs = np.array(curr_secs)
        else:

            # pad with zeros if no data to append
            if (data_final == 0).all():
                data_combined = np.array(np.append(data_combined, np.zeros((data_combined.shape[0], data_final.shape[1])), axis=1))
            elif (data_combined == 0).all():
                data_combined = np.array(np.append(np.zeros((data_final.shape[0], data_combined.shape[1])), data_final, axis=1))
                secs = np.array(curr_secs)

            # join by seconds timepoints otherwise
            else:

                # use right table timepoints
                if data_final.shape[0] > data_combined.shape[0]:
                    tmp_data = np.zeros((data_final.shape[0], data_combined.shape[1] + data_final.shape[1]))

                    # loop through timepoints and find mathcing
                    iterator = 0
                    for t, tp in enumerate(curr_secs):
                        tmp_data[t, data_combined.shape[1]:] = data_final[t, :]
                        if iterator < len(secs):
                            curr_diff = abs(tp - secs[iterator])
                            if t + 1 < len(curr_secs):
                                next_diff = abs(curr_secs[t + 1] - secs[iterator])
                            else:
                                next_diff = curr_diff + 10

                        # use closest timepoint row
                        if curr_diff <= next_diff:
                            if iterator < data_combined.shape[0]:
                                tmp_data[t, 0:data_combined.shape[1]] = data_combined[iterator, :]
                                iterator = iterator + 1
                            else:
                                tmp_data[t, 0:data_combined.shape[1]] = np.zeros((1, data_combined.shape[1]))
                        else:
                            tmp_data[t, 0:data_combined.shape[1]] = np.zeros((1, data_combined.shape[1]))

                    # update secs col
                    secs = np.array(curr_secs)

                # use left table timepoints
                elif data_final.shape[0] <= data_combined.shape[0]:
                    tmp_data = np.empty((data_combined.shape[0], data_combined.shape[1] + data_final.shape[1]))

                    # loop through timepoints and find mathcing
                    iterator = 0
                    for t, tp in enumerate(secs):
                        tmp_data[t, 0:data_combined.shape[1]] = data_combined[t, :]
                        if iterator < len(curr_secs):
                            curr_diff = abs(tp - curr_secs[iterator])
                            if t + 1 < len(secs):
                                next_diff = abs(secs[t + 1] - curr_secs[iterator])
                            else:
                                next_diff = curr_diff + 10

                        # use closest timepoint row
                        if curr_diff <= next_diff:
                            if iterator < data_final.shape[0]:
                                tmp_data[t, data_combined.shape[1]:] = data_final[iterator, :]
                                iterator = iterator + 1
                            else:
                                tmp_data[t, data_combined.shape[1]:] = np.zeros((1, data_final.shape[1]))
                        else:
                            tmp_data[t, data_combined.shape[1]:] = np.zeros((1, data_final.shape[1]))

                data_combined = np.array(tmp_data)

    return data_combined, all_rows, max_tps


# main function
def main():

    # PARAMETERS TO CHANGE ###
    project = '/home/scullydm/DoDHandsFree/'    # path to main directory
    trials_threshold = 1   # min # of instances of event to include event in analysis
    max_rows = 10800    # 121456   # MAX # OF ROWS FOR LONGEST EVENT INSTANCE TO FULLY IMPUTATE
    # use any combination of data types but change exp/par: 'AppleWatch', 'Myo_EMG', 'Myo_IMU', 'PatientSpace', 'RawXY'
    use_data = ['AppleWatch', 'EMG', 'IMU', 'PatientSpace', 'RawXY']
    use_exp = ['E1', 'E2', 'E3']   # use any combinations of the experiments: 'E1', 'E2', 'E3'
    use_par = ['P1', 'P2', 'P3', 'P4', 'P5']  # use any combination of the participants: 'P1', 'P2', 'P3', 'P4', 'P5'

    # IMPUTATE DATA ###
    events = count_events(project, trials_threshold, use_data, use_exp, use_par)     # use events above threshold
    data_combined, events_all = read_in(project, events, use_data, use_exp, use_par, max_rows)   # read in/combine data types into instances
    print(len(events_all))

main()
