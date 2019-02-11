# Analyze CELA Data
# Deirdre Scully
# June 2018
# Inputs csv file of apple watch CELA data from psql database and list of events with start and end times
# Creates plots of each event from input spreadsheet and outputs same csv input file with events mapped in
# Separates out experiment by participant by procedure by iteration of procedure


import os
import csv
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot
from datetime import datetime
import matplotlib.dates as mdates
from datetime import datetime


# read in csv data and preprocess
def read_data(file):

	# open file
	with open(file, 'r') as df:
		reader = csv.reader(df)

		# get data labels
		# headers = reader.__next__()
		
		# get all data
		data = [data for data in reader]
		# data_array = np.asarray(data)
		data_array = pd.DataFrame(data=data, columns = ['id', 'session_id', 'start_time', \
			'end_time', 'n_pts', 'frequency', 'vital', 'session_notes', 'device_id', 'device_make_model', \
			'time', 'xaccel', 'yaccel', 'zaccel', 'xrot', 'yrot', 'zrot', 'yaw', 'pitch', 'roll', 'session_id'])
		data_array['time'] = data_array['time'].astype(str).str[:-3] # strip timezone information so doesnt false convert
		data_array['time'] = pd.to_datetime(data_array['time'][0:-3], errors ='coerce')

	return data_array
	

# read in events text list
def read_events(file):

	# open file
	with open(file, 'r') as df:

		events = []
		next(df)
		for line in df:
			# parse line
			tmp = line[:-1].split(",")

			# remove empty events
			if (len(tmp) >= 2) and (tmp[2] != ''):
				events.append({'event': tmp[0], 'event_n': tmp[1], 'start_time': datetime.strptime(tmp[2] + '.000001', \
					'%Y-%m-%d %H:%M:%S.%f'), 'end_time': datetime.strptime(tmp[3] + '.999999', '%Y-%m-%d %H:%M:%S.%f'), \
				'length': tmp[4], 'video_start': tmp[5], 'video_stop': tmp[6]})

		# order by start time
		events = np.asarray(events)
		events = sorted(events, key=lambda time: (time['start_time'], time['end_time']))
		events = pd.DataFrame(data=events)

	return events


# make plots for acceleration
def plot_accel(left_event_data, right_event_data, path, event, experiment, participant):

	# get left accel data
	left_times = list(left_event_data['time'])
	left_xaccel = list(left_event_data['xaccel'].astype('float'))
	left_yaccel = list(left_event_data['yaccel'].astype('float'))
	left_zaccel = list(left_event_data['zaccel'].astype('float'))
	left_xrot = list(left_event_data['xrot'].astype('float'))
	left_yrot = list(left_event_data['yrot'].astype('float'))
	left_zrot = list(left_event_data['zrot'].astype('float'))
	left_yaw = list(left_event_data['yaw'].astype('float'))
	left_pitch = list(left_event_data['pitch'].astype('float'))
	left_roll = list(left_event_data['roll'].astype('float'))

	# get right accel data
	right_times = list(right_event_data['time'])
	right_xaccel = list(right_event_data['xaccel'].astype('float'))
	right_yaccel = list(right_event_data['yaccel'].astype('float'))
	right_zaccel = list(right_event_data['zaccel'].astype('float'))
	right_xrot = list(right_event_data['xrot'].astype('float'))
	right_yrot = list(right_event_data['yrot'].astype('float'))
	right_zrot = list(right_event_data['zrot'].astype('float'))
	right_yaw = list(right_event_data['yaw'].astype('float'))
	right_pitch = list(right_event_data['pitch'].astype('float'))
	right_roll = list(right_event_data['roll'].astype('float'))

	# format plots
	pyplot.clf()
	mins = mdates.MinuteLocator()
	secs = mdates.SecondLocator()
	minsFmt = mdates.DateFormatter('%M')
	secsFmt = mdates.DateFormatter('%S')
	fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2, figsize=(20, 15))
	pyplot.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.25)
	fig.autofmt_xdate()
	fig.suptitle(event['event'] + ' Trial Number ' + event['event_n'], size = 30)

	# left accel plot
	ax1.plot(left_times, left_xaccel, label = "X")
	ax1.plot(left_times, left_yaccel, label = "Y")
	ax1.plot(left_times, left_zaccel, label = "Z")
	ax1.set_title('Left Arm: Acceleration', size = 20)
	ax1.xaxis.set_major_locator(secs)
	ax1.xaxis.set_major_formatter(secsFmt)
	ax1.legend(loc=0, prop={'size': 10})
	ax1.set_xlabel('Time', fontsize = 15)
	ax1.set_ylabel('Acceleration', fontsize = 15)

	# left orientation plot
	ax2.plot(left_times, left_xrot, label = "X Rotation")
	ax2.plot(left_times, left_yrot, label = "Y Rotation")
	ax2.plot(left_times, left_zrot, label = "Z Rotation")
	ax2.plot(left_times, left_yaw, label = "Yaw")
	ax2.plot(left_times, left_pitch, label = "Pitch")
	ax2.plot(left_times, left_roll, label = "Roll")
	ax2.set_title('Left Arm: Orientation', size = 20)
	ax2.xaxis.set_major_locator(secs)
	ax2.xaxis.set_major_formatter(secsFmt)
	ax2.legend(loc=0, prop={'size': 10})
	ax2.set_xlabel('Time', fontsize = 15)
	ax2.set_ylabel('Orientation', fontsize = 15)

	# right accel plot
	ax3.plot(right_times, right_xaccel, label = "X")
	ax3.plot(right_times, right_yaccel, label = "Y")
	ax3.plot(right_times, right_zaccel, label = "Z")
	ax3.set_title('Right Arm: Acceleration', size = 20)
	ax3.xaxis.set_major_locator(secs)
	ax3.xaxis.set_major_formatter(secsFmt)
	ax3.legend(loc=0, prop={'size': 10})
	ax3.set_xlabel('Time', fontsize = 15)
	ax3.set_ylabel('Acceleration', fontsize = 15)

	# right orientation plot
	ax4.plot(right_times, right_xrot, label = "X Rotation")
	ax4.plot(right_times, right_yrot, label = "Y Rotation")
	ax4.plot(right_times, right_zrot, label = "Z Rotation")
	ax4.plot(right_times, right_yaw, label = "Yaw")
	ax4.plot(right_times, right_pitch, label = "Pitch")
	ax4.plot(right_times, right_roll, label = "Roll")
	ax4.set_title('Right Arm: Orientation', size = 20)
	ax4.xaxis.set_major_locator(secs)
	ax4.xaxis.set_major_formatter(secsFmt)
	ax4.legend(loc=0, prop={'size': 10})
	ax4.set_xlabel('Time', fontsize = 15)
	ax4.set_ylabel('Orientation', fontsize = 15)

	# save plots
	img_path = os.path.join(path, 'Data Collection', 'Experiment_' + experiment, 'Procedures', event['event'], 'Saved_Graphs')
	if not os.path.exists(img_path):
		os.makedirs(img_path) # os.mkdirs()
	img_file = img_path + '/p' + participant + '_' + event['event'] + '_' + event['event_n'] + '_ios_data.png'
	pyplot.savefig(img_file)
	pyplot.clf()



# main analysis
def main():

	# UPDATE ACCORDINGLY to read in data file
	experiment = '2'
	session = '6'
	participant = '4'
	path = '/Users/deirdre/Documents/DODProject/CELA-Data/'

	# set up files to read in
	left_watch_file = os.path.join(path, 'applewatch-data', 'applewatch-LeftCELATest' + session + '.csv')
	right_watch_file = os.path.join(path, 'applewatch-data', 'applewatch-RightCELATest' + session + '.csv')
	left_data = read_data(left_watch_file)
	right_data = read_data(right_watch_file)

	# UPDATE ACCORDINGLY to read in events file - FOR EVENTS FILE WITH START TIMES AND END TIMES
	events_file = os.path.join(path, 'Data Collection', 'data', 'p' + session + '_events_cleaned' + '.csv')
	events = read_events(events_file)

	# find matching events
	for e, event in events.iterrows():

		# filter dataframe by event start and end times then order by time
		print(event)
		left_event_data = left_data[(left_data['time'] >= event.start_time) \
			& (left_data['time'] <= event.end_time)].sort_values(by = 'time', ascending = True)
		right_event_data = right_data[(right_data['time'] >= event.start_time) \
			& (right_data['time'] <= event.end_time)].sort_values(by = 'time', ascending = True)

		# plot right and left together
		plot_accel(left_event_data, right_event_data, path, event, experiment, participant)

		# save event data into separate csv file - update save path
		csv_path = os.path.join(path, 'Data Collection', 'Experiment_' + experiment, 'Procedures', event['event'], 'p' + participant)
		if not os.path.exists(csv_path):
			os.makedirs(csv_path)
		l_file = csv_path + '/p' + participant + '_' + event['event'] + '_' + event['event_n'] + '_ios_data_left.csv'
		left_event_data.to_csv(l_file)
		r_file = csv_path + '/p' + participant + '_' + event['event'] + '_' + event['event_n'] + '_ios_data_right.csv'
		right_event_data.to_csv(r_file)


# run main
main()
