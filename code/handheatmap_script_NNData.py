import json
import datetime
import math
import os
import csv
import statistics
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# UPDATE dims and center?
dataJSON = None
frameRate = None
dimensions = [0, 0]
center = [0, 0]


# Grabs path to JSON file
def getJSON(experiment, participant_file):
    global dataJSON
    isValid = False
    while not(isValid):
        path = '/Users/deirdre/Documents/DODProject/CELA-Data/heatmaps/heatmaps/' \
            + 'Experiment_' + experiment + '/Event_Files/' + participant_file
        try:
            with open(path) as file:
                dataJSON = json.load(file)
                isValid = True
                pass
        except IOError as e:
            print('Unable to open file! Try again.')


# Grabs framerate
def getFrameRate():
    # p3: c1 is 24, c2 is 24, c3 is 59.94, c4 is 29.97 #
    # p4: c1 is 24, c2 is 24, c3 is 29.97, c4 is 29.97 #
    global frameRate
    frameRate = 24


# Grabs source video dimensions
def getDimensions():
    global dimensions
    dimensions[0] = 1080*2
    dimensions[1] = 1920*2


# Asks user for approximate coordinates of center of patient
def getCenter():
    global center
    center[0] = 540*2
    center[1] = 960*2


# Calculates the average position of all keypoints of a given person
def getPos(personKeypoints):
    averagePos = [0.0, 0.0]
    numValidPoints = 0
    for i in range(0, 18):
        index = i * 3
        if personKeypoints['pose_keypoints'][index + 2] != 0:
            averagePos[0] += personKeypoints['pose_keypoints'][index]
            averagePos[1] += personKeypoints['pose_keypoints'][index + 1]
            numValidPoints += 1
    averagePos[0] /= numValidPoints
    averagePos[1] /= numValidPoints
    return averagePos


# Assigns person closest to the center of the frame to the 0th index position
def attemptFix(jsonDict):
    blankFrame = False
    frameNum = 0
    for frame in jsonDict['frame']:
        if len(frame['people']) == 0:
            if not blankFrame:
                blankFrame = True

        elif blankFrame:
            blankFrame = False

        # move patient closest to center to 0 index
        if len(frame['people']) > 1:
            personIndex = 0
            mostCentered = 0
            closestDist = None

            for person in frame['people']:
                if personIndex == 0:
                    closestDist = getDist(getPos(person))
                else:
                    dist = getDist(getPos(person))
                    if dist < closestDist:
                        closestDist = dist
                        mostCentered = personIndex

                personIndex += 1
            if mostCentered != 0:

                frame['people'][0], frame['people'][mostCentered] = \
                    frame['people'][mostCentered], frame['people'][0]

        # move second medic closest to center to 1 index
        if len(frame['people']) > 2:
            personIndex = 0
            mostCentered = 1
            closestDist = None

            for person in frame['people']:
                if personIndex == 0:
                    continue
                elif personIndex == 1:
                    closestDist = getDist(getPos(person))
                else:
                    dist = getDist(getPos(person))
                    if dist < closestDist:
                        closestDist = dist
                        mostCentered = personIndex

                personIndex += 1
            if mostCentered != 1:

                frame['people'][1], frame['people'][mostCentered] = \
                    frame['people'][mostCentered], frame['people'][1]

        frameNum += 1


# Converts frame number to timestamp
def timestamp(frameNum, video_start_time):
    global frameRate
    secs = frameNum / float(frameRate)
    dec = round(secs, 3)
    dec = str(dec - int(dec))[1:]
    while len(dec) < 4:
        dec += '0'
    secs = int(secs)
    start_time = datetime.datetime.strptime(video_start_time[0:-3], "%Y-%m-%d %H:%M:%S")
    time = str(start_time + datetime.timedelta(seconds=secs))
    time += dec
    return time


# Calculates distance of person to center of the frame
def getDist(position):
    global dimensions
    global center
    a = center[0] - position[0]
    b = center[1] - position[1]
    dist = math.sqrt(a**2 + b**2)
    return dist


# Creates and returns heat map of hands from start frame to end frame
def createMap(jsonDict, write_file, limbs, video_start_time, thresh_flag):
    raw_xy = []; patient_space = []

    frameNum = 0
    for frame in jsonDict['frame']:

        personIndex = 0
        if len(frame['people']) > 1:
            for person in frame['people']:

                # paint body of patient
                if (personIndex == 0):

                    # save body info for csv array
                    if write_file == 1:

                        # get patient x, y
                        patient_xy_tmp = []
                        for i in range(54)[0::3]:
                            patient_xy_tmp.append(person['pose_keypoints'][i])
                        for i in range(54)[1::3]:
                            patient_xy_tmp.append(person['pose_keypoints'][i])

                        body_tmp_array = []
                        for i in range(54)[0::3]:
                            body_tmp_array.append([person['pose_keypoints'][i], person['pose_keypoints'][i + 1]])

                # paint hands of medics
                if (personIndex == 1):

                    # create line in csv
                    if write_file == 1:

                        # get patient x, y
                        medic_xy_tmp = []
                        for i in range(54)[0::3]:
                            medic_xy_tmp.append(person['pose_keypoints'][i])
                        for i in range(54)[1::3]:
                            medic_xy_tmp.append(person['pose_keypoints'][i])

                        # get distances for array
                        distances = []
                        for j in range(2):
                            if j == 0:
                                x1 = person['pose_keypoints'][21]
                                y1 = person['pose_keypoints'][22]
                            else:
                                x1 = person['pose_keypoints'][12]
                                y1 = person['pose_keypoints'][13]
                            for xy2 in body_tmp_array:
                                x2 = xy2[0]
                                y2 = xy2[1]
                                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                                distances.append(dist)

                        # find closest limb to LH, RH
                        lh_dist = min(distances[0:18])
                        rh_dist = min(distances[18:36])
                        lh_idx = distances[0:18].index(lh_dist)
                        rh_idx = distances[18:36].index(rh_dist)

                        # for closets limbs historgrams with or without threshold
                        lh_closest_counts = np.zeros(18); rh_closest_counts = np.zeros(18);
                        if not thresh_flag:
                            lh_closest_counts[lh_idx] = lh_closest_counts[lh_idx] + 1
                            rh_closest_counts[rh_idx] = rh_closest_counts[rh_idx] + 1
                        else:
                            if lh_dist < 100:
                                lh_closest_counts[lh_idx] = lh_closest_counts[lh_idx] + 1
                            if rh_dist < 100:
                                rh_closest_counts[rh_idx] = rh_closest_counts[rh_idx] + 1

                        patient_space.append(np.hstack([frameNum, timestamp(frameNum, video_start_time), lh_closest_counts, rh_closest_counts, distances]))

                personIndex += 1

            raw_xy.append(np.hstack([frameNum, timestamp(frameNum, video_start_time), patient_xy_tmp, medic_xy_tmp]))

        frameNum += 1

    return raw_xy, patient_space


# Check if point is within dimensions
def inBounds(point):
    if point[0] >= dimensions[0]:
        return False
    if point[0] < 0:
        return False
    if point[1] >= dimensions[1]:
        return False
    if point[1] < 0:
        return False
    return True


def main():

    # designate participant/experiment UPDATE
    thresh_flag = 0
    ###################### UPDATE FRAMERATE ###########################
    experiment = '2'    # '1','2'
    participant = '4'   # '1','2','3','4'
    camera = '2'    # '1','2','3','4'
    participant_file = 'May23(2).json'  # May21(3), May21(4), May21(Morning)(1), May21(Morning)(2), May22(2), May23(2)
    video_start_time = '2018-05-23 09:47:33-05'
        ### E2 P1 Start Times: ###
        # Camera 1: 2018-05-21 09:25:45
        # Camera 2: 2018-05-21 09:26:12
        # Clap 1: 2018-05-21 09:30:33
        ### E2 P2 Start Times: ###
        # Camera 3: 2018-05-21 13:50:59
        # Camera 4: 2018-05-21 13:51:20
        # Clap 1: 2018-05-21 13:51:32
        ### E2 P3 Start Times: ###
        # p3: c1 is 24, c2 is 24, c3 is 59.94, c4 is 29.97 #
        # Camera 1: 2018-05-22 12:53:13
        # Camera 2: 2018-05-22 12:51:23
        # Clap 1: 2018-05-22 13:13:39
        ### E2 P4 Start Times: ###
        # p4: c1 is 24, c2 is 24, c3 is 29.97, c4 is 29.97 #
        # Camera 1: 2018-05-23 09:47:33
        # Camera 2: 2018-05-23 09:47:33
        # Clap 1: 2018-05-23 10:22:56

    getJSON(experiment, participant_file)
    getFrameRate()
    attemptFix(dataJSON)

    # write by second closest limb count vector and limb distances vector
    headers1 = ['Frame','Timestamp','LH Closest Nose','LH Closest Neck','LH Closest Right Shoulder','LH Closest Right Elbow','LH Closest Right Wrist','LH Closest Left Shoulder', \
            'LH Closest Left Elbow','LH Closest Left Wrist','LH Closest Right Hip', 'LH Closest Right Knee','LH Closest Right Ankle','LH Closest Left Hip','LH Closest Left Knee', \
            'LH Closest Left Ankle','LH Closest Right Eye','LH Closest Left Eye','LH Closest Right Ear','LH Closest Left Ear', \
            'RH Closest Nose','RH Closest Neck','RH Closest Right Shoulder','RH Closest Right Elbow','RH Closest Right Wrist','RH Closest Left Shoulder','RH Closest Left Elbow', \
            'RH Closest Left Wrist','RH Closest Right Hip','RH Closest Right Knee','RH Closest Right Ankle','RH Closest Left Hip','RH Closest Left Knee','RH Closest Left Ankle', \
            'RH Closest Right Eye','RH Closest Left Eye','RH Closest Right Ear','RH Closest Left Ear', \
            'LH Dist Nose','LH Dist Neck','LH Dist Right Shoulder','LH Dist Right Elbow','LH Dist Right Wrist','LH Dist Left Shoulder', \
            'LH Dist Left Elbow','LH Dist Left Wrist','LH Dist Right Hip', 'LH Dist Right Knee','LH Dist Right Ankle','LH Dist Left Hip','LH Dist Left Knee', \
            'LH Dist Left Ankle','LH Dist Right Eye','LH Dist Left Eye','LH Dist Right Ear','LH Dist Left Ear', \
            'RH Dist Nose','RH Dist Neck','RH Dist Right Shoulder','RH Dist Right Elbow','RH Dist Right Wrist','RH Dist Left Shoulder','RH Dist Left Elbow', \
            'RH Dist Left Wrist','RH Dist Right Hip','RH Dist Right Knee','RH Dist Right Ankle','RH Dist Left Hip','RH Dist Left Knee','RH Dist Left Ankle', \
            'RH Dist Right Eye','RH Dist Left Eye','RH Dist Right Ear','RH Dist Left Ear']
    f = open('/Users/deirdre/Documents/DODProject/CELA-Data/heatmaps/heatmaps/' + 'Experiment_' \
            + experiment + '/' + 'p' + participant + '_C' + camera + '_PatientSpace.csv', 'w')
    writer_ld = csv.writer(f)
    writer_ld.writerow(headers1)

    # write by second closest limb count vector and limb distances vector
    headers2 = ['Frame','Timestamp','Patient X Nose','Patient X Neck','Patient X Right Shoulder','Patient X Right Elbow','Patient X Right Wrist','Patient X Left Shoulder', \
            'Patient X Left Elbow','Patient X Left Wrist','Patient X Right Hip', 'Patient X Right Knee','Patient X Right Ankle','Patient X Left Hip','Patient X Left Knee', \
            'Patient X Left Ankle','Patient X Right Eye','Patient X Left Eye','Patient X Right Ear','Patient X Left Ear', \
            'Patient Y Nose','Patient Y Neck','Patient Y Right Shoulder','Patient Y Right Elbow','Patient Y Right Wrist','Patient Y Left Shoulder', \
            'Patient Y Left Elbow','Patient Y Left Wrist','Patient Y Right Hip', 'Patient Y Right Knee','Patient Y Right Ankle','Patient Y Left Hip','Patient Y Left Knee', \
            'Patient Y Left Ankle','Patient Y Right Eye','Patient Y Left Eye','Patient Y Right Ear','Patient Y Left Ear', \
            'Medic X Nose','Medic X Neck','Medic X Right Shoulder','Medic X Right Elbow','Medic X Right Wrist','Medic X Left Shoulder','Medic X Left Elbow', \
            'Medic X Left Wrist','Medic X Right Hip','Medic X Right Knee','Medic X Right Ankle','Medic X Left Hip','Medic X Left Knee','Medic X Left Ankle', \
            'Medic X Right Eye','Medic X Left Eye','Medic X Right Ear','Medic X Left Ear', \
            'Medic Y Nose','Medic Y Neck','Medic Y Right Shoulder','Medic Y Right Elbow','Medic Y Right Wrist','Medic Y Left Shoulder','Medic Y Left Elbow', \
            'Medic Y Left Wrist','Medic Y Right Hip','Medic Y Right Knee','Medic Y Right Ankle','Medic Y Left Hip','Medic Y Left Knee','Medic Y Left Ankle', \
            'Medic Y Right Eye','Medic Y Left Eye','Medic Y Right Ear','Medic Y Left Ear']
    f = open('/Users/deirdre/Documents/DODProject/CELA-Data/heatmaps/heatmaps/' + 'Experiment_' \
            + experiment + '/' + 'p' + participant + '_C' + camera + '_RawXY.csv', 'w')
    writer_xy = csv.writer(f)
    writer_xy.writerow(headers2)

    # make output folders
    directory = '/Users/deirdre/Documents/DODProject/CELA-Data/heatmaps/heatmaps/' + 'Experiment_' + experiment
    if not os.path.exists(directory):
        os.makedirs(directory)

    # csv info
    limbs = ['Nose','Neck','Right Shoulder','Right Elbow','Right Wrist','Left Shoulder','Left Elbow','Left Wrist','Right Hip', \
        'Right Knee','Right Ankle','Left Hip','Left Knee','Left Ankle','Right Eye','Left Eye','Right Ear','Left Ear']

    # create maps
    raw_xy, patient_space = createMap(dataJSON, 1, limbs, video_start_time, thresh_flag)
    writer_xy.writerows(raw_xy)
    writer_ld.writerows(patient_space)


main()
