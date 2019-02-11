import pandas as pd
import glob


project = '/Users/deirdre/Documents/DODProject/CELA-Data/NeuralNetwork/'
files = glob.glob(project + 'event-files/E*.csv')

for f in files:

    # read in events
    events = pd.DataFrame.from_csv(f)
    print(f)
    print(events)

    # cycle through data types
    data_types = ['AppleWatch_Right', 'AppleWatch_Left', 'Myo_EMG', 'Myo_IMU', 'Closest_Limbs']
    for dt in data_types:
        print(dt)
