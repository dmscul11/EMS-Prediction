# EMS-Prediction
EMS Motion Data Visual Analytics


########################################################################################
########################################################################################
########################################################################################

USAGE INFORMATION:

Versions: python v3.6.3, D3 v5.9.2, safari v12.1, chrome v73

1) download /dashboard-d3 entire directory ONLY (this includes the data in it)
2) in terminal navigate into /dashboard-d3 directory
3) execute: python3 -m http.server
3) using either safari or google chrome navigate to: http://0.0.0.0:8000/
4) click and interacte with dashboard

########################################################################################
########################################################################################
########################################################################################



GitHub: https://github.com/dmscul11/EMS-Prediction

Box for Original Data: https://vanderbilt.app.box.com/folder/66875332205

Python Background Server for D3: python3 -m http.server
http://0.0.0.0:8000/

Tinker for CNN: /home/scullydm/DoDHandsFree/
    (use python3.6, pip3.6): nohup python3.6 combined_NN.py &



combined-d3:
- both squares D3 implementation

dashboard-d3:
- final squares and dashboard D3 implementation

tree-d3:
- Tree D3 implementation

squares-d3:
- Squares D3 implementation

stacks-d3:
- Squares and Tree combined D3 implementation


Main Code:
- ***read_processed_data.py = Main function to read in processed data and run ML random forest
- visualizations.py = Main function to read in ML output and create visualizations**************
- combined_NN.py = Main function to read in processed imputated data and run CNN

        --> input from combined-data
        
        --> Update params in main() function, lines 165 - 167


Parse and Combine Data Code (Run just once, UNLESS EVENTS TIMING HAS CHANGED OR MORE DATA ADDED):
- imputate_data_small.py (combine_data.py) = Main function to read in processed data and output imputation to small row size and wrap the rest
        --> input from processed-data
        --> Update params in main() function, lines 242-248
- imputate_data_large.py (combine_data.py) = Main function to read in processed data and output imputation to max row size
        --> input from processed-data
        --> Update params in main() function, lines 242-248
- parse_data_files.py = preprocess raw data and split data into individual csvs by event files info and data collection
        --> input from raw-data and event-files

Run Once Code:
- combine_data.py = Main function to read in processed data, combine and preprocess data, and run Neural Network
- handheatmap_script_NNData.py = generates csv files from openpose output
- analyze-watchdata-separate-events.py = generate basic plots of apple watch data
- limbs_randomforest.py = run random forest predictor on openpose data
- parse_data_files.py = preprocess and split data into csvs by procedure and data collection


Combined-Data:
- csv imputated data files of all same size and all data types by experiment, participant, event, trail #, time zeroed out
- input to combined_NN
- output from imputate_data.py combining data type
- generated from processed-data

Processed-Data:
- csv files of data time specific to experiment, participant, event, trial #, timestamp, data type
- input to imputate_data.py
- output from parse_data_files.py
- generated from raw-data

raw-data:
- csv files of data time specific to experiment, participant, and data type
- generated from SQL pulls (apple watch) or from Box (MYO) or AWS (video and open pose output)

Event-Files:
- procedure event details csv files for data for each experiment and participant
