# EMS-Prediction
EMS Motion Data Visual Analytics

GitHub: https://github.com/dmscul11/EMS-Prediction

Box for Data: https://vanderbilt.app.box.com/folder/66875332205

Python Background Server for D3: python3 -m http.server


Main Code:
- read_processed_data.py = Main function to read in processed data and run ML random forest
- combine_data.py = Main function to read in processed data, combine and preprocess data, and run Neural Network
- combined_NN.py = Main function to read in processed combined data instead of creating first, and run CNN
- visualizations.py = Main function to read in ML output and create visualizations**************

Run Once Code:
- handheatmap_script_NNData.py = generates csv files from openpose output
- analyze-watchdata-separate-events.py = generate basic plots of apple watch data
- limbs_randomforest.py = run random forest predictor on openpose data
- parse_data_files.py = preprocess and split data into csvs by procedure and data collection

Event-Files:
- events csv files for data

Related Works:
- similar works PDFs to implement
