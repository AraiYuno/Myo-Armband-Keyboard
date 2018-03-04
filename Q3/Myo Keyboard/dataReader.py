import csv
from Interval import Interval
from collections import defaultdict

# Global Variables

#==================================================================================
# separateSets
#   This function separate each of backward, forward, left, right and enter buttons'
#   10 inputs as an interval using "no-change periods"
#
# TODO: I have not set a return value for these intervals since we are not sure how to use the intervals
#==================================================================================
def separateSets(orientation_path, accelero_path):
    orientation_list = read_in_file(orientation_path)
    accelero_list = read_in_file(accelero_path)
    ori_timestamp_list = find_nochange_periods(orientation_list['timestamp'], orientation_list['x'],
                                               orientation_list['y'], orientation_list['z'])
    peak_value = find_peak(accelero_list['y'])
    intervals = make_intervals(peak_value, ori_timestamp_list, accelero_list)
    return intervals


#==================================================================================
# read_in_file
#   reads in a .csv file
#==================================================================================
def read_in_file( path ):
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open(path) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k
    return columns


#==================================================================================
# find_nochange_periods
#   find the no_change periods from orientationEuler and orientation .csv files
#==================================================================================
def find_nochange_periods(timestamp, x_axis, y_axis, z_axis):
    to_return = []
    i = 0
    while i < len(timestamp)-3:
        if (x_axis[i] == x_axis[i+1] and y_axis[i] == y_axis[i+1] and
            z_axis[i] == z_axis[i+1]):
            to_return.append(timestamp[i])
            i = i+3
        else:
            i = i+1
    return to_return


#==================================================================================
# find_peak
#   returns the lowest out of the 20 top peak y values in accelerometer.
#==================================================================================
def find_peak(y_axis):
    sorted_list = sorted(y_axis)
    return sorted_list[30]


#==================================================================================
# make_intervals
#   returns 10 intervals that can be used in EMG
#==================================================================================
def make_intervals(peak_value, timestamp_list, accelero_list):
    i = 1
    accelero_timestamp = accelero_list['timestamp']
    accelero_y = accelero_list['y']
    to_return = []    # to return the list of 10 intervals( could be more than 10 )
    while i < len(timestamp_list):
        start_index = accelero_timestamp.index(timestamp_list[i-1]) #finds the matching timestamp in accelerometer
        end_index = accelero_timestamp.index(timestamp_list[i])
        while start_index < end_index:
            if accelero_y[start_index] < peak_value:
                to_return.append(Interval(timestamp_list[i-1], timestamp_list[i]))
                print(timestamp_list[i-1] + ", " + timestamp_list[i])
                break
            start_index = start_index+1
        i = i+1
    return to_return


# MAIN EXECUTION
backward_invervals = separateSets('./data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv')
print("No. of backward_intervals: ", len(backward_invervals))
forward_invervals = separateSets('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv')
print("No. of forward_intervals: ", len(forward_invervals))
right_intervals = separateSets('./data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv')
print("No. of right_intervals: ", len(right_intervals))
left_intervals = separateSets('./data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv')
print("No. of left_intervals: ", len(left_intervals))
enter_intervals = separateSets('./data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv')
print("No. of enter_intervals: ", len(enter_intervals))
