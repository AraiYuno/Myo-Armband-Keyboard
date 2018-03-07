import csv
from Interval import Interval
from collections import defaultdict
from bisect import bisect_left

# Global Variables

#==================================================================================
# separateSets
#   This function separate each of backward, forward, left, right and enter buttons'
#   10 inputs as an interval using "no-change periods"
#==================================================================================
def separateSets(orientation_path, accelero_path):
    orientation_list = read_in_file(orientation_path)
    accelero_list = read_in_file(accelero_path)
    ori_timestamp_list = find_nochange_periods(orientation_list['timestamp'], orientation_list['x'],
                                               orientation_list['y'], orientation_list['z'])
    peak_value = find_peak(accelero_list['y'])
    print(peak_value)
    intervals = make_intervals(peak_value, ori_timestamp_list, accelero_list)
    intervals = refineSets(intervals)
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
               #print( timestamp_list[i-1] +", " + timestamp_list[i])
                to_return.append(Interval(timestamp_list[i-1], timestamp_list[i]))
                break
            start_index = start_index+1
        i = i+1
    return to_return


#==================================================================================
# refineSets
#   takes in intervals list as a parameter and make them all to 10 intervals
#==================================================================================
def refineSets( intervals ):
    while len(intervals) != 10:
        if len(intervals) > 10:
            intervals.pop(0)
        if len(intervals) < 10:
            intervals.append(intervals[len(intervals)-1])
    return intervals






#==================================================================================
# extract_emg
#   returns the actual EMG data sets that are separated by the intervals 2D array.
#   For example, emg_column1[0][0] contains the first interval's first EMG data record
#==================================================================================
def extract_emg( intervals, emg_path, emg_num ):
    emg = read_in_file(emg_path)
    emg_timestamp = emg['timestamp']
    emg_column = emg[emg_num]  # specific column within the emg .csv file
    emg_intervals = find_closest_timestamp_interval_list(intervals, emg_timestamp)
    to_return = temp = []
    # now we have the emg_intervals. We actually need the "DATA" sets
    add_data = False
    i = j = 0
    while i < len(emg_column) and j < len(emg_intervals):
        if float(emg_intervals[j].start_time) == float(emg_timestamp[i]):
            temp = []
            add_data = True
        if float(emg_intervals[j].end_time) == float(emg_timestamp[i]):
            add_data = False
            to_return.append(temp)
            j = j + 1
        if add_data:
            temp.append(emg_column[i])
        i = i + 1
    return to_return



#==================================================================================
# find_closest_timestamp
#   this function finds the closest timestamp to the given interval within the list
#   and returns the new list of intervals with the start and end time that are
#   corresponding to the given list
#==================================================================================
def find_closest_timestamp_interval_list(intervals, list):
    i = 0
    to_return = []
    while i < len(intervals):
        # now we have two sorted list for both start time and end time.
        start_time = min(map(float, list), key=lambda x: abs(x - float(intervals[i].start_time)))
        end_time = min(map(float, list), key=lambda x: abs(x - float(intervals[i].end_time)))
        to_return.append(Interval(str(start_time), str(end_time)))
        i = i + 1
    return to_return



#TODO: Now that we have 10 intervals(9 or 11), we split into training and testing data. 0~8 will be training and 10 will be testing data

# MAIN EXECUTION
backward_invervals = separateSets('./data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv')
emg_column1 = extract_emg(backward_invervals, './data/Backward/emg-1456704054.csv', 'emg1')
#print("Printing EMG1 data. 10 intervals' actual data")
#print("Backward: ", len(backward_invervals))


#forward_intervals = separateSets('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv')
#print("Forward: ", len(forward_intervals))
#print(forward_intervals[0].end_time)
#right_intervals = separateSets('./data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv')
#print("Right: ", len(right_intervals))
#left_intervals = separateSets('./data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv')
#print("left: ", len(left_intervals))
#enter_intervals = separateSets('./data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv')
#print("enter: ", len(enter_intervals))
