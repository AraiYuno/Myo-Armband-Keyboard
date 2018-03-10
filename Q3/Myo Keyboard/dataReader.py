#=====================================================================================================#
#                                                                                                     #
# dataReader.py                                                                                       #
#                                                                                                     #
# Author: Kyle                                                                                        #
# Purpose: This .py file handles the data structures that can be used in the multi-layer perceptron.  #
#=====================================================================================================#


import csv
from Interval import Interval
from collections import defaultdict
import numpy as np
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
#   returns 10 intervals that can be used to form 10 sets of data
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
                to_return.append(Interval(accelero_timestamp[start_index-15], accelero_timestamp[start_index+30]))
                break
            start_index = start_index+1
        i = i+1
    return refineSets(to_return)


#==================================================================================
# refineSets
#   takes in intervals list as a parameter and make them all to 10 intervals
#==================================================================================
def refineSets(intervals):
    while len(intervals) != 10:
        if len(intervals) > 10:
            intervals.pop(0)
        if len(intervals) < 10:
            intervals.append(intervals[len(intervals)-2])
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
    to_return = []
    temp = []
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
            i = 0
            j = j + 1
        if add_data:
            temp.append(emg_column[i])
        i = i + 1
    return to_return
#==================================================================================
# extract_gyro
#   returns the actual data sets of gyerometer with given 10 intervals
#   For example, emg_column1[0][0] contains the first interval's first gyrometer's
#   data record
#==================================================================================
def extract_gyro (intervals, gyro_path):
    gyro = read_in_file(gyro_path)
    gyro_data = gyro['y']
    to_return = []
    gyro_timestamp = gyro['timestamp']
    add_data = False
    i = j = 0
    while i < len(gyro_data) and j < len(intervals):
        if float(intervals[j].start_time) == float(gyro_timestamp[i]):
            temp = []
            add_data = True
        if float(intervals[j].end_time) == float(gyro_timestamp[i]):
            add_data = False
            to_return.append(temp)
            i = 0
            j = j + 1
        if add_data:
            temp.append(gyro_data[i])
        i = i + 1
    return to_return


#==================================================================================
# extract_accelero
#   returns the actual data sets of accelerometer with given 10 intervals
#   For example, emg_column1[0][0] contains the first interval's first accelerometer's
#   data record
#==================================================================================
def extract_accelero(intervals, accelero_path, axis):
    accelero_file = read_in_file(accelero_path)
    accelero_timestamp = accelero_file['timestamp']
    accelero_axis = accelero_file[axis]
    accelero_intervals = find_closest_timestamp_interval_list(intervals, accelero_timestamp)
    to_return = []
    temp = []
    # now we have the emg_intervals. We actually need the "DATA" sets
    add_data = False
    i = 0
    j = 0

    while i < len(accelero_axis) and j < len(accelero_intervals):
        if float(accelero_intervals[j].start_time) == float(accelero_timestamp[i]):
            temp = []
            add_data = True
        if float(accelero_intervals[j].end_time) == float(accelero_timestamp[i]):
            add_data = False
            to_return.append(temp)
            i = 0
            j = j + 1
        if add_data:
            temp.append(accelero_axis[i])
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


#==================================================================================
# get_input_x
#   this function takes in the paths of all the necessary .csv files and the number
#   of emg data columns. This function creates a data structure that can be used
#   as as input for our MLP.
#
#   The returning data structure looks like
#   [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
#   Where to_return[0][0] would get you the emg data for the input and to_return[0][1]
#   would get you the expected output for the corresponding emg data.
#==================================================================================
def get_input_x( path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 num_emg_columns, path_forward_emg, path_backward_emg, path_left_emg, path_right_emg, path_enter_emg):
    to_return = []
    emg_data_list = []
    expected_output_list = []
    # timestamp list for each key
    emg_forward_timestamp = read_in_file(path_forward_emg)['timestamp']
    emg_backward_timestamp = read_in_file(path_backward_emg)['timestamp']
    emg_left_timestamp = read_in_file(path_left_emg)['timestamp']
    emg_right_timestamp = read_in_file(path_right_emg)['timestamp']
    emg_enter_timestamp = read_in_file(path_enter_emg)['timestamp']

    # find the average length of emg time interval ( for generic case )
    average_length = 182

    #print("AVERAGE LENGTH: ", average_length)
    accelero_forward_intervals = separateSets(path_forward_orientation, path_forward_accelero)
    accelero_backward_intervals = separateSets(path_backward_orientation, path_backward_accelero)
    accelero_left_intervals = separateSets(path_left_orientation, path_left_accelero)
    accelero_right_intervals = separateSets(path_right_orientation, path_right_accelero)
    accelero_enter_intervals = separateSets(path_enter_orientation, path_enter_accelero)

    # now we have all the intervals, we need to extract emg data using these intervals
    i = 1
    while i < num_emg_columns:
        emg_column_str = 'emg'+str(i+1)
        forward_emg_column = extract_emg(accelero_forward_intervals, path_forward_emg, emg_column_str)
        backward_emg_column = extract_emg(accelero_backward_intervals, path_backward_emg, emg_column_str)
        left_emg_column = extract_emg(accelero_left_intervals, path_left_emg, emg_column_str)
        right_emg_column = extract_emg(accelero_right_intervals, path_right_emg, emg_column_str)
        enter_emg_column = extract_emg(accelero_enter_intervals, path_enter_emg, emg_column_str)

        # append all EMG data to the emg_data_list in one BIG LIST
        # append the corresponding output expectation in an expected_output_list
        for i in range(0, len(forward_emg_column)):
            print("nana", )
            emg_data_list.append(forward_emg_column[i])
            expected_output_list.append([1, 0, 0, 0, 0])
            emg_data_list.append(backward_emg_column[i])
            expected_output_list.append([0, 1, 0, 0, 0])
            emg_data_list.append(left_emg_column[i])
            expected_output_list.append([0, 0, 1, 0, 0])
            emg_data_list.append(right_emg_column[i])
            expected_output_list.append([0, 0, 0, 1, 0])
            emg_data_list.append(enter_emg_column[i])
            expected_output_list.append([0, 0, 0, 0, 1])
        i = i+1

    # now we compare every emg data record with the average_length.
    # CASE1: if x < average_length, pad with zeros at each side.
    # CASE2: if x > average_length, then cut off the edges.
    i = 0
    while i < len(emg_data_list):
        print("blah", len(emg_data_list[i]))
        num_zeros_required = average_length - len(emg_data_list[i])
        if num_zeros_required > 0:
            zeros_each_side = int(num_zeros_required/2)
            for x in range(0, zeros_each_side):
                emg_data_list[i].append('0')
                emg_data_list[i].insert(0, '0')
        elif num_zeros_required < 0:
            cut_each_side = abs(int(num_zeros_required/2))
            cut_from = cut_each_side
            cut_to = len(emg_data_list[i]) - cut_each_side
            emg_data_list[i] = emg_data_list[i][cut_from:cut_to]
        i = i+1
    print(emg_data_list)
    return emg_data_list, expected_output_list


#==================================================================================
# get_input_gyro
#   this function takes in the paths of all the necessary .csv files and the number
#   of emg data columns. This function creates a data structure that can be used
#   as as input for our MLP.
#
#   The returning data structure looks like
#   [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
#   Where to_return[0][0] would get you the emg data for the input and to_return[0][1]
#   would get you the expected output for the corresponding emg data.
#==================================================================================
def get_input_gyro(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro ):
    to_return = []
    gyro_data_list = []
    expected_output_list = []

    # find the average length of emg time interval ( for generic case )
    average_length = 45
    print("AVERAGE LENGTH: ", average_length)
    accelero_forward_intervals = separateSets(path_forward_orientation, path_forward_accelero)
    accelero_backward_intervals = separateSets(path_backward_orientation, path_backward_accelero)
    accelero_left_intervals = separateSets(path_left_orientation, path_left_accelero)
    accelero_right_intervals = separateSets(path_right_orientation, path_right_accelero)
    accelero_enter_intervals = separateSets(path_enter_orientation, path_enter_accelero)

    forward_gyro_column = extract_gyro(accelero_forward_intervals, path_forward_gyro)
    backward_gyro_column = extract_gyro(accelero_backward_intervals, path_backward_gyro)
    left_gyro_column = extract_gyro(accelero_left_intervals, path_left_gyro)
    right_gyro_column = extract_gyro(accelero_right_intervals, path_right_gyro)
    enter_gyro_column = extract_gyro(accelero_enter_intervals, path_enter_gyro)

    # append all gyro data to the gyro_data_list in one BIG LIST
    # append the corresponding output expectation in an expected_output_list
    for i in range(0,len(forward_gyro_column)):
        gyro_data_list.append(forward_gyro_column[i])
        expected_output_list.append([1, 0, 0, 0, 0])
        gyro_data_list.append(backward_gyro_column[i])
        expected_output_list.append([0, 1, 0, 0, 0])
        gyro_data_list.append(left_gyro_column[i])
        expected_output_list.append([0, 0, 1, 0, 0])
        gyro_data_list.append(right_gyro_column[i])
        expected_output_list.append([0, 0, 0, 1, 0])
        gyro_data_list.append(enter_gyro_column[i])
        expected_output_list.append([0, 0, 0, 0, 1])

    # now we compare every gyro data record with the average_length.
    # CASE1: if x < average_length, pad with zeros at each side.
    # CASE2: if x > average_length, then cut off the edges.
    i = 0
    while i < len(gyro_data_list):
        num_zeros_required = average_length - len(gyro_data_list[i])
        if num_zeros_required > 0:
            zeros_each_side = int(num_zeros_required / 2)
            for x in range(0, zeros_each_side):
                gyro_data_list[i].append('0')
                gyro_data_list[i].insert(0, '0')
        elif num_zeros_required < 0:
            cut_each_side = abs(int(num_zeros_required / 2))
            cut_from = cut_each_side
            cut_to = len(gyro_data_list[i]) - cut_each_side
            gyro_data_list[i] = gyro_data_list[i][cut_from:cut_to]
        i = i + 1

    # Here, we finally create an ideal data structure

    # [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
    #print("hello",gyro_data_list)
    for x in range(0, len(gyro_data_list)):
        to_add = []
        to_add.append(list(map(float, gyro_data_list[x])))  # convert it to list of float data
        to_add.append(expected_output_list[x])
        to_return.append(to_add)
    return to_return



#==================================================================================
# get_input_accelero
#   this function takes in the paths of all the necessary .csv files and the number
#   of emg data columns. This function creates a data structure that can be used
#   as as input for our MLP.
#
#   The returning data structure looks like
#   [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
#   Where to_return[0][0] would get you the emg data for the input and to_return[0][1]
#   would get you the expected output for the corresponding emg data.
#==================================================================================
def get_input_accelero(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,):
    to_return = []
    emg_data_list = []
    expected_output_list = []

    average_length = 45

    print("AVERAGE LENGTH: ", average_length)
    accelero_forward_intervals = separateSets(path_forward_orientation, path_forward_accelero)
    accelero_backward_intervals = separateSets(path_backward_orientation, path_backward_accelero)
    accelero_left_intervals = separateSets(path_left_orientation, path_left_accelero)
    accelero_right_intervals = separateSets(path_right_orientation, path_right_accelero)
    accelero_enter_intervals = separateSets(path_enter_orientation, path_enter_accelero)


    forward_accelero_axis = extract_accelero(accelero_forward_intervals, path_forward_accelero, 'y')
    backward_accelero_axis = extract_accelero(accelero_backward_intervals, path_backward_accelero, 'y')
    left_accelero_axis = extract_accelero(accelero_left_intervals, path_left_accelero, 'y')
    right_accelero_axis = extract_accelero(accelero_right_intervals, path_right_accelero, 'y')
    enter_accelero_axis = extract_accelero(accelero_enter_intervals, path_enter_accelero, 'y')

    #print(forward_accelero_axis)
    for x in range(0, len(forward_accelero_axis)):
        emg_data_list.append(forward_accelero_axis[x])
        expected_output_list.append([1, 0, 0, 0, 0])
        emg_data_list.append(backward_accelero_axis[x])
        expected_output_list.append([0, 1, 0, 0, 0])
        emg_data_list.append(left_accelero_axis[x])
        expected_output_list.append([0, 0, 1, 0, 0])
        emg_data_list.append(right_accelero_axis[x])
        expected_output_list.append([0, 0, 0, 1, 0])
        emg_data_list.append(enter_accelero_axis[x])
        expected_output_list.append([0, 0, 0, 0, 1])

    i = 0
    while i < len(emg_data_list):
        num_zeros_required = average_length - len(emg_data_list[i])
        if num_zeros_required > 0:
            zeros_each_side = int(num_zeros_required / 2)
            for x in range(0, zeros_each_side):
                emg_data_list[i].append('0')
                emg_data_list[i].insert(0, '0')
        elif num_zeros_required < 0:
            cut_each_side = abs(int(num_zeros_required / 2))
            cut_from = cut_each_side
            cut_to = len(emg_data_list[i]) - cut_each_side
            emg_data_list[i] = emg_data_list[i][cut_from:cut_to]
        i = i + 1

    # Here, we finally create an ideal data structure
    # [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
    for y in range(0, len(emg_data_list)):
        to_add = []
        to_add.append(list(map(float, emg_data_list[y])))  # convert it to list of float data
        to_add.append(expected_output_list[y])
        to_return.append(to_add)
    print(to_return)
    return to_return


def extract_input_multisensor(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro,
                 num_emg_columns, path_forward_emg, path_backward_emg, path_left_emg, path_right_emg, path_enter_emg ):

    to_return = []

    gyro_intervals = get_input_gyro(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro )

    emg_intervals,output_data_list = get_input_x(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 num_emg_columns, path_forward_emg, path_backward_emg, path_left_emg, path_right_emg, path_enter_emg)
    emg_intervals = downscale(emg_intervals)
    temp = []
    i=0
    print(len(emg_intervals),len( gyro_intervals))
    for i in range (0, len(gyro_intervals)):
        temp = []
        temp+=gyro_intervals[i][0]
        temp+=emg_intervals[i]
        to_return.append(temp)
    result =[]

    print("bbbqqqq", len(to_return), len(output_data_list))
    for i in range (0, len(to_return)):
        temp = [to_return[i], output_data_list[i]]
        result.append(temp)

    return result

#===============================================================================
# downscale
#    this takes in list of accelero_intervals as a parameter and downscales it
#
#    Currently, this function should only downscale EMG data by skipping alternating
#    values
#===============================================================================
def downscale(data_list):
    to_return = []
    for i in range(0, len(data_list)): # should be really 10
        to_add = []
        for j in range(0, len(data_list[i])):
            if j % 2 == 0:
                to_add.append(data_list[i][j])
        to_return.append(to_add)
    return to_return



# get_input_x('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
#              './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
#              './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
#              './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
#              './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv', 4,
#              './data/Forward/emg-1456703940.csv', './data/Backward/emg-1456704054.csv', './data/Left/emg-1456704106.csv',
#              './data/Right/emg-1456704146.csv', './data/Enter/emg-1456704184.csv')

# get_input_accelero('./data/Forward/orientation-1456703940.csv', './data/Forward/accelerometer-1456703940.csv',
#              './data/Backward/orientation-1456704054.csv', './data/Backward/accelerometer-1456704054.csv',
#              './data/Left/orientation-1456704106.csv', './data/Left/accelerometer-1456704106.csv',
#              './data/Right/orientation-1456704146.csv', './data/Right/accelerometer-1456704146.csv',
#              './data/Enter/orientation-1456704184.csv', './data/Enter/accelerometer-1456704184.csv')