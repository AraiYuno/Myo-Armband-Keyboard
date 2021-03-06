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
                to_return.append(Interval(timestamp_list[i - 1], timestamp_list[i]))
                #to_return.append(Interval(accelero_timestamp[start_index-1], accelero_timestamp[start_index+1]))
                break
            start_index = start_index+1
        i = i+1
    return refineSets(to_return)

def centralise_intervals(intervals, average_interval_length, timestamp, data_list):
    to_return = []
    interval_range = int(average_interval_length/2)
    for i in range(0, len(intervals)):
        to_add = False
        temp_max = 0.0
        temp_max_index = 0
        for y in range(0, len(timestamp)):
            if float(intervals[i].start_time) == float(timestamp[y]):
                to_add = True
            if float(intervals[i].end_time) == float(timestamp[y]):
                to_add = False
            if to_add:
                # we find the maximum peak in each interval as well as its index ('y')
                if temp_max < float(data_list[y]):
                    temp_max = float(data_list[y])
                    temp_max_index = y
        to_return.append(Interval(timestamp[temp_max_index-interval_range], timestamp[temp_max_index+interval_range]))
    return to_return






#==================================================================================
# refineSets
#   refines the sets
#==================================================================================
def refineSets(intervals):
    while len(intervals) != 10:
        if len(intervals) > 10:
            intervals.pop(0)
        if len(intervals) < 10:
            intervals.append(intervals[len(intervals)-2])
    return intervals





#==================================================================================
# extract_gyro
#   returns the actual data sets of gyerometer with given 10 intervals
#   For example, emg_column1[0][0] contains the first interval's first gyrometer's
#   data record
#==================================================================================
def extract_gyro (intervals, gyro_path, axis):
    gyro = read_in_file(gyro_path)
    gyro_data = gyro[axis]
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



#===============================================================================
# calculate_average_interval_length
#    this function calculates the average length of a list of intervals by
#    counting the number of timestamps. Then, it returns the average length
#    of the list of intervals
#===============================================================================
def calculate_average_interval_length(intervals, timestamp):
    to_return = 0
    for i in range(0, len(intervals)):
        to_add = False
        for y in range(0, len(timestamp)):
            if float(intervals[i].start_time) == float(timestamp[y]):
                to_add = True
            if float(intervals[i].end_time) == float(timestamp[y]):
                to_add = False
                y -= 5
            if to_add:
                to_return += 1
    return to_return/len(intervals)



#============================================================================
# set_input_entry_length
#    takes the data_list and average_length. Then, it sets all the data entries
#    in the data_list to the length of average_length
#============================================================================
def set_input_entry_length(data_list, average_length):
    to_return = data_list[:]
    for i in range(0, len(to_return)):
        num_zeros_required = average_length - len(to_return[i])
        if num_zeros_required > 0:
            zeros_each_side = int(num_zeros_required / 2)
            for x in range(0, zeros_each_side):
                to_return[i].append('0')
                to_return[i].insert(0, '0')
        elif num_zeros_required < 0:
            cut_each_side = abs(int(num_zeros_required / 2))
            cut_from = cut_each_side
            cut_to = len(to_return[i]) - cut_each_side
            to_return[i] = to_return[i][cut_from:cut_to]
        if len(to_return[i]) == average_length + 1:
            to_return[i] = to_return[i][0:-2]
        if len(to_return[i]) < average_length:
            while len(to_return[i]) != average_length:
                to_return[i].append('0')
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
    emg_data_list = []
    expected_output_list = []

    # find the average length of emg time interval ( for generic case )
    average_length = 182

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

        downscale(forward_emg_column)
        # append all EMG data to the emg_data_list in one BIG LIST
        # append the corresponding output expectation in an expected_output_list
        for i in range(0, len(forward_emg_column)):
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
    #print(emg_data_list)
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
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro, axis):
    to_return = []
    gyro_data_list = []
    expected_output_list = []

    accelero_forward_timestamp = read_in_file(path_forward_accelero)['timestamp']
    accelero_backward_timestamp = read_in_file(path_backward_accelero)['timestamp']
    accelero_left_timestamp = read_in_file(path_left_accelero)['timestamp']
    accelero_right_timestamp = read_in_file(path_right_accelero)['timestamp']
    accelero_enter_timestamp = read_in_file(path_enter_accelero)['timestamp']

    gyro_forward = read_in_file(path_forward_gyro)[axis]
    gyro_backward = read_in_file(path_backward_gyro)[axis]
    gyro_left = read_in_file(path_left_gyro)[axis]
    gyro_right = read_in_file(path_right_gyro)[axis]
    gyro_enter = read_in_file(path_forward_gyro)[axis]


    # find the average length of emg time interval ( for generic case )
    accelero_forward_intervals = separateSets(path_forward_orientation, path_forward_accelero)
    accelero_backward_intervals = separateSets(path_backward_orientation, path_backward_accelero)
    accelero_left_intervals = separateSets(path_left_orientation, path_left_accelero)
    accelero_right_intervals = separateSets(path_right_orientation, path_right_accelero)
    accelero_enter_intervals = separateSets(path_enter_orientation, path_enter_accelero)

    average_length = int((calculate_average_interval_length(accelero_forward_intervals, accelero_forward_timestamp) +
                          calculate_average_interval_length(accelero_backward_intervals, accelero_backward_timestamp) +
                          calculate_average_interval_length(accelero_left_intervals, accelero_left_timestamp) +
                          calculate_average_interval_length(accelero_right_intervals, accelero_right_timestamp) +
                          calculate_average_interval_length(accelero_enter_intervals, accelero_enter_timestamp)) / 4)

    accelero_forward_intervals = centralise_intervals(accelero_forward_intervals, average_length, accelero_forward_timestamp, gyro_forward)
    accelero_backward_intervals = centralise_intervals(accelero_backward_intervals, average_length, accelero_backward_timestamp, gyro_backward)
    accelero_left_intervals = centralise_intervals(accelero_left_intervals, average_length, accelero_left_timestamp, gyro_left)
    accelero_right_intervals = centralise_intervals(accelero_right_intervals, average_length, accelero_right_timestamp, gyro_right)
    accelero_enter_intervals = centralise_intervals(accelero_enter_intervals, average_length, accelero_enter_timestamp, gyro_enter)

    #print("AVERAGE LENGTH: ", average_length)

    forward_gyro_column = extract_gyro(accelero_forward_intervals, path_forward_gyro, axis)
    backward_gyro_column = extract_gyro(accelero_backward_intervals, path_backward_gyro, axis)
    left_gyro_column = extract_gyro(accelero_left_intervals, path_left_gyro, axis)
    right_gyro_column = extract_gyro(accelero_right_intervals, path_right_gyro, axis)
    enter_gyro_column = extract_gyro(accelero_enter_intervals, path_enter_gyro, axis)

    # append all gyro data to the gyro_data_list in one BIG LIST
    # append the corresponding output expectation in an expected_output_list
    for i in range(0, len(forward_gyro_column)):
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

    gyro_data_list = set_input_entry_length(gyro_data_list, average_length)
    # Here, we finally create an ideal data structure
    # [ [[1,2,3], [1, 0, 0, 0, 0]], [[3,2,1], [0, 1, 0, 0, 0]], ... ]
    for x in range(0, len(gyro_data_list)):
        to_add = []
        to_add.append(list(map(float, gyro_data_list[x])))  # convert it to list of float data
        to_add.append(expected_output_list[x])
        to_return.append(to_add)
    return to_return, expected_output_list



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
                 path_enter_orientation, path_enter_accelero, axis):
    to_return = []
    emg_data_list = []
    expected_output_list = []

    accelero_forward_timestamp = read_in_file(path_forward_accelero)['timestamp']
    accelero_backward_timestamp = read_in_file(path_backward_accelero)['timestamp']
    accelero_left_timestamp = read_in_file(path_left_accelero)['timestamp']
    accelero_right_timestamp = read_in_file(path_right_accelero)['timestamp']
    accelero_enter_timestamp = read_in_file(path_enter_accelero)['timestamp']

    accelero_forward = read_in_file(path_forward_accelero)[axis]
    accelero_backward = read_in_file(path_backward_accelero)[axis]
    accelero_left = read_in_file(path_left_accelero)[axis]
    accelero_right = read_in_file(path_right_accelero)[axis]
    accelero_enter = read_in_file(path_enter_accelero)[axis]

    accelero_forward_intervals = separateSets(path_forward_orientation, path_forward_accelero)
    accelero_backward_intervals = separateSets(path_backward_orientation, path_backward_accelero)
    accelero_left_intervals = separateSets(path_left_orientation, path_left_accelero)
    accelero_right_intervals = separateSets(path_right_orientation, path_right_accelero)
    accelero_enter_intervals = separateSets(path_enter_orientation, path_enter_accelero)

    average_length = int((calculate_average_interval_length(accelero_forward_intervals, accelero_forward_timestamp) +
                          calculate_average_interval_length(accelero_backward_intervals, accelero_backward_timestamp) +
                          calculate_average_interval_length(accelero_left_intervals, accelero_left_timestamp) +
                          calculate_average_interval_length(accelero_right_intervals, accelero_right_timestamp) +
                          calculate_average_interval_length(accelero_enter_intervals, accelero_enter_timestamp)) / 4)

    accelero_forward_intervals = centralise_intervals(accelero_forward_intervals, average_length, accelero_forward_timestamp, accelero_forward)
    accelero_backward_intervals = centralise_intervals(accelero_backward_intervals, average_length, accelero_backward_timestamp, accelero_backward)
    accelero_left_intervals = centralise_intervals(accelero_left_intervals, average_length, accelero_left_timestamp, accelero_left)
    accelero_right_intervals = centralise_intervals(accelero_right_intervals, average_length, accelero_right_timestamp, accelero_right)
    accelero_enter_intervals = centralise_intervals(accelero_enter_intervals, average_length, accelero_enter_timestamp, accelero_enter)

    forward_accelero_axis = extract_accelero(accelero_forward_intervals, path_forward_accelero, 'y')
    backward_accelero_axis = extract_accelero(accelero_backward_intervals, path_backward_accelero, 'y')
    left_accelero_axis = extract_accelero(accelero_left_intervals, path_left_accelero, 'y')
    right_accelero_axis = extract_accelero(accelero_right_intervals, path_right_accelero, 'y')
    enter_accelero_axis = extract_accelero(accelero_enter_intervals, path_enter_accelero, 'y')

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
    return to_return, expected_output_list



def get_input_multiaxis_accelerometer(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero):

    to_return = []

    accelero_intervals_x, output_data = get_input_accelero(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 'x' )

    accelero_intervals_y, output_data = get_input_accelero(path_forward_orientation, path_forward_accelero,
                                      path_backward_orientation, path_backward_accelero,
                                      path_left_orientation, path_left_accelero,
                                      path_right_orientation, path_right_accelero,
                                      path_enter_orientation, path_enter_accelero,
                                       'y')

    accelero_intervals_z, output_data = get_input_accelero(path_forward_orientation, path_forward_accelero,
                                      path_backward_orientation, path_backward_accelero,
                                      path_left_orientation, path_left_accelero,
                                      path_right_orientation, path_right_accelero,
                                      path_enter_orientation, path_enter_accelero,
                                       'z')


    for i in range(0, len(accelero_intervals_y)):
        temp = []
        temp += accelero_intervals_x[i][0] + accelero_intervals_y[i][0]+ accelero_intervals_z[i][0]
        to_return.append(temp)
    result = []

    for i in range(0, len(to_return)):
        temp = [to_return[i], output_data[i]]
        result.append(temp)
    return result


def get_input_multiaxis_gyrometer(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro):

    to_return = []

    gyro_intervals_x, output_data = get_input_gyro(path_forward_orientation, path_forward_accelero,
                 path_backward_orientation, path_backward_accelero,
                 path_left_orientation, path_left_accelero,
                 path_right_orientation, path_right_accelero,
                 path_enter_orientation, path_enter_accelero,
                 path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro, path_enter_gyro,'x' )

    gyro_intervals_y, output_data = get_input_gyro(path_forward_orientation, path_forward_accelero,
                                      path_backward_orientation, path_backward_accelero,
                                      path_left_orientation, path_left_accelero,
                                      path_right_orientation, path_right_accelero,
                                      path_enter_orientation, path_enter_accelero,
                                      path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro,
                                      path_enter_gyro, 'y')

    gyro_intervals_z, output_data = get_input_gyro(path_forward_orientation, path_forward_accelero,
                                      path_backward_orientation, path_backward_accelero,
                                      path_left_orientation, path_left_accelero,
                                      path_right_orientation, path_right_accelero,
                                      path_enter_orientation, path_enter_accelero,
                                      path_forward_gyro, path_backward_gyro, path_left_gyro, path_right_gyro,
                                      path_enter_gyro, 'z')


    for i in range(0, len(gyro_intervals_y)):
        temp = []
        temp += gyro_intervals_x[i][0] + gyro_intervals_y[i][0]+ gyro_intervals_z[i][0]# +emg_intervals[i]
        to_return.append(temp)
    result = []

    for i in range(0, len(to_return)):
        temp = [to_return[i], output_data[i]]
        result.append(temp)
    return result


