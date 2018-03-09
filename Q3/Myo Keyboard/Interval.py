#=====================================================================================================#
#                                                                                                     #
# Interval.py                                                                                         #
#                                                                                                     #
# Author: Kyle                                                                                        #
# Purpose: This .py file contains a class called "Interval" which stores a start_time and end_time of #
#          an interval that can be used to separate sets of input                                     #
#=====================================================================================================#

class Interval:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def get_start_time(self):
        return self.start_time
