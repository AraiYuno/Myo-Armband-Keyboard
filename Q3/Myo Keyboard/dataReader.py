import csv
from collections import defaultdict

# Global Variables
orientation_list = defaultdict(list)  #Stores the list of all the toCompare values in a list. Retrive with index.


#=================================================================================
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


def main():
    orientation_list = read_in_file('./data/Backward/orientation-1456704054.csv')
    print(orientation_list['toCompare'])


main()