#!/usr/bin/env python

# ---------------------------------------------------------------
#This reducer code will input a line of text and
#    output <word, total-count>
# ---------------------------------------------------------------
import sys

last_key = None              #initialize these variables
running_total = 0

# -----------------------------------
# Loop thru file
#  --------------------------------
for input_line in sys.stdin:
    input_line = input_line.strip()

    # --------------------------------
    # Get Next Word    # --------------------------------
    this_key, value = input_line.split("\t", 1)  #the Hadoop default is tab separates key value
                          #the split command returns a list of strings, in this case into 2 variables
    if value.isdigit():
        value = int(value)
        if last_key == this_key:
            running_total += value   # add value to running total
        else:
            running_total = value    #reset values
            last_key = this_key
    else:
        print("{0}\t{1}".format(this_key, running_total))
        running_total = 0    #reset values
        last_key = this_key

if last_key == this_key:
    print("{0}\t{1}".format(last_key, running_total))

