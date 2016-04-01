#!/usr/bin/env python
import sys


for line in sys.stdin:

    line = line.strip()   #strip out carriage return
    key_value = line.split(",")   #split line, into key and value, returns a list
    if key_value[1].isdigit():
        key_in = key_value[0]   #key is first item in list
        value_in = key_value[1]   #value is 2nd item
        print('%s\t%s' % (key_in, value_in))
    elif key_value[1][0:3] == "ABC":
        key_in = key_value[0]   #key is first item in list
        value_in = key_value[1]   #value is 2nd item
        print('%s\t%s' % (key_in, value_in))  #print a string tab and string

#Note that Hadoop expects a tab to separate key value
#but this program assumes the input file has a ',' separating key value
