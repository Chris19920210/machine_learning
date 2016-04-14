import pandas as pd
import numpy as np
import collections
import json

with open('/home/chris/json_file/json_example') as f:
    data = json.load(f)
# flatten json


def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '_')
        elif isinstance(x, list):
            for i, value in enumerate(x):
                flatten(value, name + str(i) + '_')
        else:
            out[str(name[:-1])] = str(x)

    flatten(y)
    return out



# flatten nested dictionary
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

print flatten_json(data[0])







