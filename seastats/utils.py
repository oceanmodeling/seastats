import pandas as pd
import numpy as np


def json_format(d):
    for key, value in d.items():
        if isinstance(value, (dict, list, tuple)):
            json_format(value)  # Recurse into nested dictionaries
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()  # Convert NumPy array to list
        elif isinstance(value, pd.Timestamp):
            d[key] = value.strftime("%Y-%m-%d %H:%M:%S")  # Convert pandas Timestamp to string
        elif isinstance(value, pd.Timedelta):
            d[key] = str(value)  # Convert pandas Timedelta to string
        else: 
            d[key] = str(value)
    return d
