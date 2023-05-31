import numpy as np

def read_file(filename, collapse=False):
    """
    Read data from a file and return it as a numpy array.

    Parameters:
    - filename (str): The name of the file to read.
    - collapse (bool, optional): Whether to collapse the data into sublists of length 1000. Default is False.

    Returns:
    - numpy.ndarray: The data read from the file as a numpy array.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            values = [float(val) for val in line.strip().split()]
            data.append(values)
        
        if collapse:
            collapsed_data = []
            x = 1000
            for i in range(0, len(data), x):
                collapsed_data.append(data[i:i+x])
            return collapsed_data
        
        return np.array(data)


def strip_time(data):
    """
    Strip the first element (time) from each sublist in the data.

    Parameters:
    - data (numpy.ndarray): The input data as a numpy array.

    Returns:
    - numpy.ndarray: The stripped data as a numpy array.
    """
    stripped_data = []
    for sublist in data:
        stripped_sublist = []
        for value in sublist:
            stripped_value = value[1:]
            stripped_sublist.append(stripped_value)
        stripped_data.append(stripped_sublist)
    return np.array(stripped_data)