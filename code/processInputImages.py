import matplotlib.pyplot as plt
import numpy as np
import utilities
import h5py

# Load HD5F file
h5f = h5py.File('LowRes_13434_overlapping_pairs.h5','r')
labels = h5f['dataset_1'][...,1]
h5f.close()

# Clean labels
labels = utilities.cleanLabelNearestNeighbour_alllabels(labels)

# Crop to 88x88 pixels and save processed numpy arrays
labels = utilities.makeXbyY(labels, 88, 88)
np.save('ydata_88x88_0123_onehot', labels)
xdata = utilities.makeXbyY(xdata, 88, 88).reshape((13434,88,88, 1))
np.save('xdata_88x88', xdata)

