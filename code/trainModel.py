
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
#import utilities
from OverlapSegmentationNet import OverlapSegmentationNet

import tensorflow as tf
import tensorflow as tf
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gist_earth'
import numpy as np


# In[ ]:

# Load data
xdata = np.load('xdata_88x88.npy')
labels = np.load('ydata_88x88_0123_onehot.npy')
train_test_boundary_index = round(13434*.8)


# In[ ]:

model = OverlapSegmentationNet(input_shape=(88,88,1))

# Choose loss
model.compile(loss='mean_squared_error', optimizer='adam')

# Specify the number of epochs to run
num_epoch = 5
for i range(num_epoch):
    
    # Fit
    model.fit(x=xdata, y=labels, epochs=1, validation_split=0.2) 
    filename = 'models/savedmodel_' + str(i) + 'epoch'
    model.save(filename)
    
    # Predict and plot images
    predictions = model.predict(xdata[0:4,...])
    utilities.plotSamplesOneHots(predictions[0:4,...].round())
   
    # Calculate mIOU
    y_pred_train = model.predict(xdata[0:train_test_boundary_index,...]).round()
    trainIOU = utilities.meanIU(y_pred_train, labels[0:train_test_boundary_index,...])
    print('Training meanIU: ' + str(trainIOU))
    trainAccuracy = utilities.globalAccuracy(y_pred_train, labels[0:train_test_boundary_index,...]
    print('Training accuracy: ' + str(trainAccuracy)))
    del y_pred_train
    
    y_pred_test = model.predict(xdata[train_test_boundary_index:,...]).round()
    testIOU = utilities.meanIU(y_pred_test, labels[train_test_boundary_index:,...])
    print('Testing meanIU: ' + str(testIOU))
    testAccuracy = utilities.globalAccuracy(y_pred_test, labels[train_test_boundary_index:,...])
    print('Testing accuracy: ' + str(testAccuracy))
    del y_pred_test

