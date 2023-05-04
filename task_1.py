#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
# current processing
# tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

from importlib import import_module
import keras
from keras.api._v2 import keras as KerasAPI
keras: KerasAPI = import_module("tensorflow.keras")
print(tf.__version__)

from keras import Model, layers
from keras.models import Sequential
from keras.layers import preprocessing
from keras.utils import plot_model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

import os
import time
import itertools
import shutil
import cv2
import zipfile

from IPython import get_ipython
from sklearn.metrics import confusion_matrix


# In[8]:


name = 'task_1.ipynb'


# In[9]:


# symbol, telugu, english
# ఐ, Ai, I 
# చ, chA, ch
# డా, dA, Dr
# ల, lA, The
# త, tA, Th

# అ, a, That
# బ, bA, b


# In[10]:


get_ipython().system('osascript -e \'tell application "System Events" to keystroke "s" using command down\'')
get_ipython().system(f'jupyter nbconvert {name} --to python')

