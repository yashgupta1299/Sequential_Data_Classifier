#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
# current processing
tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'))

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
import librosa
import librosa.display

import os
import time
import itertools
import shutil
import cv2
import zipfile

from sklearn.preprocessing import LabelEncoder
from IPython import get_ipython
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


# In[2]:


name = 'task_2.ipynb'

path = ''
# from google.colab import drive
# drive.mount('/content/drive')
# path = '/content/drive/MyDrive/deepLearningAs3/'

random_state_global = 42

pathfinal = path + 'model_history/part_b/'
pathfinal2 = path + 'model_images/'
weightPath = path + 'model_history/part_b/weights/'

epoch_val = 3000
batch_size_val = 32
threshold_val = 1e-4

checkpoint_path = pathfinal+'weights/checkpoint'


# In[3]:


def unzip_data(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
  
unzip_data(f"{path}CV_Data.zip")


# In[4]:


def delete_folder_contents(path_erase):
    '''
    take path to erase all data present on it
    '''
    folder_name = path_erase
    # Get all files in the folder
    files = os.listdir(folder_name)

    # Loop through the files and delete them
    for file in files:
        if(file == 'weights'):
            continue
        file_path = os.path.join(folder_name, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
# delete_folder_contents(pathfinal)
delete_folder_contents(pathfinal2)


# ### Data Loading

# In[5]:


class_names = ['ba', 'ne', 'ni','paa', 're']
train_X = []
len_X = [] # later sequence changed
train_Y = []

test_X = []
test_Y = []
len_Y = [] # later sequence changed

l0 = path+'CV_Data'
totalFiles=0
train_files = 0
test_files = 0
for l1 in os.listdir(l0):
    if(l1 in class_names):
        f1 = os.path.join(l0, l1)
        for l2 in os.listdir(f1):
            if(l2=='.DS_Store'):
                continue
            f2 = os.path.join(f1, l2)
            flag = False
            for l3 in os.listdir(f2):
                if(l3=='.DS_Store'):
                    continue
                if(flag==False):
                    if(l2 == 'Train'):
                        train_files+=len(os.listdir(f2))
                    else:
                        test_files+=len(os.listdir(f2))
                    totalFiles+=len(os.listdir(f2))
                    print(f'{l1} length: {len(os.listdir(f2))}, {l2}')
                flag=True
                f3 = os.path.join(f2, l3)
                with open(f3, 'r') as f:
                    lines = f.readlines()
                    mfcc_data = [list(map(float, line.strip().split(' '))) for line in lines]
                    length = len(mfcc_data)
                    if(l2 == 'Train'):
                        train_X.append(mfcc_data)
                        train_Y.append(l1)
                        len_X.append(length)
                    else:
                        test_X.append(mfcc_data)
                        test_Y.append(l1)
                        len_Y.append(length)

print(f'Total files: {totalFiles}, Training Files {train_files}, Testing Files {test_files}')

def suffleData(list1, list2, random_state_global):
    np.random.seed(random_state_global)
    # Combine the lists using zip()
    combined = list(zip(list1, list2))
    # Shuffle the combined list
    np.random.shuffle(combined)

    # Unzip the shuffled list
    shuffled_list1, shuffled_list2 = zip(*combined)
    return shuffled_list1, shuffled_list2

train_X, train_Y = suffleData(train_X, train_Y, random_state_global)
test_X, test_Y = suffleData(test_X, test_Y, random_state_global)

train_M_Y = LabelEncoder().fit_transform(train_Y)
test_M_Y = LabelEncoder().fit_transform(test_Y)
# maximum_sequence_length = max(np.max(len_X), np.max(len_Y))
print(class_names)
class_ind = LabelEncoder().fit_transform(class_names)
print(class_ind)
print(f'Sequence length range {min(np.min(len_X), np.min(len_Y))} to {max(np.max(len_X), np.max(len_Y))}')


# ### Frequency of seq length

# In[6]:


maximum_sequence_length = 50
mask_value = 0.0 ## because dataset do not have zero value
fig,ax = plt.subplots(1,2, figsize=(10, 5))
ax = ax.reshape(-1)
for i in range(2):
    ax[i].hist(len_X if i==0 else len_Y)
    ax[i].set_title('Train' if i==0 else 'Test')
    ax[i].set_xlabel('Length of (2d) Points')
    ax[i].set_ylabel('Frequency')


# ### Visualizing Data

# In[7]:


def plot_data(data_X, data_Y):
    sorted_lists = sorted(zip(data_Y, data_X))
    data_Y, data_X = zip(*sorted_lists)
    fig, ax = plt.subplots(5,1, figsize=(7, 12))
    ax = ax.reshape(-1)
    c=0
    cl=0
    mid=2
    colors = ['r', 'g', 'b', 'y', 'm']
    for i, ele in enumerate(data_Y):
        if(ele==cl):
            im = librosa.display.specshow(np.array(data_X[i]).T, x_axis='time',ax=ax[c])
            fig.colorbar(im, ax=ax[c])
            ax[c].set_title(f'{class_names[cl]}')
            ax[c].set_ylabel('MFCC Coefficients')
            ax[c].set_xlabel('Time')
            c+=1
            cl+=1
            if(c==5):
                break
    plt.tight_layout()
    plt.show()         

plot_data(train_X, train_M_Y)


# ### Functions

# In[8]:


# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=10): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_Y, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
    

def inferences(df_model_history, model, data_X, data_Y):
    print(f'Training Accuracy for model: {df_model_history["accuracy"].to_list()[-1]*100:.2f}%')
    print(f'Validation Accuracy for model: {df_model_history["val_accuracy"].to_list()[-1]*100:.2f}%')
    print(f'Test Accuracy for model: {model.evaluate(data_X, data_Y, verbose=0)[1]*100:.2f}%')

    df_model_history.plot(title="Accuracy / Loss vs Epoch", xlabel='Epoch', ylabel='Accuracy / Loss')
    plt.show()
    
    df_model_history['loss'].plot(title="Average training error vs epochs", xlabel='Epoch', ylabel='Loss')
    plt.show()

def makingPredictionWithCM(model, data_X, data_Y, class_names):
    y_prob_a = model.predict(data_X, verbose=0)
    y_pred_a = y_prob_a.argmax(axis=1)
    make_confusion_matrix(data_Y, y_pred_a, class_names)

model_arch_list = []
def plottingModel(model):
    plot_model(model,to_file=f'model_images/model.png', show_shapes=True, show_layer_activations=True, expand_nested=True)
    img = plt.imread(f'model_images/model.png')
    model_arch_list.append(img)
    plt.figure(figsize=(5, 10))
    plt.imshow(img)
    plt.axis(False)
    plt.show()

def showResults(model, history, data_X, data_Y, class_names):
    inferences(history, model, data_X, data_Y)
    makingPredictionWithCM(model, data_X, data_Y, class_names)
    # plottingModel(model)


# ### Padding Sequence

# In[9]:


def paddingSequence(data_X, maxlen=maximum_sequence_length):
    data_X = keras.preprocessing.sequence.pad_sequences(data_X, maxlen=maximum_sequence_length, padding='post', value=mask_value, dtype='float32')
    return data_X

train_M_X = paddingSequence(train_X)
test_M_X = paddingSequence(test_X)
np.min(train_M_X), np.max(train_M_X), np.min(test_M_X), np.max(test_M_X)


# ### Callbacks

# In[10]:


class ModelSaving(keras.callbacks.Callback):
    def __init__(self):
        self.currentEpoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.currentEpoch = epoch

    def on_train_end(self, logs=None):
        self.model.save(f'{pathfinal}{self.model.name}_{self.currentEpoch+1}.tf')
        # print("Training has ended!, model saved")

    
class HistorySaver(keras.callbacks.Callback):
    def __init__(self, initial_history):
        super(HistorySaver, self).__init__()
        self.history = {}
        self.currentEpoch = 0
        
        for key, value in [('loss', initial_history[0]), ('accuracy', initial_history[1]), ('val_loss', initial_history[2]), ('val_accuracy', initial_history[3])]:
            self.history.setdefault(key, []).append(value)
        
        # logs.items() = dict_items([('loss', 1.3612865209579468), ('accuracy', 0.46034255623817444), ('val_loss', 1.1157031059265137), ('val_accuracy', 0.6484848856925964)])
    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)
        self.currentEpoch = epoch
        
    def on_train_end(self, logs=None):
        pd.DataFrame(self.history).to_csv(f'{pathfinal}{self.model.name}_{self.currentEpoch+1}.csv', index=False)
        # print("Training has ended!, model history saved")



# create the callbacks

model_saver = ModelSaving()

# This means if for 5 epochs the accuracy has no progress on 
# the validation set then it would stop and store the previous best value.
early_stopping_cb = keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=2,
                                                  min_delta=threshold_val,
                                                  mode='min',
                                                  verbose=1)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_path,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)


# ### Building a RNN,LSTM Model

# In[11]:


train_M_X.shape, test_M_X.shape


# In[12]:


tf.random.set_seed(42)
input_shape = (train_M_X.shape[1], train_M_X.shape[2])
model_1 = Sequential()
model_1.add(layers.Input(shape=input_shape))
model_1.add(layers.Masking(mask_value=mask_value))
model_1.add(layers.LSTM(64, return_sequences=True))
model_1.add(layers.LSTM(32, return_sequences=True))
model_1.add(layers.LSTM(16))
model_1.add(layers.Dense(5, activation='softmax'))

# # Train the RNN
model_1.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model_1.summary()


# In[13]:


# # Evaluate the model_1 initial losses
# initial_train_loss, initial_train_acc = model_1.evaluate(train_M_X, train_M_Y, verbose=0)
# initial_valid_loss, initial_valid_acc = model_1.evaluate(test_M_X, test_M_Y, verbose=0)

# history_1 = model_1.fit(train_M_X, train_M_Y, 
#                 validation_data=(test_M_X, test_M_Y),
#                 callbacks=[HistorySaver((initial_train_loss, initial_train_acc, initial_valid_loss, initial_valid_acc)), 
#                                 checkpoint_callback,
#                                 early_stopping_cb],
#                 batch_size=batch_size_val, epochs=epoch_val, verbose=1)


# In[14]:


model_1.load_weights(checkpoint_path)
df_history_1 = pd.read_csv(f'{pathfinal}sequential_3_19.csv')
# # df_history_1 = pd.DataFrame(history_1.history)
showResults(model_1, df_history_1, test_M_X, test_M_Y, class_names)
plot_model(model_1,to_file=f'{path}model_images/model.png', show_shapes=True, show_layer_activations=True, expand_nested=True, dpi=999)


# In[15]:


delete_folder_contents(pathfinal2)
get_ipython().system('osascript -e \'tell application "System Events" to keystroke "s" using command down\'')
get_ipython().system(f'jupyter nbconvert {name} --to python')

