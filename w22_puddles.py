#imports

import numpy as np


from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from typing import TypeVar, Callable
dframe = TypeVar('pd.core.frame.DataFrame')
narray = TypeVar('numpy.ndarray')
import math
import json
import warnings
warnings.filterwarnings('ignore')


import sklearn.metrics

def hello():
  return "Welcome to w22_puddles library"


def heat_map(zipped, label_list):
  case_list = []
  for i in range(len(label_list)):
    inner_list = []
    for j in range(len(label_list)):
      inner_list.append(zipped.count((label_list[i], label_list[j])))
    case_list.append(inner_list)


  fig, ax = plt.subplots(figsize=(5, 5))
  ax.imshow(case_list)
  ax.grid(False)
  title = ''
  for i,c in enumerate(label_list):
    title += f'{i}={c} '
  ax.set_title(title)
  ax.set_xlabel('Predicted outputs', fontsize=16, color='black')
  ax.set_ylabel('Actual outputs', fontsize=16, color='black')
  ax.xaxis.set(ticks=range(len(label_list)))
  ax.yaxis.set(ticks=range(len(label_list)))
  
  for i in range(len(label_list)):
      for j in range(len(label_list)):
          ax.text(j, i, case_list[i][j], ha='center', va='center', color='white', fontsize=32)
  plt.show()
  return None

###########  week 5

def x_by_binary_y(*, table, x_column, y_column, bins=20):
  assert len(table[y_column].unique())==2, f'y_column must be binary'
  col_pos = [table.loc[i, x_column] for i in range(len(table)) if table.loc[i, y_column] == 1]
  col_neg = [table.loc[i, x_column] for i in range(len(table)) if table.loc[i, y_column] == 0]
  col_stacked = [col_pos, col_neg]
  plt.rcParams["figure.figsize"] = (15,8)  #15 by 8 inches
  bins = min(bins, int(table[x_column].max()))
  result = plt.hist(col_stacked, bins, stacked=True, label=[1, 0])
  plt.xlabel(x_column)
  plt.ylabel('Count')
  plt.title(f'{x_column} by {y_column}')
  plt.legend()
  plt.show()
 
###################### Naive Bayes

def wrangle_text(*, essay):
  assert isinstance(essay, str), f'essay must be string but is {type(essay)}'
  doc = nlp(essay)
  word_list = [w.text.lower() for w in doc if not w.is_stop and not w.is_oov and w.is_alpha]
  return word_list


#########  ANNs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import GridSearchCV
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#libraries to help visualize training results later
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
#%matplotlib inline
rcParams['figure.figsize'] = 10,8

#Used to show progress bar in loop
from IPython.display import HTML, display
import time
def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))


#from spring 20

def ann_build_model(n:int, layer_list: list, seed=1234, metrics='binary_accuracy'):
  assert isinstance(n, int), f'n is an int, the number of columns/features of each sample. Instead got {type(n)}'
  assert isinstance(layer_list, list) or isinstance(layer_list, tuple), f'layer_list is a list or tuple, the number of nodes per layer. Instead got {type(layer_list)}'

  if len(layer_list) == 1:
    print('Warning: layer_list has only 1 layer, the output layer. So no hidden layers')

  if layer_list[-1] != 1:
    print(f'Warning: layer_list has more than one node in the output layer: {layer_list[-1]}')

  np.random.seed(seed=seed)
  tf.random.set_seed(seed)

  model = Sequential()  #we will always use this in our class. It means left-to-right as we have diagrammed.
  model.add(Dense(units=layer_list[0], activation='relu', input_dim=n))  #first hidden layer needs number of inputs
  for u in layer_list[1:-1]:
    model.add(Dense(units=u, activation='relu'))
    
  #now output layer
  u = layer_list[-1:][0]
  model.add(Dense(units=u, activation='sigmoid'))
  loss_choice = 'binary_crossentropy'
  optimizer_choice = 'sgd'
  model.compile(loss=loss_choice,
              optimizer=optimizer_choice,
              metrics=[metrics])  #metrics is just to help us to see what is going on. kind of debugging info.
  return model

def ann_train(model, x_train:list, y_train:list, epochs:int,  batch_size=1):
  assert isinstance(x_train, list), f'x_train is a list, the list of samples. Instead got {type(x_train)}'
  assert isinstance(y_train, list), f'y_train is a list, the list of samples. Instead got {type(y_train)}'
  assert len(x_train) == len(y_train), f'x_train must be the same length as y_train'
  assert isinstance(epochs, int), f'epochs is an int, the number of epochs to repeat. Instead got {type(epochs)}'
  assert model.input_shape[1] == len(x_train[0]), f'model expecting sample size of {model.input_shape[1]} but saw {len(x_train[0])}'

  if epochs == 1:
    print('Warning: epochs is 1, typically too small.')
  print('Start training ...')
  xnp = np.array(x_train)
  ynp = np.array(y_train)
  training = model.fit(xnp, ynp, epochs=epochs, batch_size=batch_size, verbose=0)  #3 minutes
  
  plt.plot(training.history['binary_accuracy'])
  plt.plot(training.history['loss'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['binary accuracy', 'loss'], loc='upper left')
  plt.show()
  return training

#for grid search
def create_model(input_dim=300, lyrs=(64,32)):
    model = ann_build_model(input_dim, lyrs, metrics='accuracy')
    return model
  
def grid_search(layers_list, epochs_list, X_train, Y_train, indim=300):
  tup_layers = tuple([tuple(l) for l in layers_list])
  tup_epochs = tuple(epochs_list)
  
  model = KerasClassifier(build_fn=create_model, verbose=0)  #use our create_model
  
  # define the grid search parameters
  batch_size = [1]  #starting with just a few choices
  epochs = tup_epochs
  lyrs = tup_layers

  #use this to override our defaults. keys must match create_model args
  param_grid = dict(batch_size=batch_size, epochs=epochs, input_dim=[indim], lyrs=lyrs)

  # buld the search grid
  grid = GridSearchCV(estimator=model,   #we created model above
                      param_grid=param_grid,
                      cv=3,  #use 3 folds for cross-validation
                      verbose=2)  # include n_jobs=-1 if you are using CPU
  
  grid_result = grid.fit(np.array(X_train), np.array(Y_train))
  
  # summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
      



