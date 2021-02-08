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

def hello():
  return "Welcome to s21_puddles library"

###Week 4

def check():
  return euclidean_distance([1,2],[3,4])

def knn(*, table, target_list:list, differencer:str='euclidean') -> list:
  assert isinstance(table, pd.core.frame.DataFrame), f'table is not a dataframe but instead a {type(table)}'
  assert isinstance(target_list, list), f'target_list is not a list but a {type(target_list)}'
  assert len(target_list) == (len(table.loc[0])-1), f"Mismatching length for table and target_list: {len(target_list)} and {len(table.loc[0])-1}"
  assert all([not isinstance(x,str) for x in target_list]), f'target_list contains one or more string values'
  assert differencer in ['euclidean', 'reverse_cosine'], f'expecting one of {['euclidean', 'reverse_cosine']} for differencer but saw {differencer}.'
  distance_record = []
  n = len(table)

  if differencer=='euclidean'
    for i in range(n):
      crowd_row = table.loc[i].to_list()
      crowd_numbers = crowd_row[:-1]
      choice = crowd_row[-1]
      d = euclidean_distance(target_list, crowd_numbers)
      distance_record += [[d,choice]]

    sorted_record = sorted(distance_record, reverse=False)  #ascending
    
  if differencer=='reverse_cosine'
      for i in range(n):
        crowd_row = table.loc[i].to_list()
        crowd_numbers = crowd_row[:-1]
        choice = crowd_row[-1]
        d = 1.0 - cosine_similarity(target_list, crowd_numbers)
        distance_record += [[d,choice]]
      sorted_record = sorted(distance_record, reverse=False)  #ascending
    
  return sorted_record

def get_knn_winner(expert_list):
  #[ [d,choice], ... ]
  counts = {}
  for d,c in expert_list:
    if c not in {}:
      counts[c] = 0
    counts[c] += 1
    
  m = max(counts.values())
  j = counts.values().find(m)
  return counts.keys()[j]


def knn_accuracy(training_table, testing_table, k):
  training_choices = training_table[training_table.columns[-1]].unique().tolist()
  testing_choices = testing_table[testing_table.columns[-1]].unique().tolist()
  choices = list(set(training_choices + testing_choices))
  n = len(testing_table)
  record = []
  correct = 0
  for i in range(n):
    test_row = testing_table.loc[i].to_list()
    choice = test_row[-1]
    number_list = test_row[:-1]
    result = knn(training_table, number_list, k)
    votes = [c for d,c in result]
    vote_counts = []
    
    for c in choices:
      count = votes.count(c)
      vote_counts += [count]

    m = max(vote_counts)
    j = vote_counts.index(m)
    winner = choices[j]
    if winner == choice:
      correct += 1
    record += [(winner, choice)]

  heat_map(record, choices)

  return correct/n

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"
  assert all([not isinstance(x,str) for x in vect1]), f'vect1 contains one or more string values'
  assert all([not isinstance(x,str) for x in vect2]), f'vect2 contains one or more string values'
  
  '''
  sum = 0
  for i in range(len(vect1)):
      sum += (vect1[i] - vect2[i])**2
      
  #could put assert here on result   
  return sum**.5  # I claim that this square root is not needed in K-means - see why?
  '''
  a = np.array(vect1, dtype='int64')
  b = np.array(vect2, dtype='int64')
  return norm(a-b)

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  '''
  sumxx, sumxy, sumyy = 0, 0, 0
  for i in range(len(vect1)):
      x = vect1[i]; y = vect2[i]
      sumxx += x*x
      sumyy += y*y
      sumxy += x*y
      denom = sumxx**.5 * sumyy**.5  #or (sumxx * sumyy)**.5
  #have to invert to order on smallest
  return sumxy/denom if denom > 0 else 0.0
  '''
  a = np.array(vect1)
  b = np.array(vect2)
  cosine = np.dot(a,b)/(norm(a)*norm(b))
  return cosine
