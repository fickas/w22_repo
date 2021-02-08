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

def knn(table, target_list):

  distance_record = []
  n = len(table)

  for i in range(n):
    crowd_row = table.loc[i].to_list()
    crowd_numbers = crowd_row[:-1]
    choice = crowd_row[-1]
    d = euclidean_distance(target_list, crowd_numbers)
    distance_record += [[d,choice]]

  sorted_record = sorted(distance_record)
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

