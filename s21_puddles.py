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

import spacy, os
os.system('python -m spacy download en_core_web_md')
import en_core_web_md
nlp = en_core_web_md.load()

import sklearn.metrics

def hello():
  return "Welcome to s21_puddles library"

###Week 3

def knn(*, table, target_list:list, differencer:str='euclidean') -> list:
  assert isinstance(table, pd.core.frame.DataFrame), f'table is not a dataframe but instead a {type(table)}'
  assert isinstance(target_list, list), f'target_list is not a list but a {type(target_list)}'
  assert len(target_list) == (len(table.loc[0].to_list())-1), f"Mismatching length for table and target_list: {len(target_list)} and {len(table.loc[0].to_list())-1}"
  assert all([not isinstance(x,str) for x in target_list]), f'target_list contains one or more string values'
  assert differencer in ['euclidean', 'reverse_cosine'], f"expecting one of {['euclidean', 'reverse_cosine']} for differencer but saw '{differencer}'."
  distance_record = []
  n = len(table)

  if differencer=='euclidean':
    for i in range(n):
      crowd_row = table.loc[i].to_list()
      crowd_numbers = crowd_row[:-1]
      choice = crowd_row[-1]
      d = euclidean_distance(target_list, crowd_numbers)
      distance_record += [[d,choice]]

    sorted_record = sorted(distance_record, reverse=False)  #ascending
    
  if differencer=='reverse_cosine':
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


def knn_accuracy(*, training_table, testing_table, k, differencer:str='euclidean'):
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table is not a dataframe but instead a {type(training_table)}'
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'testing_table is not a dataframe but instead a {type(testing_table)}'
  assert isinstance(k, int), f'k must be int but is instead a {type(k)}'
  assert k >= 1 and k <= len(training_table), f'k must be between 1 and {len(training_table)} but is {k}'
  assert differencer in ['euclidean', 'reverse_cosine'], f"expecting one of {['euclidean', 'reverse_cosine']} for differencer but saw '{differencer}'."
  
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
    result = knn(table=training_table, target_list=number_list, differencer=differencer)[:k]
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

  rev_record = [(act,pred) for pred,act in record]
  heat_map(rev_record, choices)

  return correct/n

#maybe use this instead of knn_accuracy
#allows you to demonstrate ROC by manipulating threshold
def knn_accuracy_threshold(*, training_table, testing_table, k, differencer:str='euclidean', threshold):
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table is not a dataframe but instead a {type(training_table)}'
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'testing_table is not a dataframe but instead a {type(testing_table)}'
  assert isinstance(k, int), f'k must be int but is instead a {type(k)}'
  assert k >= 1 and k <= len(training_table), f'k must be between 1 and {len(training_table)} but is {k}'
  assert differencer in ['euclidean', 'reverse_cosine'], f"expecting one of {['euclidean', 'reverse_cosine']} for differencer but saw '{differencer}'."
  
  training_choices = training_table[training_table.columns[-1]].unique().tolist()
  testing_choices = testing_table[testing_table.columns[-1]].unique().tolist()
  choices = [0,1] #list(set(training_choices + testing_choices))
  n = len(testing_table)
  record = []
  correct = 0
  for i in range(n):
    test_row = testing_table.loc[i].to_list()
    choice = test_row[-1]
    number_list = test_row[:-1]
    result = knn(table=training_table, target_list=number_list, differencer=differencer)[:k]
    votes = [c for d,c in result]
    vote_counts = []
    
    for c in choices:
      count = votes.count(c)
      vote_counts += [count]

    #pos_diff = vote_counts[1] - vote_counts[0]
    #winner = 1 if pos_diff >= threshold else 0
    winner = 1 if vote_counts[1] >= threshold else 0
    if winner == choice:
      correct += 1
    record += [(winner, choice)]
    
  #add this to  knn_accuracy
  rev_record = [(act,pred) for pred,act in record]
  heat_map(rev_record, choices)

  '''
  fig, ax = plt.subplots()
  plot_precision_recall([c for w,c in record], [w for w,c in record], ax=ax)
  '''
  accuracy = correct/n
  precision = sklearn.metrics.precision_score([c for w,c in record], [w for w,c in record])
  recall = sklearn.metrics.recall_score([c for w,c in record], [w for w,c in record])
  f1 = sklearn.metrics.f1_score([c for w,c in record], [w for w,c in record])
  print(f'Accuracy:\t{accuracy}')
  print(f'Precision:\t{precision}')
  print(f'Recall:\t\t{recall}')
  print(f'F1:\t\t{f1}')
  return record


def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"
  assert all([not isinstance(x,str) for x in vect1]), f'vect1 contains one or more string values'
  assert all([not isinstance(x,str) for x in vect2]), f'vect2 contains one or more string values'
  
  the_sum = sum([(p-q)**2 for p,q in zip(vect1,vect2)])
  return the_sum**.5  # I claim that this square root is not needed in K-means - see why?
  '''
  a = np.array(vect1, dtype='int64')
  b = np.array(vect2, dtype='int64')
  return norm(a-b)
  '''

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

def build_word_bag(*, sentence_list:list, outcome_list:list):
  assert isinstance(sentence_list, list)
  assert isinstance(outcome_list, list)
  try:
    type(nlp)==spacy.lang.en.English
  except:
    assert False, 'spacy must be loaded and nlp defined'

  outcome_unique = list(set(outcome_list))  #[0,1]
  as_ids = ['C_'+str(x) for x in outcome_unique]
  word_table = pd.DataFrame(columns=['word'] + as_ids)
  word_table = word_table.set_index('word')
  index_set = set()  #faster lookup than word_table.index.values

  for i,s in enumerate(sentence_list):
    outcome = 'C_'+str(outcome_list[i])
    doc = nlp(s)
    word_set = set()
    for token in doc:
      if not token.is_alpha: continue  #throwing out digits
      if token.is_stop: continue
      if token.is_punct: continue  #overkill with is_alpha
      if token.is_oov: continue
      real_word = token.text.lower()
      word_set.add(real_word)  #just keep unique values

    for word in list(word_set):
      if word not in index_set:
        word_table.loc[word] = [0]*(len(outcome_unique))
        index_set.add(word)
      word_table.loc[word, outcome] += 1

  return word_table.sort_index()

def ordered_by_column(*, word_table, column):
  assert isinstance(word_table, pd.core.frame.DataFrame), f'word_table is not a dataframe but instead a {type(table)}'
  assert column in ['C_0', 'C_1'], f'column must be C_0 or C_1 but is instead {column}'
  
  denom0 = sum(word_table['C_0'].to_list())
  denom1 = sum(word_table['C_1'].to_list())

  def tf_idf(*, word):
    assert word in word_table.index, f'unrecognized word: {word}. Check spelling and case.'
    
    tf_word_winning = word_table.loc[word, 'C_1']/denom1
    tf_word_losing = word_table.loc[word, 'C_0']/denom0

    df_word = min(1, word_table.loc[word, 'C_1']) + min(1, word_table.loc[word, 'C_0'])

    idf_word = math.log(2/df_word)

    tf_idf_word_losing = tf_word_losing * idf_word
    tf_idf_word_winning = tf_word_winning * idf_word

    return [tf_idf_word_losing, tf_idf_word_winning]

  values = []
  for word in word_table.index:
    result = tf_idf(word=word)
    values.append([word]+result)

  ordered = sorted(values, key=lambda triple: triple[1] if column=='C_0' else triple[2], reverse=True)
  return ordered

def add_tf_idf(*, word_table):
  assert isinstance(word_table, pd.core.frame.DataFrame), f'word_table is not a dataframe but instead a {type(word_table)}'
  outcomes = word_table.columns.to_list()
  new_table = word_table.copy(deep=True)
  for c in outcomes:
    new_table["tf_"+str(c)] = -1  #add new columns

  column_totals = []
  for column in outcomes:
    x = sum(word_table[column].to_list())
    column_totals += [x]

  index_list = word_table.index.values.tolist()
  for word in index_list:
    n = len(outcomes)
    non_zero_count = sum([word_table.loc[word,c]!=0 for c in outcomes])
    if not non_zero_count:
      assert False, f'{word} has all zero columns'
    idf = math.log10(n/non_zero_count)
    for i,column in enumerate(outcomes):
      c = word_table.loc[word,column]
      tf = c/column_totals[i]
      new_table.loc[word, "tf_"+str(column)] = tf * idf

  return new_table

#assumes word_list has been trimmed
def two_class_naive_bayes(*, word_list:list, word_bag, class0_count:int,
                          class1_count:int, laplace:float=1.0) -> list:
  assert isinstance(word_list, list), f'word_list not a list but instead a {type(word_list)}'
  assert all([isinstance(item, str) for item in word_list]), f'word_list must be list of strings but contains {type(word_list[0])}'
  assert isinstance(word_bag, pd.core.frame.DataFrame), f'word_bag not a dframe but instead a {type(word_bag)}'
  assert word_bag.index.name == 'word', f'word_bag must have index of "word"'
  assert 'C_0' in word_bag.columns.values, f'word_bag must have column C_0'
  assert 'C_1' in word_bag.columns.values, f'word_bag must have column C_1'

  evidence = list(set(word_list))  #remove duplicates
  index_list = word_bag.index.tolist()  #words

  total_samples = (class0_count+class1_count)
  class0_prob = class0_count/total_samples
  class1_prob = class1_count/total_samples

  #now have counts and probs for all classes

  results = []
  for c in ['C_0', 'C_1']:
    class_prob = class0_prob if c == 'C_0' else class1_prob
    prods = [math.log(class_prob)]  #P(O) in numerator to start with
    for ei in word_list:
      counts = class0_count if c == 'C_0' else class1_count
      if ei not in index_list:
        #did not see word in training set
        the_value =  laplace/(counts + laplace*len(word_bag)) 
      else:
        value = word_bag.loc[ei, c]
        the_value = ((value+laplace)/(counts + laplace*len(word_bag)))
      prods.append(math.log(the_value))
  
    results.append((c, sum(prods)))
  the_min = min(results, key=lambda pair: pair[1])[1]  #shift so smallest is 0
  return [[a,r+abs(the_min)]    for a,r in results]


#assumes word_list has been wrangled
def naive_bayes(word_list:list, word_bag, label_list:list, laplace:float=1.0) -> tuple:
  assert isinstance(word_list, list), f'word_list not a list but instead a {type(word_list)}'
  assert all([isinstance(item, str) for item in word_list]), f'word_list must be list of strings but contains {type(word_list[0])}'
  assert isinstance(word_bag, pd.core.frame.DataFrame), f'word_bag not a dframe but instead a {type(word_bag)}'
  assert word_bag.index.name == 'word', f'word_bag must have index of "word"'

  category_list = word_bag.columns.tolist()  #possible target values in list form
  assert all([label in category_list for label in label_list]), f'label_list must contain only values in {category_list}'
  
  evidence = list(set(word_list))  #remove duplicates
  index_list = word_bag.index.tolist()

  counts = []
  probs = []
  for category in category_list:
    ct = label_list.count(category)
    counts.append(ct)
    probs.append(ct/len(label_list))

  #now have counts and probs for all classes

  results = []
  for i, category in enumerate(category_list):
    prods = [math.log(probs[i])]  #P(author)
    for ei in word_list:
      if ei not in index_list:
        #did not see word in training set
        the_value =  1/(counts[i] + len(word_bag))
      else:
        value = word_bag.loc[ei, category]
        the_value = ((value+laplace)/(counts[i] + laplace*len(word_bag)))
      prods.append(math.log(the_value))
  
    results.append((category, sum(prods)))
  the_min = min(results, key=lambda pair: pair[1])[1]  #shift so smallest is 0
  return [[a,r+abs(the_min)]    for a,r in results]
