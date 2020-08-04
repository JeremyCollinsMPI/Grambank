from CreateDataFrame import *
import numpy as np
from numpy import random
from nexus import NexusReader


def rep(x,y):
	new=[]
	for m in range(y):
		new.append(x)
	return new	        


np.random.seed(10)

def make_grambank_dataframe():
  data = readData('data.txt')
  # dict = createDictionary(df)
  languages = getUniqueLanguages(data)
  features = getUniqueFeatures(data)

  print(len(features))
  print(len(languages))

  df = createDataFrame(data)
  df = df.replace('?', np.nan) 
#   cheating at the moment by replacing '?' with 0
  df = df.replace(np.nan, 0)
  

  array = df.to_numpy()
  missing_data_matrix = np.ones(array.shape)

  print(array[0])
  return array, missing_data_matrix, len(languages), len(features), df

def make_simulated_array():
  '''
  structure should be (languages, samples)
  you have seven clusters
  e.g. ten features, 70 languages
  cluster1 = [1,1,1,0,0,0,0,0,1,1]
  cluster2 = [0,0,1,0,0,0,0,0,1,1]
  '''
  clusters = []
  number_of_clusters = 7
  features = 10
  languages_per_cluster = 10
  for i in range(number_of_clusters):
    clusters.append(random.randint(2,size =features))
  result = []
  for i in range(number_of_clusters):
    for j in range(languages_per_cluster):
      result.append(clusters[i])
  result = np.array(result)
  missing_data_matrix = np.ones(result.shape)
  input_array = []
  for i in range(number_of_clusters):
    to_append = rep(0, number_of_clusters)
    to_append[i] = 1
    for j in range(languages_per_cluster):
      input_array.append([to_append])      
  return result, missing_data_matrix, number_of_clusters*languages_per_cluster, features, input_array

def make_indo_european_array():
  n = NexusReader.from_file('IELex_Bouckaert2012.nex')
  df = pd.DataFrame.from_dict(n.data.matrix, orient='index')
  df = df.replace('?', np.nan) 
#   cheating at the moment by replacing '?' with 0
  df = df.replace(np.nan, 0)
  array = df.to_numpy()
  array = np.ndarray.astype(array, dtype=np.float32)
  missing_data_matrix = np.ones(array.shape, dtype=np.float32)
  number_of_clusters = 7
  features = 6280
  samples = 103
  return array, missing_data_matrix, samples, features, df





# def find_next_pair(index1, index2, last_index):
#   if index2 < last_index:
#     index2 = index2 + 1
#     return index1, index2
#   else:
#     if index1 < last_index - 1:
#       index1 = index1 + 1
#       index2 = index1 + 1
#       return index1, index2
#     else:
#       return None, None
# 
# def find_nulls(array1, array2):
#   result = array1 + array2
#   result = np.invert(np.isnan(result))
#   return result.astype(int)
# 
# def make_arrays():
#   input_array = []
#   output_array = []
#   null_layer = []
#   last_index = len(languages) - 1
#   index1 = 0
#   index2 = 1
#   end = False
#   while not end:
#     values1 = dict[languages[index1]]['values']
#     values2 = dict[languages[index2]]['values']
#     nulls = find_nulls(values1, values2)
#     input_array.append(values1)
#     output_array.append(values2)
#     null_layer.append(nulls)
#     index1, index2 = find_next_pair(index1, index2, last_index)
#     if index1 == None and index2 == None:
#       end = True
#       input_array = np.array(input_array)
#       output_array = np.array(output_array)
#       null_layer = np.array(null_layer)
#       input_array[np.isnan(input_array)] = 0
#       output_array[np.isnan(output_array)] = 0
#       return input_array, output_array, null_layer
# 
# def pickle_arrays():
#   input_array, output_array, null_layer = make_arrays()
#   np.save('input_array.npy', input_array)
#   np.save('output_array.npy', output_array)
#   np.save('null_layer.npy', null_layer)

# pickle_arrays()

# print(input_array[0])
# print(output_array[0])
# print(null_layer[0])
# print(np.shape(input_array))
# print(np.shape(output_array))
# print(np.shape(null_layer))


  
  