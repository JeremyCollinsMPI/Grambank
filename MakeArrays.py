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
  languages = getUniqueLanguages(data)
  features = getUniqueFeatures(data)

  print(len(features))
  print(len(languages))

  df = createDataFrame(data)
  df = df.replace('?', np.nan) 
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
  df = df.replace(np.nan, 0)
  array = df.to_numpy()
  array = np.ndarray.astype(array, dtype=np.float32)
  missing_data_matrix = np.ones(array.shape, dtype=np.float32)
  number_of_clusters = 7
  features = 6280
  samples = 103
  return array, missing_data_matrix, samples, features, df





