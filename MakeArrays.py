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

def make_grambank_dataframe(dataframe_given=False, df=None):
  data = readData('data.txt')
  languages = getUniqueLanguages(data)
  features = getUniqueFeatures(data)
  if not dataframe_given:
    df = createDataFrame(data)
    samples = len(languages)
  else:
    samples = len(df)
  df = df.replace('?', np.nan) 
  df = df.replace(np.nan, 0)
  array = df.to_numpy()
  missing_data_matrix = np.ones(array.shape)
  return array, missing_data_matrix, samples, len(features), df


def find_relatedness(index1, index2, languages_dataframe):
  lineage1 = languages_dataframe.lineage[index1]
  lineage2 = languages_dataframe.lineage[index2]
  if pd.isnull(lineage1) or pd.isnull(lineage2):
    return 100
  lineage1 = lineage1.split('/')
  lineage2 = lineage2.split('/')
  genera_in_common = list(set(lineage1).intersection(set(lineage2)))
  if genera_in_common == []:
    return 100
  last_genus_in_common = genera_in_common[-1]
  position1 = len(lineage1) - lineage1.index(last_genus_in_common)
  position2 = len(lineage2) - lineage2.index(last_genus_in_common)
  return max(position1, position2)
  
def find_distance(index1, index2, languages_dataframe):
  lat1 = languages_dataframe.latitude[index1]
  lat2 = languages_dataframe.latitude[index2]
  lon1 = languages_dataframe.longitude[index1]
  lon2 = languages_dataframe.longitude[index2]
  return haversine(lon1, lat1, lon2, lat2)

def make_relatedness_array(dataframe, languages_dataframe):
  result = []
  for index1 in dataframe.index:  
    temp = []
    for index2 in dataframe.index:
      print(index1)
      print(index2)
      if not index1 == index2:
        relatedness = find_relatedness(index1, index2, languages_dataframe)
        print(relatedness)
        temp.append(relatedness)
    result.append(temp)
  result = np.array(result)
  return result

def make_distance_array(dataframe, languages_dataframe):
  result = []
  for index1 in dataframe.index:  
    temp = []
    for index2 in dataframe.index:
      if not index1 == index2:
        distance = find_distance(index1, index2, languages_dataframe)
        temp.append(distance)
    result.append(temp)
  result = np.array(result)
  return result



  
  