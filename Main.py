from MakeArrays import *
from model import *
import os

glottocodes = pd.read_csv('languages.txt', header = 0, index_col=0)
languages_dataframe = glottocodes
geography = pd.read_csv('Languages.csv', header = 0, index_col=3)

def test1():
  if not 'relatedness_array.npy' in os.listdir('.'):
    relatedness_array = make_relatedness_array(dataframe, languages_dataframe)
    np.save('relatedness_array.npy', relatedness_array)
  else:
    relatedness_array = np.load('relatedness_array.npy')
  if not 'distance_array.npy' in os.listdir('.'):
     distance_array = make_distance_array(dataframe, geography)
     np.save('distance_array.npy', distance_array)
  else: 
    distance_array = np.load('distance_array.npy')
  output_array, missing_data_matrix, samples, features, df = make_grambank_dataframe(dataframe_given=True, df=dataframe)
  features = 201
  input_array = []
  for i in range(samples):
    thing = []
    for j in range(len(output_array)):
      if not i == j:
        thing.append(output_array[j])
    input_array.append(thing)
  input_array = np.array(input_array)
  model = Model(input_array, output_array, relatedness_array, distance_array, samples, features)
  model.train(steps=10000)
  relatedness_weightings, contact_weightings = model.show_weightings()
  print(relatedness_weightings)
  print(contact_weightings)
  
if __name__ == '__main__':
  if not 'dataframe.csv' in os.listdir('.'):
    array, missing_data_matrix, samples, features, df = make_grambank_dataframe()
    df.to_csv('dataframe.csv')
    dataframe = df
  else:
    dataframe = pd.read_csv('dataframe.csv', header = 0, index_col=0)
  for index in dataframe.index:
    if not index in geography.index:
      dataframe = dataframe.drop(index)
  samples = len(dataframe) - 1 
  test1()

