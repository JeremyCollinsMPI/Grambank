from MakeArrays import *
from model import *






#   def __init__(self, output, missing_data_matrix, samples, types, classes, features)
  







def test1():
  array, missing_data_matrix , samples, features, input_array = make_simulated_array()
  model = Model1(array, missing_data_matrix, samples, 1, 7, features)
  model.train(steps=3000) 
  model.show_classes()


def test2():
  array, missing_data_matrix , samples, features, input_array = make_simulated_array()
  print(array)
  model = Model1(array, missing_data_matrix, samples, 1, 7, features)
  model.train_with_fixed_input(input_array=input_array, steps=3000) 
  model.show_filters()
  model.train_with_fixed_filters(steps = 3000)
  model.show_classes()

def test3():
  array, missing_data_matrix, samples, features, df = make_indo_european_array()
#   print(df)
  print(df.index)
  print(len(df.index))
  array2 = np.array(rep(1, 103))
  df2 = pd.DataFrame(array2, index = df.index)
  print(df2)
  clusters = 7
  types = 1
  model = Model1(array, missing_data_matrix, samples, 1, clusters, features)
  model.train(steps=4000) 
  print("******")
  classes = model.show_classes()
  classes = np.reshape(classes, (103,clusters))
  print(classes)
  df2 = pd.DataFrame(classes, index = df.index)
  df2 = pd.DataFrame.astype(df2, dtype = int)
  print(df2)
  df2.to_csv('ie_result.csv')

def test4():
  array, missing_data_matrix, samples, features, df = make_grambank_dataframe()
  features = 201
  print(df)
  clusters = 7
  types = 1
  print(df.index)
  
  model = Model1(array, missing_data_matrix, samples, types, clusters, features)
  model.train(steps=20000) 
  classes = model.show_classes()
  classes = np.reshape(classes, (samples,clusters))
  print(classes)
  df2 = pd.DataFrame(classes, index = df.index)
  df2 = pd.DataFrame.astype(df2, dtype = int)
  print(df2)
  df2.to_csv('grambank_result_1.csv')



test4()