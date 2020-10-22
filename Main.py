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
  print(relatedness_array.shape)
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
  model = Model1(input_array, output_array, relatedness_array, samples, features)
  model.train(steps=10000)
  weighting = model.show_weighting()
  print(weighting)

def test2():
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
  model = Model5(input_array, output_array, relatedness_array, distance_array, samples, features)
  model.train(steps=10000)


def test3():
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
  def run():
    model = Model5(input_array, output_array, relatedness_array, distance_array, samples, features)
    model.train(steps=3000)
    return model
  if not 'relatedness_weights.npy' in os.listdir('.'):
    model = run()
    relatedness_weights, contact_weights = model.get_weightings()
    universal_bias = model.get_universal_bias()
    np.save('relatedness_weights.npy', relatedness_weights)
    np.save('contact_weights.npy', contact_weights)
    np.save('universal_bias_2.npy', universal_bias)
  else:
    relatedness_weights = np.load('relatedness_weights.npy')
    contact_weights = np.load('contact_weights.npy')
    universal_bias = np.load('universal_bias_2.npy')
  def find_row_number(index):
    return np.where(df.index==index)[0][0]
  samples = 1
  dict = {'English': 'stan1293', 'Dutch': 'dutc1256', 'Swedish': 'swed1254', 'Mandarin': 'mand1415', 
  'Cantonese': 'yuec1235', 'Japanese': 'nucl1643', 'Korean': 'kore1280',
  'Thai': 'thai1261', 'Lao': 'laoo1244', 'Paiwan': 'paiw1248', 'Maori': 'maor1246', 'Mongolian': 'peri1253'}
  comparandum1_index = dict['Japanese']
  comparandum2_index = dict['Korean']
  comparandum1_row_number = find_row_number(comparandum1_index)
  comparandum2_row_number = find_row_number(comparandum2_index)
  comparandum1 = np.array([output_array[comparandum1_row_number]], np.float64)
  comparandum2 = np.array([output_array[comparandum2_row_number]], np.float64)
  print(comparandum1)
  print(comparandum2)
  model = Model7(samples, features, relatedness_weights, contact_weights, universal_bias)
  model.infer(comparandum1, comparandum2)

def test4():
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
  def run():
    model = Model8(input_array, output_array, relatedness_array, distance_array, samples, features)
    model.train(steps=2000)
    return model
  if not 'relatedness_a.npy' in os.listdir('.'):
    model = run()
    relatedness_a, relatedness_b, relatedness_c, contact_a, contact_b, contact_c = model.get_weightings()
    universal_bias = model.get_universal_bias()
    np.save('relatedness_a.npy', relatedness_a)
    np.save('relatedness_b.npy', relatedness_b)
    np.save('relatedness_c.npy', relatedness_c)
    np.save('contact_a.npy', contact_a)
    np.save('contact_b.npy', contact_b)
    np.save('contact_c.npy', contact_c)
    np.save('universal_bias.npy', universal_bias)
  else:
    relatedness_a = np.load('relatedness_a.npy')
    relatedness_b = np.load('relatedness_b.npy')
    relatedness_c = np.load('relatedness_c.npy')
    contact_a = np.load('contact_a.npy')
    contact_b = np.load('contact_b.npy')
    contact_c = np.load('contact_c.npy')
    universal_bias = np.load('universal_bias.npy')
  def find_row_number(index):
    return np.where(df.index==index)[0][0]
  samples = 1
  model = Model9(input_array, output_array, relatedness_array, distance_array, samples, features, relatedness_a, relatedness_b, relatedness_c, contact_a, contact_b, contact_c, universal_bias)
  dict = {'English': 'stan1293', 'Dutch': 'dutc1256', 'Swedish': 'swed1254', 'Mandarin': 'mand1415', 
  'Cantonese': 'yuec1235', 'Japanese': 'nucl1643', 'Korean': 'kore1280',
  'Thai': 'thai1261', 'Lao': 'laoo1244', 'Paiwan': 'paiw1248', 'Maori': 'maor1246', 'Mongolian': 'peri1253'}
  comparandum1_index = dict['Korean']
  comparandum2_index = dict['Mandarin']
  comparandum1_row_number = find_row_number(comparandum1_index)
  comparandum2_row_number = find_row_number(comparandum2_index)
  comparandum1 = np.array([output_array[comparandum1_row_number]], np.float64)
  comparandum2 = np.array([output_array[comparandum2_row_number]], np.float64)
  print(comparandum1)
  print(comparandum2)
  model.infer(comparandum1, comparandum2, steps=2000)

def test5():
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
  def run():
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None)
    model.train(steps=3000)
    return model
  if not 'weights.npy' in os.listdir('.'):
    model = run()
    weights = model.sess.run(model.weights)
    intercept = model.sess.run(model.intercept)
    np.save('weights.npy', weights)
    np.save('intercept.npy', intercept)
  else:
    weights = np.load('weights.npy')
    intercept = np.load('intercept.npy')
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=True, weights=weights, intercept=intercept)
  def find_row_number(index):
    return np.where(df.index==index)[0][0]
  dict = {'English': 'stan1293', 'Dutch': 'dutc1256', 'Swedish': 'swed1254', 'Mandarin': 'mand1415', 
  'Cantonese': 'yuec1235', 'Japanese': 'nucl1643', 'Korean': 'kore1280',
  'Thai': 'thai1261', 'Lao': 'laoo1244', 'Paiwan': 'paiw1248', 'Maori': 'maor1246', 'Mongolian': 'peri1253',
  'Koyr1': 'koyr1240', 'Koyr2': 'koyr1242', 'Tundra Nenets': 'nene1249', 'Finnish': 'finn1318', 'Veps': 'veps1250',
  'Tigon Mbembe': 'tigo1236', 'Congo Swahili': 'cong1236', 'Tahaggart Tamahaq': 'taha1241', 'Modern Hebrew': 'hebr1245',
  'Gulf Arabic': 'gulf1241', 'Narungga': 'naru1238', 'Yinggarda': 'ying1247', 'Thurawal': 'thur1254', 'Plateau Malagasy': 'plat1254',
  'Chippewa': 'chip1241', 'Northwestern Ojibwa': 'nort2961', 'Northern Uzbek': 'nort2690', 'Tuvinian': 'tuvi1240', 'Shor': 'shor1247'}
  comparandum1_index = dict['Tuvinian']
  comparandum2_index = dict['Shor']
  comparandum1_row_number = find_row_number(comparandum1_index)
  comparandum2_row_number = find_row_number(comparandum2_index)
  comparandum1 = np.array([output_array[comparandum1_row_number]], np.float64)
  comparandum2 = np.array([output_array[comparandum2_row_number]], np.float64)
  model.infer(comparandum1, comparandum2)

def test6():
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
  def run():
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None)
    model.train(steps=3000)
    return model
  if not 'contact_weights.npy' in os.listdir('.'):
    model = run()
    weights = model.sess.run(model.contact_weights)
    intercept = model.sess.run(model.contact_intercept)
    np.save('contact_weights.npy', weights)
    np.save('contact_intercept.npy', intercept)
    weights = model.sess.run(model.weights)
    intercept = model.sess.run(model.intercept)
    np.save('weights.npy', weights)
    np.save('intercept.npy', intercept)
  else:
    contact_weights = np.load('contact_weights.npy')
    contact_intercept = np.load('contact_intercept.npy')
    weights = np.load('weights.npy')
    intercept = np.load('intercept.npy')
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=True, weights=weights, intercept=intercept, contact_weights=contact_weights, contact_intercept=contact_intercept)
  def find_row_number(index):
    return np.where(df.index==index)[0][0]
  dict = {'English': 'stan1293', 'Dutch': 'dutc1256', 'Swedish': 'swed1254', 'Mandarin': 'mand1415', 
  'Cantonese': 'yuec1235', 'Japanese': 'nucl1643', 'Korean': 'kore1280',
  'Thai': 'thai1261', 'Lao': 'laoo1244', 'Paiwan': 'paiw1248', 'Maori': 'maor1246', 'Mongolian': 'peri1253',
  'Koyr1': 'koyr1240', 'Koyr2': 'koyr1242', 'Tundra Nenets': 'nene1249', 'Finnish': 'finn1318', 'Veps': 'veps1250',
  'Tigon Mbembe': 'tigo1236', 'Congo Swahili': 'cong1236', 'Tahaggart Tamahaq': 'taha1241', 'Modern Hebrew': 'hebr1245',
  'Gulf Arabic': 'gulf1241', 'Narungga': 'naru1238', 'Yinggarda': 'ying1247', 'Thurawal': 'thur1254', 'Plateau Malagasy': 'plat1254',
  'Chippewa': 'chip1241', 'Northwestern Ojibwa': 'nort2961', 'Northern Uzbek': 'nort2690', 'Tuvinian': 'tuvi1240', 'Shor': 'shor1247'}
  comparandum1_index = dict['Mandarin']
  comparandum2_index = dict['Thai']
  comparandum1_row_number = find_row_number(comparandum1_index)
  comparandum2_row_number = find_row_number(comparandum2_index)
  comparandum1 = np.array([output_array[comparandum1_row_number]], np.float64)
  comparandum2 = np.array([output_array[comparandum2_row_number]], np.float64)
  model.contact_infer(comparandum1, comparandum2)

def test7():
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
  def run():
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None)
    model.train(steps=20000)
    return model
  if not 'contact_weights.npy' in os.listdir('.'):
    model = run()
    weights = model.sess.run(model.contact_weights)
    intercept = model.sess.run(model.contact_intercept)
    np.save('contact_weights.npy', weights)
    np.save('contact_intercept.npy', intercept)
    weights = model.sess.run(model.weights)
    intercept = model.sess.run(model.intercept)
    relatedness_distance_weights = model.sess.run(model.relatedness_distance_weights)
    relatedness_distance_intercept = model.sess.run(model.relatedness_distance_intercept)
    np.save('weights.npy', weights)
    np.save('intercept.npy', intercept)
    np.save('relatedness_distance_weights.npy', relatedness_distance_weights)
    np.save('relatedness_distance_intercept.npy', relatedness_distance_intercept)
  else:
    contact_weights = np.load('contact_weights.npy')
    contact_intercept = np.load('contact_intercept.npy')
    weights = np.load('weights.npy')
    intercept = np.load('intercept.npy')
    relatedness_distance_weights = np.load('relatedness_distance_weights.npy')
    relatedness_distance_intercept = np.load('relatedness_distance_intercept.npy')
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=True, weights=weights, intercept=intercept, contact_weights=contact_weights, contact_intercept=contact_intercept, relatedness_distance_weights=relatedness_distance_weights, relatedness_distance_intercept=relatedness_distance_intercept)
  def find_row_number(index):
    return np.where(df.index==index)[0][0]
  dict = {'English': 'stan1293', 'Dutch': 'dutc1256', 'Swedish': 'swed1254', 'Mandarin': 'mand1415', 
  'Cantonese': 'yuec1235', 'Japanese': 'nucl1643', 'Korean': 'kore1280',
  'Thai': 'thai1261', 'Lao': 'laoo1244', 'Paiwan': 'paiw1248', 'Maori': 'maor1246', 'Mongolian': 'peri1253',
  'Koyr1': 'koyr1240', 'Koyr2': 'koyr1242', 'Tundra Nenets': 'nene1249', 'Finnish': 'finn1318', 'Veps': 'veps1250',
  'Tigon Mbembe': 'tigo1236', 'Congo Swahili': 'cong1236', 'Tahaggart Tamahaq': 'taha1241', 'Modern Hebrew': 'hebr1245',
  'Gulf Arabic': 'gulf1241', 'Narungga': 'naru1238', 'Yinggarda': 'ying1247', 'Thurawal': 'thur1254', 'Plateau Malagasy': 'plat1254',
  'Chippewa': 'chip1241', 'Northwestern Ojibwa': 'nort2961', 'Northern Uzbek': 'nort2690', 'Tuvinian': 'tuvi1240', 'Shor': 'shor1247'}
  comparandum1_index = dict['English']
  comparandum2_index = dict['Swedish']
  comparandum1_row_number = find_row_number(comparandum1_index)
  comparandum2_row_number = find_row_number(comparandum2_index)
  comparandum1 = np.array([output_array[comparandum1_row_number]], np.float64)
  comparandum2 = np.array([output_array[comparandum2_row_number]], np.float64)
  model.infer(comparandum1, comparandum2)



def test8():
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
  def run():
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None)
    model.train(steps=20000)
    return model
  if not 'contact_weights.npy' in os.listdir('.'):
    model = run()
    weights = model.sess.run(model.contact_weights)
    intercept = model.sess.run(model.contact_intercept)
    np.save('contact_weights.npy', weights)
    np.save('contact_intercept.npy', intercept)
    weights = model.sess.run(model.weights)
    intercept = model.sess.run(model.intercept)
    relatedness_distance_weights = model.sess.run(model.relatedness_distance_weights)
    relatedness_distance_intercept = model.sess.run(model.relatedness_distance_intercept)
    np.save('weights.npy', weights)
    np.save('intercept.npy', intercept)
    np.save('relatedness_distance_weights.npy', relatedness_distance_weights)
    np.save('relatedness_distance_intercept.npy', relatedness_distance_intercept)
  else:
    contact_weights = np.load('contact_weights.npy')
    contact_intercept = np.load('contact_intercept.npy')
    weights = np.load('weights.npy')
    intercept = np.load('intercept.npy')
    relatedness_distance_weights = np.load('relatedness_distance_weights.npy')
    relatedness_distance_intercept = np.load('relatedness_distance_intercept.npy')
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=True, weights=weights, intercept=intercept, contact_weights=contact_weights, contact_intercept=contact_intercept, relatedness_distance_weights=relatedness_distance_weights, relatedness_distance_intercept=relatedness_distance_intercept)
  
  comparandum1_array = np.reshape(input_array, [2783892, features])
  comparandum2_array = []
  for i in range(samples):
    for j in range(samples-1):
      comparandum2_array.append(output_array[i])
  comparandum2_array = np.array(comparandum2_array)
  print(np.shape(comparandum1_array))
  print(np.shape(comparandum2_array))
  comparandum1 = comparandum1_array
  comparandum2 = comparandum2_array
  result = model.infer(comparandum1, comparandum2)  
  actual = np.reshape(relatedness_array, [2783892, 1])
  print(result)
  print(actual)
  np.save('result.npy', result)
  loss = model.show_loss(comparandum1, comparandum2, actual)
  print(loss)
  recall = model.show_relatedness_recall(comparandum1, comparandum2, actual)
  print('Recall:', recall)
  precision = model.show_relatedness_precision(comparandum1, comparandum2, actual)
  print('Precision: ', precision)

  if not 'glottocode_pairs_array.npy' in os.listdir('.'):
    glottocode_pairs_array = make_glottocode_pairs_array(dataframe, languages_dataframe)
    np.save('glottocode_pairs_array.npy', glottocode_pairs_array)
  else:
    glottocode_pairs_array = np.load('glottocode_pairs_array.npy')
  false_positives = model.show_false_positives(comparandum1, comparandum2, actual)
  true_positives = model.show_true_positives(comparandum1, comparandum2, actual)
  print('False positives: ', false_positives)
  print(np.shape(false_positives))
  print(glottocode_pairs_array[false_positives])
  print('True positives: ', true_positives)
  print(np.shape(true_positives))
  print(glottocode_pairs_array[true_positives])
 


def test9():
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
  def run():
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None)
    model.train(steps=20000)
    return model
  if not 'contact_weights.npy' in os.listdir('.'):
    model = run()
    weights = model.sess.run(model.contact_weights)
    intercept = model.sess.run(model.contact_intercept)
    np.save('contact_weights.npy', weights)
    np.save('contact_intercept.npy', intercept)
    weights = model.sess.run(model.weights)
    intercept = model.sess.run(model.intercept)
    relatedness_distance_weights = model.sess.run(model.relatedness_distance_weights)
    relatedness_distance_intercept = model.sess.run(model.relatedness_distance_intercept)
    np.save('weights.npy', weights)
    np.save('intercept.npy', intercept)
    np.save('relatedness_distance_weights.npy', relatedness_distance_weights)
    np.save('relatedness_distance_intercept.npy', relatedness_distance_intercept)
  else:
    contact_weights = np.load('contact_weights.npy')
    contact_intercept = np.load('contact_intercept.npy')
    weights = np.load('weights.npy')
    intercept = np.load('intercept.npy')
    relatedness_distance_weights = np.load('relatedness_distance_weights.npy')
    relatedness_distance_intercept = np.load('relatedness_distance_intercept.npy')
    model = Model10(input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=True, weights=weights, intercept=intercept, contact_weights=contact_weights, contact_intercept=contact_intercept, relatedness_distance_weights=relatedness_distance_weights, relatedness_distance_intercept=relatedness_distance_intercept)  
  comparandum1_array = np.reshape(input_array, [2783892, features])
  comparandum2_array = []
  for i in range(samples):
    for j in range(samples-1):
      comparandum2_array.append(output_array[i])
  comparandum2_array = np.array(comparandum2_array)
  print(np.shape(comparandum1_array))
  print(np.shape(comparandum2_array))
  batch_size = 100
  actual_array = np.reshape(relatedness_array, [2783892, 1])
  false_positives = np.array([], dtype=np.bool_)
  true_positives = np.array([], dtype=np.bool_)
  false_negatives = np.array([], dtype=np.bool_)
  recalls = np.array([], dtype = np.float64)
  precisions = np.array([], dtype = np.float64)
  relatedness_distance_predictions = np.array([], dtype = np.float64)
  length = 2783892
  for i in range(int(length/batch_size)):
    begin = i * batch_size
    print(begin)
    end = min((i+1)*batch_size, length)
    comparandum1 = comparandum1_array[begin:end]
    comparandum2 = comparandum2_array[begin:end]
    actual = actual_array[begin:end]
    batch_recall = model.show_relatedness_recall(comparandum1, comparandum2, actual)
    batch_false_positives = model.show_false_positives(comparandum1, comparandum2, actual)
    batch_true_positives = model.show_true_positives(comparandum1, comparandum2, actual)
    batch_false_negatives = model.show_false_negatives(comparandum1, comparandum2, actual)
    batch_relatedness_distance_predictions = model.show_relatedness_distance_predictions(comparandum1, comparandum2, actual)
    recalls = np.concatenate([recalls, [batch_recall]])
    false_positives = np.concatenate([false_positives, batch_false_positives])
  precision = 1 - (np.sum(false_positives) / (np.sum(false_positives) + np.sum(true_positives)))
  print('Precision: ', precision)
  if not 'glottocode_pairs_array.npy' in os.listdir('.'):
    glottocode_pairs_array = make_glottocode_pairs_array(dataframe, languages_dataframe)
    np.save('glottocode_pairs_array.npy', glottocode_pairs_array)
  else:
    glottocode_pairs_array = np.load('glottocode_pairs_array.npy')
  glottocode_pairs_array = glottocode_pairs_array[0:len(false_positives)]
  np.save('false_positives.npy', glottocode_pairs_array[false_positives])
  print('False positives: ', false_positives)
  print(glottocode_pairs_array[false_positives])

def test9_find_families():
  def find_family(glottocode):
    family = glottocodes.Family_name[glottocode]
    try:
      if np.isnan(family): 
        family = glottocodes.Name[glottocode]
    except:
      pass
    return family
  if not 'false_positives.npy' in os.listdir('.') or not 'fp_relatedness_distance_predictions.npy' in os.listdir('.'):
    test9()
  pairs = np.load('false_positives.npy')
  fp_relatedness_distance_predictions = np.load('fp_relatedness_distance_predictions.npy')
  for i in range(len(pairs)):
    pair = pairs[i]
    try:
      print(find_family(pair[0]), find_family(pair[1]), fp_relatedness_distance_predictions[i])
    except:
      pass
  
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
  test9()
  test9_find_families()

