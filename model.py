import tensorflow as tf
import numpy as np
from numpy import random

def rep(x,y):
	new=[]
	for m in range(y):
		new.append(x)
	return new	 

def loss_function(x, y):
  return tf.abs(tf.subtract(x, y))


class Model2:
  '''
  you have 
  
  input array, which is again something like samples * 1 * 8
  then you turn this into something more complicated;
  [1,0,0...] and [0,1,0,0,...] 
  
  you are turning it into something like;
  there are 7 filters.
  
  [1,0,0...] goes to [1,1,0,1,0,0,0]
  [0,1,0...] goes to [1,1,0,0,1,0,0]
  [0,0,1,...] goes to [1,0,1,0,0,1,0]
  etc.
  I can write this programatically, but it basically needs to be written out.
  
  [root, 1st, 2nd, 1st_1, 1st_2, 2nd_1, 2nd_2]
  
  so you have the input array.
  you then multiply this by something like an array of 1 x 8.
  
  in detail:
  input array is of shape [samples, types, classes].  assume types is 1. classes here is 8.
  [samples, 1, 8].
  then you want the result of multiplying by the first layer to be of shape [samples, 1, 7]
  what you are multiplying by is of shape [8, 7]
  I will work out the details for that later.
  
  so you now have something of shape [samples, 1, 7].
  then you have seven filters.  so this layer is of shape [7, features]
  but you also may have more than one tree.
  so this layer should actually be of shape [1, 7, features]
  the result of multiplying is you have something of shape [samples, 1, 7, features]
  you then use reduce_sum along axis 2.  you get something of shape [samples, 1, features].
  you are then taking the minimum of the results and 1, and 
  the maximum of the results and 0.
  you are not taking into account transition probabilities between internal nodes yet,
  but ignore that for the moment.
  
  
  you end up with an output of shape [samples, 1, features].
  this is equivalent to prediction_a_1
  the rest is then the same.
  

  '''
  learn_rate = 0.001
  def __init__(self, output, missing_data_matrix, samples, types, classes, features):
    
    def generate_conversion(number_of_clades):
      result = []
      def binary(integer):
        return bin(integer).replace('0b', '')
      for i in range(number_of_clades):
        x = binary(integer)
        to_append = rep(0, number_of_clades-1)
        if integer[0] == '0':
          to_append
          
      '''
      first two are the top two clades
      next two are two clades in clade A
      next two are two clades in clade B
      next two are two clades in clade AA
      next two are two clades in clade AB
      next two are two clades in clade BA
      next two are two clades in clade BB
      
      so the pattern is that there are 8 clades that something can belong to;
      AAA, AAB, ABA etc.
      if there is 'A' at the beginning, then the first one should be 1
      if there is 'B' at the beginning, then the second one should be 1
      etc.
      
      
      '''
    
    
    self.samples = samples
    self.types = types
    self.classes = classes
    self.features = features   
    self.sess = tf.Session()
    self.input1 = tf.Variable(tf.zeros([samples, types, classes]), trainable=False)
    self.input2 = tf.Variable(tf.zeros([samples, types, classes]), trainable=False)


    '''now need conversion to other input type'''
    
    self.conversion1 = 
    self.conversion2 = 
    
    '''then multiply by weights.  
    
    the final output will be called prediction_a_1 and prediction_a_2
    before that, you will have the following steps:
    weights are of shape [1, 7, features]
    multiplying by weights gives something of shape [samples, 1, 7, features]
    
    
    
 
    
    '''



    self.weights = tf.Variable(tf.zeros([types, 7, features]))
    self.output_a1 = tf.tensordot(self.conversion1, self.weights)
    self.output_a2 = tf.tensordot(self.conversion2, self.weights)
    
    '''then reduce_sum'''
    
    self.output_b1 = tf.reduce_sum(self.output_a1, axis=2)
    self.output_b2 = tf.reduce_sum(self.output_a2, axis=2)
    self.output_c1 = tf.minimum(tf.maximum(self.output_b1, 1), 0)
    self.output_c2 = tf.minimum(tf.maximum(self.output_b2, 1), 0)
    self.prediction_a_1 = self.output_c1
    self.prediction_a_2 = self.output_c2
        
    self.missing_data_layer = tf.placeholder(tf.float32, [samples, features])
    self.final_prediction1 = tf.multiply(self.prediction_b_1, self.missing_data_layer)

    self.prediction_a_2 = tf.tensordot(self.input2, self.weights, axes=1)
    self.prediction_b_2 = tf.reduce_sum(self.prediction_a_2, axis=1) 
    self.final_prediction2 = tf.multiply(self.prediction_b_2, self.missing_data_layer)

    self.output = output
    self.missing_data_matrix = missing_data_matrix
    self.output_placeholder = tf.placeholder(tf.float32, [samples, features])

    self.differences1 = loss_function(self.output_placeholder, self.final_prediction1)
    self.differences2 = loss_function(self.output_placeholder, self.final_prediction2)

    self.differences1 = tf.reduce_sum(self.differences1, axis = 1)
    self.differences2 = tf.reduce_sum(self.differences2, axis = 1)

    self.differences1 = tf.reshape(self.differences1, [samples, 1, 1])
    self.differences1 = self.differences1 - 0.001
    self.differences2 = tf.reshape(self.differences2, [samples, 1, 1])

    self.averaged_differences = (self.differences1 + self.differences2) * 0.5
    self.total_loss = tf.reduce_sum(self.averaged_differences)

    self.differences_minimum = tf.minimum(self.differences1, self.differences2)
    self.differences1_equal_minimum = tf.cast(tf.equal(self.differences1, self.differences_minimum), tf.float32) 
    self.differences2_equal_minimum = tf.cast(tf.equal(self.differences2, self.differences_minimum), tf.float32) 
    
    self.differences1_equal_minimum = tf.broadcast_to(self.differences1_equal_minimum, [samples, types, classes])
    self.differences2_equal_minimum = tf.broadcast_to(self.differences2_equal_minimum, [samples, types, classes])

    self.new_input1 = (tf.multiply(self.input1, self.differences1_equal_minimum) + tf.multiply(self.input2, self.differences2_equal_minimum)) 

  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)
    samples = self.samples
    types = self.types
    classes = self.classes

    initial = random.randint(samples*types*classes, size=(samples, types, classes))
    maximum = np.reshape(np.max(initial, axis=2), (samples, types, 1))
    equal_maximum = np.equal(initial, maximum) * 1.0
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32) 
    self.initial_input1_step =  tf.assign(self.input1, random_input) 
    self.sess.run(self.initial_input1_step)
    to_change = random.randint(samples, size = 1)
    to_append = np.array(rep(0, classes))
    to_append[random.randint(classes, size = 1)] = 1
    equal_maximum[to_change] = to_append
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
    self.initial_input2_step =  tf.assign(self.input2, random_input) 
    self.sess.run(self.initial_input2_step)
    
    
    self.update_input1_step = tf.assign(self.input1, self.new_input1)
    self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
    print(self.sess.run(self.input1, feed_dict = self.feed))
    
    print(self.sess.run(self.differences2_equal_minimum, feed_dict = self.feed))
    
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       self.sess.run(self.train_step, feed_dict = self.feed)
       self.sess.run(self.update_input1_step, feed_dict = self.feed)
       
       new = self.sess.run(self.input1, feed_dict = self.feed)
       to_change = random.randint(samples, size = 1)[0]
       print("*****")
       print(to_change)
       to_append = np.array(rep(0, classes))
       print(to_append)
       to_append[random.randint(classes, size = 1)] = 1
       print(to_append)
       print("&&&&&&")
       print(new)
       new[to_change] = to_append
       random_input = tf.convert_to_tensor(new, dtype=tf.float32)
       self.update_input2_step = tf.assign(self.input2, random_input)       
       self.sess.run(self.update_input2_step)
       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))




  def show_classes(self):
    print(self.sess.run(self.input1))
    return self.sess.run(self.input1)
  
  def show_filters(self):
    print(self.sess.run(self.weights))

  def train_with_fixed_input(self, input_array, steps=100):
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)
    samples = self.samples
    types = self.types
    classes = self.classes

    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    
    self.initial_input1_step =  tf.assign(self.input1, input_tensor) 
    self.sess.run(self.initial_input1_step)
    self.initial_input2_step =  tf.assign(self.input2, input_tensor)
    self.sess.run(self.initial_input2_step)
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       self.sess.run(self.train_step, feed_dict = self.feed)
       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))

  def train_with_fixed_filters(self, steps=100):
    samples = self.samples
    types = self.types
    classes = self.classes

    initial = random.randint(samples*types*classes, size=(samples, types, classes))
    maximum = np.reshape(np.max(initial, axis=2), (samples, types, 1))
    equal_maximum = np.equal(initial, maximum) * 1.0
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32) 
    self.initial_input1_step =  tf.assign(self.input1, random_input) 
    self.sess.run(self.initial_input1_step)
    to_change = random.randint(samples, size = 1)
    to_append = np.array(rep(0, classes))
    to_append[random.randint(classes, size = 1)] = 1
    equal_maximum[to_change] = to_append
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
    self.initial_input2_step =  tf.assign(self.input2, random_input) 
    self.sess.run(self.initial_input2_step)
    
    
    self.update_input1_step = tf.assign(self.input1, self.new_input1)

    
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       
       self.sess.run(self.update_input1_step, feed_dict = self.feed)
       new = self.sess.run(self.input1, feed_dict = self.feed)
       to_change = random.randint(samples, size = 1)[0]
       print("*****")
       print(to_change)
       to_append = np.array(rep(0, classes))
       print(to_append)
       to_append[random.randint(classes, size = 1)] = 1
       print(to_append)
       print(new)
       new[to_change] = to_append
       random_input = tf.convert_to_tensor(new, dtype=tf.float32)
       self.update_input2_step = tf.assign(self.input2, random_input)       
       self.sess.run(self.update_input2_step)
       

       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))


class Model1:
  learn_rate = 0.001
  def __init__(self, output, missing_data_matrix, samples, types, classes, features):
    
    self.samples = samples
    self.types = types
    self.classes = classes
    self.features = features
    
    self.sess = tf.Session()
    self.input1 = tf.Variable(tf.zeros([samples, types, classes]), trainable=False)
    self.input2 = tf.Variable(tf.zeros([samples, types, classes]), trainable=False)
    self.weights = tf.Variable(tf.zeros([classes, features]))
    self.missing_data_layer = tf.placeholder(tf.float32, [samples, features])

    self.prediction_a_1 = tf.tensordot(self.input1, self.weights, axes=1)
    self.prediction_b_1 = tf.reduce_sum(self.prediction_a_1, axis=1) 
    self.final_prediction1 = tf.multiply(self.prediction_b_1, self.missing_data_layer)

    self.prediction_a_2 = tf.tensordot(self.input2, self.weights, axes=1)
    self.prediction_b_2 = tf.reduce_sum(self.prediction_a_2, axis=1) 
    self.final_prediction2 = tf.multiply(self.prediction_b_2, self.missing_data_layer)

    self.output = output
    self.missing_data_matrix = missing_data_matrix
    self.output_placeholder = tf.placeholder(tf.float32, [samples, features])

    self.differences1 = loss_function(self.output_placeholder, self.final_prediction1)
    self.differences2 = loss_function(self.output_placeholder, self.final_prediction2)

    self.differences1 = tf.reduce_sum(self.differences1, axis = 1)
    self.differences2 = tf.reduce_sum(self.differences2, axis = 1)

    self.differences1 = tf.reshape(self.differences1, [samples, 1, 1])
    self.differences1 = self.differences1 - 0.001
    self.differences2 = tf.reshape(self.differences2, [samples, 1, 1])

    self.averaged_differences = (self.differences1 + self.differences2) * 0.5
    self.total_loss = tf.reduce_sum(self.averaged_differences)

    self.differences_minimum = tf.minimum(self.differences1, self.differences2)
    self.differences1_equal_minimum = tf.cast(tf.equal(self.differences1, self.differences_minimum), tf.float32) 
    self.differences2_equal_minimum = tf.cast(tf.equal(self.differences2, self.differences_minimum), tf.float32) 
    
    self.differences1_equal_minimum = tf.broadcast_to(self.differences1_equal_minimum, [samples, types, classes])
    self.differences2_equal_minimum = tf.broadcast_to(self.differences2_equal_minimum, [samples, types, classes])

    self.new_input1 = (tf.multiply(self.input1, self.differences1_equal_minimum) + tf.multiply(self.input2, self.differences2_equal_minimum)) 

  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)
    samples = self.samples
    types = self.types
    classes = self.classes

    initial = random.randint(samples*types*classes, size=(samples, types, classes))
    maximum = np.reshape(np.max(initial, axis=2), (samples, types, 1))
    equal_maximum = np.equal(initial, maximum) * 1.0
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32) 
    self.initial_input1_step =  tf.assign(self.input1, random_input) 
    self.sess.run(self.initial_input1_step)
    to_change = random.randint(samples, size = 1)
    to_append = np.array(rep(0, classes))
    to_append[random.randint(classes, size = 1)] = 1
    equal_maximum[to_change] = to_append
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
    self.initial_input2_step =  tf.assign(self.input2, random_input) 
    self.sess.run(self.initial_input2_step)
    
    
    self.update_input1_step = tf.assign(self.input1, self.new_input1)
    self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
    print(self.sess.run(self.input1, feed_dict = self.feed))
    
    print(self.sess.run(self.differences2_equal_minimum, feed_dict = self.feed))
    
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       self.sess.run(self.train_step, feed_dict = self.feed)
       self.sess.run(self.update_input1_step, feed_dict = self.feed)
       
       new = self.sess.run(self.input1, feed_dict = self.feed)
       to_change = random.randint(samples, size = 1)[0]
       print("*****")
       print(to_change)
       to_append = np.array(rep(0, classes))
       print(to_append)
       to_append[random.randint(classes, size = 1)] = 1
       print(to_append)
       print("&&&&&&")
       print(new)
       new[to_change] = to_append
       random_input = tf.convert_to_tensor(new, dtype=tf.float32)
       self.update_input2_step = tf.assign(self.input2, random_input)       
       self.sess.run(self.update_input2_step)
       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))




  def show_classes(self):
    print(self.sess.run(self.input1))
    return self.sess.run(self.input1)
  
  def show_filters(self):
    print(self.sess.run(self.weights))

  def train_with_fixed_input(self, input_array, steps=100):
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)
    samples = self.samples
    types = self.types
    classes = self.classes

    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    
    self.initial_input1_step =  tf.assign(self.input1, input_tensor) 
    self.sess.run(self.initial_input1_step)
    self.initial_input2_step =  tf.assign(self.input2, input_tensor)
    self.sess.run(self.initial_input2_step)
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       self.sess.run(self.train_step, feed_dict = self.feed)
       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))

  def train_with_fixed_filters(self, steps=100):
    samples = self.samples
    types = self.types
    classes = self.classes

    initial = random.randint(samples*types*classes, size=(samples, types, classes))
    maximum = np.reshape(np.max(initial, axis=2), (samples, types, 1))
    equal_maximum = np.equal(initial, maximum) * 1.0
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32) 
    self.initial_input1_step =  tf.assign(self.input1, random_input) 
    self.sess.run(self.initial_input1_step)
    to_change = random.randint(samples, size = 1)
    to_append = np.array(rep(0, classes))
    to_append[random.randint(classes, size = 1)] = 1
    equal_maximum[to_change] = to_append
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
    self.initial_input2_step =  tf.assign(self.input2, random_input) 
    self.sess.run(self.initial_input2_step)
    
    
    self.update_input1_step = tf.assign(self.input1, self.new_input1)

    
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       
       self.sess.run(self.update_input1_step, feed_dict = self.feed)
       new = self.sess.run(self.input1, feed_dict = self.feed)
       to_change = random.randint(samples, size = 1)[0]
       print("*****")
       print(to_change)
       to_append = np.array(rep(0, classes))
       print(to_append)
       to_append[random.randint(classes, size = 1)] = 1
       print(to_append)
       print(new)
       new[to_change] = to_append
       random_input = tf.convert_to_tensor(new, dtype=tf.float32)
       self.update_input2_step = tf.assign(self.input2, random_input)       
       self.sess.run(self.update_input2_step)
       

       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))


  def train_old(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)
    samples = self.samples
    types = self.types
    classes = self.classes

    initial = random.randint(samples*types*classes, size=(samples, types, classes))
    maximum = np.reshape(np.max(initial, axis=2), (samples, types, 1))
    equal_maximum = np.equal(initial, maximum) * 1.0
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32) 
    self.initial_input1_step =  tf.assign(self.input1, random_input) 
    self.sess.run(self.initial_input1_step)
    to_change = random.randint(samples, size = 1)
    to_append = np.array(rep(0, classes))
    to_append[random.randint(classes, size = 1)] = 1
    equal_maximum[to_change] = to_append
    random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
    self.initial_input2_step =  tf.assign(self.input2, random_input) 
    self.sess.run(self.initial_input2_step)
    
    
    self.update_input1_step = tf.assign(self.input1, self.new_input1)

    
    for i in range(steps):
       print(i)
       
       '''
       '''
       self.feed = {self.output_placeholder: self.output, self.missing_data_layer: self.missing_data_matrix}
       
       self.sess.run(self.train_step, feed_dict = self.feed)
       
       new = self.sess.run(self.input1, feed_dict = self.feed)
       to_change = random.randint(samples, size = 1)
       to_append = np.array(rep(0, classes))
       to_append[random.randint(classes, size = 1)] = 1
       new[to_change] = to_append
       random_input = tf.convert_to_tensor(equal_maximum, dtype=tf.float32)
       self.update_input2_step = tf.assign(self.input2, random_input)       
       self.sess.run(self.update_input2_step)


       print("After %d iterations:" % i)
       print(self.sess.run(self.total_loss, feed_dict = self.feed))



#        '''
#        you then need to have a step which finds the best fitting of the two inputs for each language
#        what does this do?
#        you have an array which is of shape [samples,2].  each value is 0 or 1.  1 means it is the maximum.
#        you then multiply this array by an array which is of shape [samples, 2, types, classes]
#        the idea is you then get a result which is
#        
#        [samples, 2, types, classes]
#        you then reduce sum to get 
#        [samples, types, classes]
#        
#        so maybe recast the first array as [samples, 2, 1, 1] then multiply by the second array.
#        seems to work so far.
#        then reduce sum along axis = 1, to get an array of shape
#        [samples, types, classes]
#        
#        you need to produce the array which shows the maximum of the two inputs
#        
#        
#        this is done by the differences
#        
#        differences1 and differences2 are both of shape [samples, features]
#        you find the element-wise maximum
#        then you compare the first one to the maximum
#        differences_maximum = tf.max (...)
#        tf.equal (differences1, differences_maximum) if this is correct.  you need this to be an integer 0 or 1.
#        tf.cast(tf.equal(differences1, differences_maximum), tf.float32)
#        
#        
#        self.differences_maximum is of shape [samples, features]
#        so is self.differences1_equal_maximum is as well.  it is of shape [samples, features], and is 0s and 1s.
#       
#        another way of doing this
#        you have differences1 whether it equals the maximum
#        similarly for differences2
#        these are of shape [samples, features]
#        you can multiply the things comparing differences1 with the maximum by 
#        input1 which is of shape [samples, types, classes]
#        similarly for the one comparing differences2 with the maximum, which you multiply by input2
#        then you add them element wise
#        
#        
#        
#        '''
#        
#        
# 
#        
#        
#        
#        
#        
#        
#        '''
#       you have input1 and input2
#       you want to have these randomly initialised first.
#       
#       at each step you have current, and some new version.
#       so in the first step you have current = initial1 and new = initial2
#       
#       
#        '''
       
       
#     predictions = self.sess.run(self.final_prediction, feed_dict = self.feed )
#     for i in range(len(predictions)):
#       print(predictions[i])
    


'''

next things to do;

need to have random sampling of filters in the train function

need to check that self.differences can be part of reduce_sum
differences shape is equal to (samples, features)
to get a single number, i think you can just do reduce_sum.










the input should be;
in the first iteration, there is a single number saying what filter to use.
a single one-hot vector
num_types = 1
num_classes = e.g. 7
so shape = [num_types, samples, num_classes]
and then you have the number of samples
not sure if the order is correct here, will check

then in the first iteration, 
you are simply multiplying this by a layer which has filters.

self.weights = tf.Variable(tf.zeros([num_types, features, num_classes]))

again not sure the order is correct here


self.initial_prediction = tf.matmul(self.input, self.weights)

you then have a loss function which takes into account the rate of change of each feature.

but say that first you do not have branch lengths or rates of change.

so you just have a simple loss function.

need to get th 


what am I trying to work out?

i have something like this for the input:

[samples, num_types, num_classes]

you then want to take the tensor product (?) with the weights, which look something like this:

[num_classes, features]

the desired output has something like this shape:

[samples, features]

so what is the operation?

let's say samples = 1 and num_types = 1 and num_classes = 7



all you are doing is:

[[1,0,0,0,0,0,0]]

then multiplying it by 

[[....], [...], [..]...]

 import numpy as np
>>> a = np.array([[1,0,0,0,0]])
>>> b = [[1,2,3],[4,5,6],[6,7,8],[3,2,4],[8,6,8]]
>>> b = np.array([[1,2,3],[4,5,6],[6,7,8],[3,2,4],[8,6,8]])
>>> np.matmul(a,b)
array([[1, 2, 3]])
>>> 

now try having two samples
 a = np.array([[1,0,0,0,0],[0,1,0,0,0]])
>>> np.matmul(a,b)
array([[1, 2, 3],
       [4, 5, 6]])
seems fine
now try two types

a = np.array([[1,0,0,0,0], [0,1,0,0,0]])

i think i haven't got the shape of a right yet.

should be 

a = np.array([[[1,0,0,0,0]]])
> np.matmul(a,b)
array([[[1, 2, 3]]])

a = np.array([[[1,0,0,0,0]],[[0,1,0,0,0]]])
 a = np.array([[[1,0,0,0,0]],[[0,1,0,0,0]]])
>>> np.matmul(a,b)
array([[[1, 2, 3]],

       [[4, 5, 6]]])
       
a = np.array([[[1,0,0,0,0],[1,0,0,0,0]]])       
two types - looks wrong, or may be fine.  unclear.
 a = np.array([[[1,0,0,0,0],[1,0,0,0,0]]]) 
>>> np.matmul(a,b)
array([[[1, 2, 3],
        [1, 2, 3]]])
        
        
        
trying tensordot now
 a = np.array([[[1,0,0,0,0]]])
>>> np.tensordot(a,b,axes=0)
array([[[[[1, 2, 3],
          [4, 5, 6],
          [6, 7, 8],
          [3, 2, 4],
          [8, 6, 8]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]]]])
          
not the result I want.

np.tensordot(a,b,axes=1)
array([[[1, 2, 3]]])

this looks right.


two samples;


a = np.array([[[1,0,0,0,0]],[[0,1,0,0,0]]])

array([[[1, 2, 3]],

       [[4, 5, 6]]])
might be fine.

what is the result I want?

I want samples x features, basically.

two types;

>>> a = np.array([[[1,0,0,0,0],[1,0,0,0,0]]])   
>>> np.tensordot(a,b,axes=1)
array([[[1, 2, 3],
        [1, 2, 3]]])

again looks weird.

i'm not sure what i'm trying to do.

a.shape == (1, 2, 5)

b.shape == (5, 3)

i want a tensor with shape (1, 3).  instead it is-
>>> np.tensordot(a,b,axes=1).shape
(1, 2, 3)

actually this is fine.

so you want:

(samples, types, classes)

and weights are

(classes, features)

and you want 

tf.tensordot(input, weights, axes = 1)


which will have shape

(samples, types, features)

you then reduce sum

np.sum(thing, axis=1)





'''


# example that apparently works;
#         self.input_reshaped = tf.broadcast_to(self.input_rounded, [2, num_types, variants, samples, num_classes])
# #     self.input_reshaped = self.input_reshaped + 0.01
#     self.input_reshaped = tf.minimum(self.input_reshaped, 1.0)
#     self.output = output
#     self.output_placeholder = tf.placeholder(tf.float32, [variants, samples, 4])
#     self.weights = tf.Variable(tf.zeros([1, num_types, variants, num_classes, 4]))
#     self.weights_reshaped = tf.broadcast_to(self.weights, [2, num_types, variants, num_classes, 4])
#     self.initial_prediction = tf.matmul(self.input_reshaped, self.weights_reshaped) 


        
#   def train(self, steps=100): 
#     self.differences = tf.abs(tf.subtract(self.output_placeholder, self.initial_prediction))
#     self.cost = tf.reduce_sum(self.differences)
#     self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cost)
#     self.feed = {self.output_placeholder: self.output}
#     init = tf.initialize_all_variables()
#     self.sess.run(init)
#     for i in range(steps):
#        print(i)
#        self.sess.run(self.train_step, feed_dict = self.feed)
#        print("After %d iterations:" % i)
#        print(self.sess.run(self.cost, feed_dict = self.feed))
# 
#     predictions = self.sess.run(self.final_prediction, feed_dict = self.feed )
#     for i in range(len(predictions)):
#       print(predictions[i])
#     
#     print(self.sess.run(self.input_rounded, feed_dict = self.feed))
#     print(self.sess.run(self.input, feed_dict = self.feed))
#     print(self.sess.run(self.differences, feed_dict = self.feed))


'''
the input always needs to be rounded.  the maximum one becomes one and the others become zero.


'''



'''
samples = 2504
variants = 1747
output is array of shape (samples, variants, 2, 4)
input is a one hot vector.  
num_classes = 4
indices = (samples, 1)
input = tf.keras.backend.one_hot(
    indices, num_classes
)
so the input is shaped (samples, 1, num_classes)
if a sample is in a particular class, it is predicting an output for that sample of shape (variants, 1, 4).
then you need to do this for both copies.  
so actually you need to double the length of the output.  that's one way of doing it.
another way.

num_classes = 4
indices = (samples, 2)
input = tf.keras.backend.one_hot(
    indices, num_classes
)
input is of shape (samples, 2, num_classes)
flipped_input = somehow reverse the order of the 2, so along axis 1 (check this is correct in numpy)
flipped_input = tf.reverse(input, [1])

you then multiply input by a tensor 
(num_classes, variants, 4)
to get the output (samples, variants, 2, 4)


you have an output for that sample of shape (variants, 2, 4).  you also want the permuted version of that.
you have input, which generates an output of shape (samples, variants, 2, 4)
you also want flipped_input which generates a flipped_output of same shape.  you calculate the loss (sum of losses) of both output and flipped_output and 
take the mean.

input is of shape (samples, 2, num_classes)
weights is of shape (num_classes, variants, 4)
need to rethink the shape since the multiplication isn't working.

input is of shape (3, 2, 4)
weights is of shape (4, 1, 4) 
weights is of shape (1, 4, 4)
output should be of shape (3, 1, 2, 4)

rethinking.

the input should be of shape
(samples, number_of_types, 2, num_classes)
so indices = (samples, 1, 2)

so input is of shape (3,1,2,4)
weights is of shape (1,4,4)
output is then of shape (3,1,2,4)
from experimenting in numpy


input = 


type 1, type 2
[1 0 0 0 0], [0 1 0 0 0]

A T
A G
...

[1 0 0 0] [0 1 0 0]
[1 0 0 0] [0 0 1 0]



[[1 0 0 0] [0 1 0 0] [1 0 0 0] [...] [...]]
[[1 0 0 0] [0 0 1 0] [...] [...] [...]]

(2, 5)
(2, )

---------


1 0 0 0 0
...
...

x

1 0
0 1
0 0
0 0 . . .



(samples, num_classes)  x (num_classes, 4)
=> (samples, 4)


---

so what I wanted (assuming no copies and only one type allowed) was
(variants, samples, num_classes) x (variants, num_classes,4)
resulting in 
(variants, samples, 4)

----

however; first, you want to have two copies for each variant.

(2, variants, samples, num_classes) x (2, variants, num_classes, 4)

---

also you want to have multiple types.
how do you deal with that?
you would do that before the copies

(2, num_types, variants, samples, num_classes) x (2, num_types, variants, num_classes, 4)
=> [2, num_types, variants, samples, 4]

[[1 0 0 0] [0 1 0 0] [1 0 0 0] [...] [...]]





a b    e f
c d    g h

ae+bg af+bh
ce+dg cf+dh
------




'''





































