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


