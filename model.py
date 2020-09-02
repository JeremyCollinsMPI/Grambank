import tensorflow as tf
import numpy as np
from numpy import random
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def reverse_sigmoid(x):
  return np.log((1 / x) - 1) * -1

def rep(x,y):
	new=[]
	for m in range(y):
		new.append(x)
	return new	 

def loss_function(x, y):
  return tf.abs(tf.subtract(x, y))


class Model:
  learn_rate = 0.01
  def __init__(self, input_array, output_array, relatedness_array, distance_array, samples, features):  
    self.sess = tf.Session()
    self.input_array = input_array
    self.output_array = output_array
    self.relatedness_array = relatedness_array
    self.distance_array = distance_array
    self.samples = samples
    self.features = features
    self.randomly_selected_relatedness_input = tf.placeholder(tf.float64, [samples, features])
    self.relatedness_placeholder = tf.placeholder(tf.float64, shape=[samples, 1])   
    self.contact_placeholder = tf.placeholder(tf.float64, shape=[samples, 1])
    self.randomly_selected_contact_input = tf.placeholder(tf.float64, [samples, features])
    number_of_relatedness_intervals = 6
    self.relatedness_intervals = tf.convert_to_tensor(np.array([0,2,4,6,8,10], np.float64), tf.float64)
    self.relatedness_intervals  = tf.reshape(self.relatedness_intervals, [1, 1, number_of_relatedness_intervals])
    self.relatedness_intervals = tf.broadcast_to(self.relatedness_intervals, [samples, features, number_of_relatedness_intervals]) 
    self.relatedness_tensor = tf.reshape(self.relatedness_placeholder, [samples, 1, 1])
    self.relatedness_tensor = tf.broadcast_to(self.relatedness_tensor, [samples, features, number_of_relatedness_intervals])
    self.relatedness_intervals_max = tf.convert_to_tensor(np.array([2,4,6,8,10,1000], np.float64), tf.float64)
    self.relatedness_intervals_max  = tf.reshape(self.relatedness_intervals_max, [1, 1, number_of_relatedness_intervals])
    self.relatedness_intervals_max = tf.broadcast_to(self.relatedness_intervals_max, [samples, features, number_of_relatedness_intervals])   
    self.relatedness_comparison_a = tf.cast(tf.greater_equal(self.relatedness_tensor, self.relatedness_intervals), tf.float64)
    self.relatedness_comparison_b = tf.cast(tf.less(self.relatedness_tensor, self.relatedness_intervals_max), tf.float64)
    self.relatedness_comparison_c = tf.multiply(self.relatedness_comparison_a, self.relatedness_comparison_b)
    self.relatedness_weights = tf.get_variable(name='relatedness_weighting', dtype=tf.float64, shape=[features, number_of_relatedness_intervals], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_weighting = tf.reshape(self.relatedness_weights, [1, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.broadcast_to(self.relatedness_weighting, [samples, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.multiply(self.relatedness_weighting, self.relatedness_comparison_c)
    self.relatedness_weighting = tf.reduce_sum(self.relatedness_weighting, axis=2) 
    number_of_contact_intervals = 8
    self.contact_intervals = tf.convert_to_tensor(np.array([0,100,200,300,400,500,600,700], np.float64), tf.float64)
    self.contact_intervals  = tf.reshape(self.contact_intervals, [1, 1, number_of_contact_intervals])
    self.contact_intervals = tf.broadcast_to(self.contact_intervals, [samples, features, number_of_contact_intervals]) 
    self.contact_tensor = tf.reshape(self.contact_placeholder, [samples, 1, 1])
    self.contact_tensor = tf.broadcast_to(self.contact_tensor, [samples, features, number_of_contact_intervals])
    self.contact_intervals_max = tf.convert_to_tensor(np.array([100,200,300,400,500,600,700,100000], np.float64), tf.float64)
    self.contact_intervals_max  = tf.reshape(self.contact_intervals_max, [1, 1, number_of_contact_intervals])
    self.contact_intervals_max = tf.broadcast_to(self.contact_intervals_max, [samples, features, number_of_contact_intervals])    
    self.contact_comparison_a = tf.cast(tf.greater_equal(self.contact_tensor, self.contact_intervals), tf.float64)
    self.contact_comparison_b = tf.cast(tf.less(self.contact_tensor, self.contact_intervals_max), tf.float64)
    self.contact_comparison_c = tf.multiply(self.contact_comparison_a, self.contact_comparison_b)
    self.contact_weights = tf.get_variable(name='contact_weighting', dtype=tf.float64, shape=[features, number_of_contact_intervals], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_weighting = tf.reshape(self.contact_weights, [1, features, number_of_contact_intervals])
    self.contact_weighting = tf.broadcast_to(self.contact_weighting, [samples, features, number_of_contact_intervals])
    self.contact_weighting = tf.multiply(self.contact_weighting, self.contact_comparison_c)
    self.contact_weighting = tf.reduce_sum(self.contact_weighting, axis=2)
    self.universal_weighting = tf.convert_to_tensor(np.array([5], np.float64), dtype=tf.float64)   
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias_original = tf.get_variable(name='universal_bias', dtype=tf.float64, shape=[1, features], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.02))
    self.universal_bias = tf.broadcast_to(self.universal_bias_original, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) + tf.multiply(self.randomly_selected_contact_input, self.contact_weighting) 
    self.output = tf.placeholder(tf.float64, shape=[samples, features])    
    self.total_loss = tf.reduce_sum(tf.log(1 - tf.abs(self.output - self.prediction))) * -1
    
  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.clip_op1 = tf.assign(self.relatedness_weights, tf.clip_by_value(self.relatedness_weights, 0, 100000))
    self.clip_op2 = tf.assign(self.contact_weights, tf.clip_by_value(self.contact_weights, 0, 100000))
    self.clip_op3 = tf.assign(self.universal_bias_original, tf.clip_by_value(self.universal_bias_original, 0.01, 0.99))
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    for i in range(steps):
      relatedness_feed = []
      contact_feed = []
      random_relatedness_input_feed = []
      random_contact_input_feed = []
      for j in range(self.samples):
        number1 = random.randint(self.samples-2)
        relatedness_feed.append([self.relatedness_array[j][number1]])
        random_relatedness_input_feed.append(self.input_array[j][number1])
        number2 = random.randint(self.samples-2)
        x = self.distance_array[j][number2]
        contact_feed.append([x])
        random_contact_input_feed.append(self.input_array[j][number2])
        
      self.feed = {self.relatedness_placeholder: relatedness_feed, self.contact_placeholder: contact_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array, self.randomly_selected_contact_input: random_contact_input_feed}
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op1)
      self.sess.run(self.clip_op2)
      self.sess.run(self.clip_op3)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
    
  def show_weightings(self):
    return self.sess.run(self.relatedness_weights, feed_dict = self.feed), self.sess.run(self.contact_weights, feed_dict = self.feed)


