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




class Model10:

  learn_rate = 0.01
  def __init__(self, input_array, output_array, relatedness_array, distance_array, samples, features, use_weights=False, weights=None, intercept=None, contact_weights=None, contact_intercept=None, relatedness_distance_weights=None, relatedness_distance_intercept=None):  
    self.sess = tf.Session()
    self.input_array = input_array
    self.output_array = output_array
    self.relatedness_array = relatedness_array
    self.distance_array = distance_array
    self.samples = samples
    self.features = features
    self.comparandum1 = tf.placeholder(tf.float64, [None, features])
    self.comparandum2 = tf.placeholder(tf.float64, shape=[None, features])
    self.relatedness_placeholder = tf.placeholder(tf.float64, shape=[None, 1])
    self.relatedness = self.relatedness_placeholder / 100.0
    self.relatedness_class1 = tf.cast(tf.less(self.relatedness, 0.5), tf.float64)
    self.unrelated = 1.0 - self.relatedness_class1
    
    '''
    how do you calculate the cross entropy?
    
    you have the real relatedness;
    you have your estimate for relatedness_class1;
    you also have your estimate for the relatedness.
    then let's say that it is to do with abs(your estimate for the relatedness - relatedness)
    say that it is a normal distribution;
    
    
    tf.distributions.Normal(loc= predicted_relatedness, scale=0.1)
    self.relatedness_difference = self.predicted_relatedness - self.relatedness
    normal_dist = tf.distributions.Normal(0, 0.2)
    self.relatedness_difference_loss = normal_dist.cdf(self.relatedness_distance + 0.1) - normal_dist.cdf(self.relatedness_distance - 0.1)
    
    
    
    
    '''
    
    
    self.contact_placeholder = tf.placeholder(tf.float64, shape=[None, 1])
    self.contact = self.contact_placeholder
    self.contact_class1 = tf.cast(tf.less(self.contact, 10000), tf.float64)

    self.layer1a = tf.cast(tf.equal(self.comparandum1, 0.0), tf.float64)
    self.layer1b = tf.cast(tf.equal(self.comparandum2, 0.0), tf.float64)
    self.layer1c = self.layer1a * self.layer1b

    
    self.layer2a = tf.cast(tf.equal(self.comparandum1, 1.0), tf.float64)
    self.layer2b = tf.cast(tf.equal(self.comparandum2, 1.0), tf.float64)
    self.layer2c = self.layer2a * self.layer2b

    self.layer3 = tf.concat([self.layer1c, self.layer2c], axis=1)
    if not use_weights:
      self.weights = tf.get_variable(name='weights', dtype = tf.float64, shape = [2*features, 1],  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.00001))
      self.intercept = tf.get_variable(name='intercepts', dtype = tf.float64, shape = [1],  initializer=tf.truncated_normal_initializer(mean=-2.0, stddev=1.0))
      self.relatedness_distance_weights = tf.get_variable(name='relatedness_distance_weights', dtype = tf.float64, shape = [2*features, 1],  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.00001))
      self.relatedness_distance_intercept = tf.get_variable(name='relatedness_distance_intercepts', dtype = tf.float64, shape = [1],  initializer=tf.truncated_normal_initializer(mean=-2.0, stddev=1.0))

      self.contact_weights = tf.get_variable(name='contact_weights', dtype = tf.float64, shape = [2*features, 1],  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.00001))
      self.contact_intercept = tf.get_variable(name='contact_intercepts', dtype = tf.float64, shape = [1],  initializer=tf.truncated_normal_initializer(mean=-0, stddev=1.0))

    else:
      self.weights = tf.convert_to_tensor(weights, tf.float64)
      self.intercept = tf.convert_to_tensor(intercept, tf.float64)
      self.relatedness_distance_weights = tf.convert_to_tensor(relatedness_distance_weights, tf.float64)
      self.relatedness_distance_intercept = tf.convert_to_tensor(relatedness_distance_intercept, tf.float64)
      self.contact_weights = tf.convert_to_tensor(contact_weights, tf.float64)
      self.contact_intercept = tf.convert_to_tensor(contact_intercept, tf.float64)
    self.prediction = tf.sigmoid(tf.matmul(self.layer3, self.weights) + self.intercept)
    self.relatedness_distance_prediction = tf.sigmoid(tf.matmul(self.layer3, self.relatedness_distance_weights) + self.relatedness_distance_intercept)
    self.contact_prediction = tf.sigmoid(tf.matmul(self.layer3, self.contact_weights) + self.contact_intercept)

    '''
    if unrelated, then it is loss
    if related, then it is probability of being related * the probability deduced from the relatedness distance prediction
    '''
   
    self.loss = 1.0 - tf.abs(self.prediction - self.relatedness_class1)


    
#     self.relatedness_distance_loss = 1.0 - tf.sigmoid(tf.abs(self.relatedness - self.relatedness_distance_prediction))
    self.relatedness_distance_prediction = tf.cast(self.relatedness_distance_prediction, tf.float32)
    self.relatedness = tf.cast(self.relatedness, tf.float32)
    normal_dist = tf.distributions.Normal(self.relatedness_distance_prediction, 0.03)
#     
#     
#     self.relatedness_difference = self.relatedness_distance_prediction - self.relatedness
#     self.relatedness_difference = tf.cast(self.relatedness_difference, tf.float32)
    self.relatedness_distance_loss = normal_dist.cdf(self.relatedness + 0.01) - normal_dist.cdf(self.relatedness - 0.01)
# 
# #     self.relatedness_distance_loss = 1.0
    self.relatedness_distance_loss = tf.cast(self.relatedness_distance_loss, tf.float64)
    self.unrelated_loss = self.unrelated * self.loss
    self.related_loss = self.relatedness_class1 * self.prediction * self.relatedness_distance_loss
    self.final_loss = tf.log(self.unrelated_loss + self.related_loss) * -1
    self.total_loss = tf.reduce_sum(self.final_loss)
    self.contact_loss = tf.log(1.0 - tf.abs(self.contact_prediction - self.contact_class1)) * -1.0
    self.contact_total_loss = tf.reduce_sum(self.contact_loss)

    self.relatedness_accuracy_threshold = 0.95
    self.relatedness_prediction_rounded = tf.cast(tf.greater_equal(self.prediction, self.relatedness_accuracy_threshold), tf.float64)
    self.recall = tf.reduce_sum(tf.minimum(self.relatedness_prediction_rounded, self.relatedness_class1)) / tf.maximum(tf.reduce_sum(self.relatedness_class1), 0.1)
    self.fp_rate = tf.reduce_sum(tf.minimum(self.relatedness_prediction_rounded, self.unrelated)) / tf.maximum(tf.reduce_sum(self.relatedness_prediction_rounded), 0.1)

    self.false_positives = tf.minimum(self.relatedness_prediction_rounded, self.unrelated)
    self.true_positives = tf.minimum(self.relatedness_prediction_rounded, self.relatedness_class1)
    self.false_negatives = tf.minimum(self.relatedness_class1, 1.0 - self.relatedness_prediction_rounded)
    self.false_positives = tf.cast(tf.reshape(self.false_positives, [-1]), tf.bool)
    self.true_positives = tf.cast(tf.reshape(self.true_positives, [-1]), tf.bool)
    self.false_negatives = tf.cast(tf.reshape(self.false_negatives, [-1]), tf.bool)


  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.contact_train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.contact_total_loss)  

    self.clip_op = tf.assign(self.relatedness_distance_weights, tf.clip_by_value(self.relatedness_distance_weights, -100, 0))
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
        contact_feed.append([self.distance_array[j][number1]])
        random_contact_input_feed.append(self.input_array[j][number1]) 
      '''
      
      '''      
      self.feed = {self.relatedness_placeholder: relatedness_feed, self.contact_placeholder: contact_feed, self.comparandum1: random_contact_input_feed, self.comparandum2: self.output_array}
      print("After %d iterations:" % i)
      contact_total_loss = self.sess.run(self.contact_total_loss, feed_dict = self.feed)
      if np.isnan(contact_total_loss):
        print(self.sess.run(self.contact_weights))
        print(self.sess.run(self.contact_intercept))
        exit(0)
      self.sess.run(self.train_step, feed_dict = self.feed)
      print(self.sess.run(self.total_loss, feed_dict=self.feed))
#       print(self.sess.run(self.clip_op))
#       self.sess.run(self.contact_train_step, feed_dict = self.feed)
      if i % 1000 == 0:
        print(self.sess.run(self.relatedness_distance_prediction, feed_dict=self.feed))
        print(self.sess.run(self.relatedness, feed_dict=self.feed))
        print(self.sess.run(self.related_loss, feed_dict=self.feed))
        print(self.sess.run(self.relatedness_class1, feed_dict=self.feed))
        print(self.sess.run(self.related_loss, feed_dict=self.feed))
        print(self.sess.run(self.relatedness_distance_weights))
        print(self.sess.run(self.relatedness_distance_intercept))


  def infer(self, comparandum1, comparandum2):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2}
    print(self.sess.run(self.prediction, feed_dict = self.feed), self.sess.run(self.relatedness_distance_prediction, feed_dict = self.feed))
    return self.sess.run(self.prediction, feed_dict = self.feed), self.sess.run(self.relatedness_distance_prediction, feed_dict = self.feed)

  def contact_infer(self, comparandum1, comparandum2):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2}
    print(self.sess.run(self.contact_prediction, feed_dict = self.feed))

  '''
  you want a function which shows the loss first
  
  '''
  
  def show_loss(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(self.total_loss, feed_dict = self.feed)

  def show_relatedness_recall(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(self.recall, feed_dict = self.feed)
    
  def show_false_positives(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(self.false_positives, feed_dict = self.feed)    

  def show_true_positives(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(self.true_positives, feed_dict = self.feed)    

  def show_false_negatives(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(self.false_negatives, feed_dict = self.feed)    

  def show_relatedness_distance_predictions(self, comparandum1, comparandum2, actual):
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2, self.relatedness_placeholder: actual}
    return self.sess.run(tf.reshape(self.relatedness_distance_prediction, [-1]), feed_dict = self.feed) 

class Model9:
  learn_rate = 0.1
  def __init__(self, input_array, output_array, relatedness_array, distance_array, samples, features, relatedness_a, relatedness_b, relatedness_c, contact_a, contact_b, contact_c, universal_bias):  
    self.sess = tf.Session()
    self.input_array = input_array
    self.output_array = output_array
    self.relatedness_array = relatedness_array
    self.distance_array = distance_array
    self.samples = samples
    self.features = features
 
    self.comparandum1 = tf.placeholder(tf.float64, [samples, features])
    self.comparandum2 = tf.placeholder(tf.float64, [samples, features])

    self.relatedness_variable = tf.get_variable(name='relatedness', dtype=tf.float64, shape=[samples, 1],initializer=tf.truncated_normal_initializer(mean=5, stddev=1))    
    self.contact_variable = tf.get_variable(name='contact', dtype=tf.float64, shape=[samples, 1],initializer=tf.truncated_normal_initializer(mean=5000, stddev=1000))
   
    self.relatedness_tensor = tf.broadcast_to(self.relatedness_variable, [samples, features]) 
    self.contact_tensor = tf.broadcast_to(self.contact_variable, [samples, features]) / 1000
    
    self.relatedness_a = tf.convert_to_tensor(relatedness_a, tf.float64)
    self.relatedness_b = tf.convert_to_tensor(relatedness_b, tf.float64)
    self.relatedness_c = tf.convert_to_tensor(relatedness_c, tf.float64)
    self.contact_a = tf.convert_to_tensor(contact_a, tf.float64)
    self.contact_b = tf.convert_to_tensor(contact_b, tf.float64)
    self.contact_c = tf.convert_to_tensor(contact_c, tf.float64)
    
    self.relatedness_tensor = tf.pow(self.relatedness_tensor, self.relatedness_c)
    self.relatedness_weighting = tf.pow(self.relatedness_b, self.relatedness_tensor)
    self.relatedness_weighting = tf.multiply(self.relatedness_a, self.relatedness_weighting)   
    self.contact_tensor = tf.pow(self.contact_tensor, self.contact_c)
    self.contact_weighting = tf.pow(self.contact_b, self.contact_tensor)
    self.contact_weighting = tf.multiply(self.contact_a, self.contact_weighting)

    self.universal_weighting = tf.convert_to_tensor(np.array([5], np.float64), dtype=tf.float64)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias_original = tf.convert_to_tensor(universal_bias, tf.float64)
    self.universal_bias = tf.broadcast_to(self.universal_bias_original, [samples, features])

    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.comparandum1, self.relatedness_weighting) + tf.multiply(self.comparandum1, self.contact_weighting) 
    self.output = self.comparandum2
    self.total_loss = tf.reduce_sum(tf.log(tf.maximum(1 - tf.abs(self.output - self.prediction), 0.00000001))) * -1

      
  def infer(self, comparandum1, comparandum2, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.clip_op1 = tf.assign(self.relatedness_variable, tf.clip_by_value(self.relatedness_variable, 0.1, 100000))
    self.clip_op2 = tf.assign(self.contact_variable, tf.clip_by_value(self.contact_variable, 0.1, 100000))
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    for i in range(steps):        
      self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2}
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op1)
      self.sess.run(self.clip_op2)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_variable))
      print(self.sess.run(self.contact_variable))


class Model8:
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
    self.randomly_selected_contact_input = tf.placeholder(tf.float64, [samples, features])
    self.relatedness_placeholder = tf.placeholder(tf.float64, shape=[samples, 1])
    self.contact_placeholder = tf.placeholder(tf.float64, shape=[samples, 1])    
    self.relatedness_tensor = tf.broadcast_to(self.relatedness_placeholder, [samples, features]) 
    self.contact_tensor = tf.broadcast_to(self.contact_placeholder, [samples, features]) / 1000
    
    self.relatedness_a = tf.get_variable('relatedness_a', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_b = tf.get_variable('relatedness_b', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=0.99, stddev=0.00001))
    self.relatedness_c = tf.get_variable('relatedness_c', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=1, stddev=0.00001))
#     self.relatedness_a_broadcast = tf.broadcast_to(self.relatedness_a, [samples, features])
#     self.relatedness_b_broadcast = tf.broadcast_to(self.relatedness_b, [samples, features])
    self.relatedness_tensor = tf.pow(self.relatedness_tensor, self.relatedness_c)
    self.relatedness_weighting = tf.pow(self.relatedness_b, self.relatedness_tensor)
    self.relatedness_weighting = tf.multiply(self.relatedness_a, self.relatedness_weighting)
    self.k = self.relatedness_weighting
    self.contact_a = tf.get_variable('contact_a', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_b = tf.get_variable('contact_b', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=0.99, stddev=0.00001))
    self.contact_c = tf.get_variable('contact_c', [1, features], tf.float64, initializer=tf.truncated_normal_initializer(mean=1, stddev=0.00001))
#     self.contact_a_broadcast = tf.broadcast_to(self.contact_a, [samples, features])
#     self.contact_b_broadcast = tf.broadcast_to(self.contact_b, [samples, features])
    self.contact_tensor = tf.pow(self.contact_tensor, self.contact_c)
    self.l1 = self.contact_tensor
    self.contact_weighting = tf.exp(tf.maximum(tf.log(self.contact_b) * self.contact_tensor, -10))
    self.l2 = self.contact_weighting
    self.l3 = tf.log(self.contact_b) * self.contact_tensor
    self.contact_weighting = tf.multiply(self.contact_a, self.contact_weighting)
    self.l = self.contact_weighting
    self.universal_weighting = tf.convert_to_tensor(np.array([5], np.float64), dtype=tf.float64)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias_original = tf.get_variable(name='universal_bias', dtype=tf.float64, shape=[1, features], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.02))
    self.universal_bias = tf.broadcast_to(self.universal_bias_original, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) + tf.multiply(self.randomly_selected_contact_input, self.contact_weighting) 
#     self.prediction = self.universal_bias
    self.output = tf.placeholder(tf.float64, shape=[samples, features])
    self.total_loss = tf.reduce_sum(tf.log(tf.maximum(1 - tf.abs(self.output - self.prediction), 0.00000001))) * -1

      
  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.clip_op1 = tf.assign(self.relatedness_b, tf.clip_by_value(self.relatedness_b, 0.01, 0.99999999))
    self.clip_op2 = tf.assign(self.contact_b, tf.clip_by_value(self.contact_b, 0.01, 0.99999999))
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
#         number2 = random.randint(self.samples-2)
        x = self.distance_array[j][number1]
        contact_feed.append([x])
        random_contact_input_feed.append(self.input_array[j][number1])
        
      self.feed = {self.relatedness_placeholder: relatedness_feed, self.contact_placeholder: contact_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array, self.randomly_selected_contact_input: random_contact_input_feed}
      print("After %d iterations:" % i)
      total_loss = self.sess.run(self.total_loss, feed_dict = self.feed)
      print(total_loss)
      if np.isnan(total_loss):
        print(np.sum(self.sess.run(self.relatedness_a, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.relatedness_b, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.relatedness_c, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.contact_a, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.contact_b, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.contact_c, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.prediction, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.relatedness_weighting, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.contact_weighting, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.k, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.l, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.l1, feed_dict = self.feed)))
        print(np.sum(self.sess.run(self.l2, feed_dict = self.feed)))
        print(self.sess.run(self.l2, feed_dict = self.feed))
        print(self.sess.run(self.contact_c))
        print(self.sess.run(self.l, feed_dict = self.feed))

        exit(0)
      
#       print(self.sess.run(self.l1, feed_dict = self.feed)[100])
#       print(self.sess.run(self.l3, feed_dict = self.feed)[100])
      
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op1)
      self.sess.run(self.clip_op2)

      if i % 100 == 0:
        print(self.sess.run(self.relatedness_a, feed_dict = self.feed))
        print(self.sess.run(self.relatedness_b, feed_dict = self.feed))
        print(self.sess.run(self.relatedness_c, feed_dict = self.feed))

        print(self.sess.run(self.contact_a, feed_dict = self.feed))
        print(self.sess.run(self.contact_b, feed_dict = self.feed))
        print(self.sess.run(self.contact_c, feed_dict = self.feed))

        print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed)) 
        print(self.sess.run(self.contact_weighting, feed_dict = self.feed)) 

  def get_weightings(self):
    relatedness_a = self.sess.run(self.relatedness_a)
    relatedness_b = self.sess.run(self.relatedness_b)
    relatedness_c = self.sess.run(self.relatedness_c)
    contact_a = self.sess.run(self.contact_a)
    contact_b = self.sess.run(self.contact_b)
    contact_c = self.sess.run(self.contact_c)
    return relatedness_a, relatedness_b, relatedness_c, contact_a, contact_b, contact_c

  def get_universal_bias(self):
    return self.sess.run(self.universal_bias_original)


#       print(self.sess.run(self.prediction, feed_dict = self.feed))

#   def get_weightings(self):
#     return self.sess.run(self.relatedness_weights, feed_dict = self.feed), self.sess.run(self.contact_weights, feed_dict = self.feed)
# 
#   def get_universal_bias(self):
#     return self.sess.run(self.universal_bias_original, feed_dict = self.feed)


class Model7:
  learn_rate = 0.01
  def __init__(self, samples, features, relatedness_weights, contact_weights, universal_bias):  
    self.sess = tf.Session()
    self.samples = 1
    self.features = features

    self.comparandum1 = tf.placeholder(tf.float64, [samples, features])
    self.comparandum2 = tf.placeholder(tf.float64, [samples, features])
    self.number_of_relatedness_intervals = len(relatedness_weights[0])  
    number_of_relatedness_intervals = self.number_of_relatedness_intervals  
    self.relatedness_comparison = tf.placeholder(tf.float64, [1, number_of_relatedness_intervals])
    self.relatedness_weights = tf.convert_to_tensor(relatedness_weights, tf.float64)
    self.relatedness_weighting = tf.reshape(self.relatedness_weights, [1, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.broadcast_to(self.relatedness_weighting, [samples, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.multiply(self.relatedness_weighting, self.relatedness_comparison)
    self.relatedness_weighting = tf.reduce_sum(self.relatedness_weighting, axis=2)    
    self.number_of_contact_intervals = len(contact_weights[0])
    number_of_contact_intervals = self.number_of_contact_intervals
    self.contact_comparison = tf.placeholder(tf.float64, [samples, number_of_contact_intervals])
    self.contact_weights = tf.convert_to_tensor(contact_weights, dtype=tf.float64)
    self.contact_weighting = tf.reshape(self.contact_weights, [1, features, number_of_contact_intervals])
    self.contact_weighting = tf.broadcast_to(self.contact_weighting, [samples, features, number_of_contact_intervals])
    self.contact_weighting = tf.multiply(self.contact_weighting, self.contact_comparison)
    self.contact_weighting = tf.reduce_sum(self.contact_weighting, axis=2)
    self.universal_weighting = tf.convert_to_tensor(np.array([5], np.float64), dtype=tf.float64)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias_original = tf.convert_to_tensor(universal_bias, tf.float64)
    self.universal_bias = tf.broadcast_to(self.universal_bias_original, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.comparandum1, self.relatedness_weighting) + tf.multiply(self.comparandum1, self.contact_weighting) 
    self.output = self.comparandum2
    self.total_loss = tf.reduce_sum(tf.log(1 - tf.abs(self.output - self.prediction))) * -1

  def infer(self, comparandum1, comparandum2):
    number_of_contact_intervals = self.number_of_contact_intervals
    number_of_relatedness_intervals = self.number_of_relatedness_intervals  
    self.feed = {self.comparandum1: comparandum1, self.comparandum2: comparandum2}
    for i in range(number_of_relatedness_intervals):
      relatedness_comparison = [rep(0, number_of_relatedness_intervals)]
      relatedness_comparison[0][i] = 1
      relatedness_comparison = np.array(relatedness_comparison)
      self.feed[self.relatedness_comparison] = relatedness_comparison
      for j in range(number_of_contact_intervals):
        contact_comparison = [rep(0, number_of_contact_intervals)]
        contact_comparison[0][j] = 1
        contact_comparison = np.array(contact_comparison)
        self.feed[self.contact_comparison] = contact_comparison
        print(i, j)
        print(self.sess.run(self.total_loss, feed_dict = self.feed))

class Model6:
  learn_rate = 0.01
  def __init__(self, samples, features, relatedness_weights, contact_weights, universal_bias, comparandum1, comparandum2):  
    self.sess = tf.Session()
    self.samples = samples
    self.features = features
    self.comparandum1 = tf.convert_to_tensor(comparandum1, tf.float64)
    number_of_relatedness_intervals = len(relatedness_weights[0])    
    self.relatedness_comparison_original = tf.get_variable(name='relatedness', dtype=tf.float64, shape=[samples, number_of_relatedness_intervals],initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))    
    self.relatedness_comparison_totals = tf.reduce_sum(self.relatedness_comparison_original, axis = 1)
    self.relatedness_comparison = self.relatedness_comparison_original / self.relatedness_comparison_totals
    self.relatedness_weights = tf.convert_to_tensor(relatedness_weights, tf.float64)
    self.relatedness_weighting = tf.reshape(self.relatedness_weights, [1, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.broadcast_to(self.relatedness_weighting, [samples, features, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.multiply(self.relatedness_weighting, self.relatedness_comparison)
    self.relatedness_weighting = tf.reduce_sum(self.relatedness_weighting, axis=2)    
    number_of_contact_intervals = len(contact_weights[0])
    self.contact_comparison = tf.get_variable(name='contact', dtype=tf.float64, shape=[samples, number_of_contact_intervals],initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))
    self.contact_weights = tf.convert_to_tensor(contact_weights, dtype=tf.float64)
    self.contact_weighting = tf.reshape(self.contact_weights, [1, features, number_of_contact_intervals])
    self.contact_weighting = tf.broadcast_to(self.contact_weighting, [samples, features, number_of_contact_intervals])
    self.contact_weighting = tf.multiply(self.contact_weighting, self.contact_comparison)
    self.contact_weighting = tf.reduce_sum(self.contact_weighting, axis=2)
    self.universal_weighting = tf.convert_to_tensor(np.array([5], np.float64), dtype=tf.float64)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias_original = tf.convert_to_tensor(universal_bias, tf.float64)
    self.universal_bias = tf.broadcast_to(self.universal_bias_original, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.comparandum1, self.relatedness_weighting) + tf.multiply(self.comparandum1, self.contact_weighting) 
    self.output = tf.convert_to_tensor(comparandum2, tf.float64)
    self.total_loss = tf.reduce_sum(tf.log(1 - tf.abs(self.output - self.prediction))) * -1

  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    self.clip_op1 = tf.assign(self.relatedness_comparison_original, tf.clip_by_value(self.relatedness_comparison, 0, 1))
    self.clip_op2 = tf.assign(self.contact_comparison, tf.clip_by_value(self.contact_comparison, 0, 1))
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    for i in range(steps):
      self.sess.run(self.train_step)
      self.sess.run(self.clip_op1)
      self.sess.run(self.clip_op2)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss))
      print(self.sess.run(self.relatedness_comparison))
      print(self.sess.run(self.contact_comparison))


class Model5:
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

    '''
    i now want a tensor of shape [samples, 1, 2]
    
    '''
    number_of_relatedness_intervals = 6

    self.relatedness_intervals = tf.convert_to_tensor(np.array([0,2,4,6,8,10], np.float64), tf.float64)
    self.relatedness_intervals  = tf.reshape(self.relatedness_intervals, [1, 1, number_of_relatedness_intervals])
    self.relatedness_intervals = tf.broadcast_to(self.relatedness_intervals, [samples, features, number_of_relatedness_intervals]) 
    self.relatedness_tensor = tf.reshape(self.relatedness_placeholder, [samples, 1, 1])
    self.relatedness_tensor = tf.broadcast_to(self.relatedness_tensor, [samples, features, number_of_relatedness_intervals])
    
    '''
    now want to compare them
    '''
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
#     self.total_loss = tf.reduce_sum(tf.abs(self.output - self.prediction))

      
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
#         number2 = random.randint(self.samples-2)
        x = self.distance_array[j][number1]
        contact_feed.append([x])
        random_contact_input_feed.append(self.input_array[j][number1])
        
      self.feed = {self.relatedness_placeholder: relatedness_feed, self.contact_placeholder: contact_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array, self.randomly_selected_contact_input: random_contact_input_feed}
      self.sess.run(self.train_step, feed_dict = self.feed)
      self.sess.run(self.clip_op1)
      self.sess.run(self.clip_op2)
      self.sess.run(self.clip_op3)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_weights, feed_dict = self.feed))
      print(self.sess.run(self.contact_weights, feed_dict = self.feed))

  def get_weightings(self):
    return self.sess.run(self.relatedness_weights, feed_dict = self.feed), self.sess.run(self.contact_weights, feed_dict = self.feed)

  def get_universal_bias(self):
    return self.sess.run(self.universal_bias_original, feed_dict = self.feed)


class Model4:
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

    '''
    i now want a tensor of shape [samples, 1, 2]
    
    '''
    number_of_relatedness_intervals = 6

    self.relatedness_intervals = tf.convert_to_tensor(np.array([0,2,4,6,8,10], np.float64), tf.float64)
    self.relatedness_intervals  = tf.reshape(self.relatedness_intervals, [1, 1, number_of_relatedness_intervals])
    self.relatedness_intervals = tf.broadcast_to(self.relatedness_intervals, [samples, 1, number_of_relatedness_intervals]) 
    self.relatedness_tensor = tf.reshape(self.relatedness_placeholder, [samples, 1, 1])
    self.relatedness_tensor = tf.broadcast_to(self.relatedness_tensor, [samples, 1, number_of_relatedness_intervals])
    
    '''
    now want to compare them
    '''
    self.relatedness_intervals_max = tf.convert_to_tensor(np.array([2,4,6,8,10,1000], np.float64), tf.float64)
    self.relatedness_intervals_max  = tf.reshape(self.relatedness_intervals_max, [1, 1, number_of_relatedness_intervals])
    self.relatedness_intervals_max = tf.broadcast_to(self.relatedness_intervals_max, [samples, 1, number_of_relatedness_intervals]) 
    
    self.relatedness_comparison_a = tf.cast(tf.greater_equal(self.relatedness_tensor, self.relatedness_intervals), tf.float64)
    self.relatedness_comparison_b = tf.cast(tf.less(self.relatedness_tensor, self.relatedness_intervals_max), tf.float64)
    self.relatedness_comparison_c = tf.multiply(self.relatedness_comparison_a, self.relatedness_comparison_b)
    self.relatedness_weights = tf.get_variable(name='relatedness_weighting', dtype=tf.float64, shape=[1, number_of_relatedness_intervals], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_weighting = tf.reshape(self.relatedness_weights, [1, 1, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.broadcast_to(self.relatedness_weighting, [samples, 1, number_of_relatedness_intervals])
    self.relatedness_weighting = tf.multiply(self.relatedness_weighting, self.relatedness_comparison_c)
    self.relatedness_weighting = tf.reduce_sum(self.relatedness_weighting, axis=2)
    
    
    number_of_contact_intervals = 8

    self.contact_intervals = tf.convert_to_tensor(np.array([0,100,200,300,400,500,600,700], np.float64), tf.float64)
    self.contact_intervals  = tf.reshape(self.contact_intervals, [1, 1, number_of_contact_intervals])
    self.contact_intervals = tf.broadcast_to(self.contact_intervals, [samples, 1, number_of_contact_intervals]) 
    self.contact_tensor = tf.reshape(self.contact_placeholder, [samples, 1, 1])
    self.contact_tensor = tf.broadcast_to(self.contact_tensor, [samples, 1, number_of_contact_intervals])


    self.contact_intervals_max = tf.convert_to_tensor(np.array([100,200,300,400,500,600,700,100000], np.float64), tf.float64)
    self.contact_intervals_max  = tf.reshape(self.contact_intervals_max, [1, 1, number_of_contact_intervals])
    self.contact_intervals_max = tf.broadcast_to(self.contact_intervals_max, [samples, 1, number_of_contact_intervals]) 
    
    self.contact_comparison_a = tf.cast(tf.greater_equal(self.contact_tensor, self.contact_intervals), tf.float64)
    self.contact_comparison_b = tf.cast(tf.less(self.contact_tensor, self.contact_intervals_max), tf.float64)
    self.contact_comparison_c = tf.multiply(self.contact_comparison_a, self.contact_comparison_b)
    self.contact_weights = tf.get_variable(name='contact_weighting', dtype=tf.float64, shape=[1, number_of_contact_intervals], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_weighting = tf.reshape(self.contact_weights, [1, 1, number_of_contact_intervals])
    self.contact_weighting = tf.broadcast_to(self.contact_weighting, [samples, 1, number_of_contact_intervals])
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
#     self.total_loss = tf.reduce_sum(tf.abs(self.output - self.prediction))

      
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
      print(self.sess.run(self.universal_weighting, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed))
      print(self.sess.run(self.contact_weighting, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_weights, feed_dict = self.feed))
      print(self.sess.run(self.contact_weights, feed_dict = self.feed))
      print(self.sess.run(self.prediction, feed_dict = self.feed))
      
      


#       print(self.sess.run(self.relatedness_a, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_b, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_c, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.contact_a, feed_dict = self.feed))
#       print(self.sess.run(self.contact_b, feed_dict = self.feed))
#       print(self.sess.run(self.contact_c, feed_dict = self.feed))
#       print(self.sess.run(self.contact_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.j, feed_dict = self.feed))
#       print(self.sess.run(self.k, feed_dict = self.feed))
      
#   def show_relatedness_weighting(self):
#     a = self.sess.run(self.relatedness_a, feed_dict = self.feed)
#     b = self.sess.run(self.relatedness_b, feed_dict = self.feed)
#     c = self.sess.run(self.relatedness_c, feed_dict = self.feed)
#     return a, b, c
# 
#   def show_contact_weighting(self):
#     a = self.sess.run(self.contact_a, feed_dict = self.feed)
#     b = self.sess.run(self.contact_b, feed_dict = self.feed)
#     c = self.sess.run(self.contact_c, feed_dict = self.feed)
#     return a, b, c



class Model3:
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
    self.relatedness_tensor = tf.placeholder(tf.float64, shape=[samples, 1])
    weight_initer = tf.truncated_normal_initializer(mean=20, stddev=1)
    self.relatedness_a = tf.get_variable(name='relatedness_a', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_b = tf.get_variable(name='relatedness_b', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_c = tf.get_variable(name='relatedness_c', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.relatedness_weighting = tf.square(self.relatedness_tensor) * self.relatedness_a
    self.relatedness_weighting = self.relatedness_weighting + (self.relatedness_tensor * self.relatedness_b)
    self.relatedness_weighting = self.relatedness_weighting + self.relatedness_c
    self.relatedness_weighting = 1 / self.relatedness_weighting
    self.j = self.relatedness_weighting
    self.randomly_selected_contact_input = tf.placeholder(tf.float64, [samples, features])
    self.contact_tensor = tf.placeholder(tf.float64, shape=[samples, 1])
    self.contact_a = tf.get_variable(name='contact_a', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_b = tf.get_variable(name='contact_b', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_c = tf.get_variable(name='contact_c', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=5, stddev=1))
    self.contact_weighting = tf.square(self.contact_tensor) * self.contact_a
    self.contact_weighting = self.contact_weighting + (self.contact_tensor * self.contact_b)
    self.contact_weighting = self.contact_weighting + self.contact_c
    self.contact_weighting = 1 / self.contact_weighting
    self.k = self.contact_weighting
    self.universal_weighting = tf.get_variable(name='universal_weighting', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=2, stddev=1))
#     self.universal_weighting = tf.sigmoid(self.universal_weighting)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
#     self.weighting_total = self.relatedness_weighting + self.contact_weighting 

    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
#     self.universal_bias = tf.get_variable(name='universal_bias', dtype=tf.float64, shape=[1, features], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.02))
#     self.universal_bias = tf.broadcast_to(self.universal_bias, [samples, features])
#     self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) + tf.multiply(self.randomly_selected_contact_input, self.contact_weighting) 
    self.prediction = tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) + tf.multiply(self.randomly_selected_contact_input, self.contact_weighting) 

    self.output = tf.placeholder(tf.float64, shape=[samples, features])
#     self.total_loss = tf.reduce_sum(tf.log(tf.minimum(1 - tf.abs(self.output - self.prediction), 0.01))) * -1
#     self.total_loss = tf.reduce_sum(tf.log(1 - tf.abs(self.output - self.prediction))) * -1
    self.total_loss = tf.reduce_sum(tf.abs(self.output - self.prediction))

 
     
  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
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
        
      self.feed = {self.relatedness_tensor: relatedness_feed, self.contact_tensor: contact_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array, self.randomly_selected_contact_input: random_contact_input_feed}
      self.sess.run(self.train_step, feed_dict = self.feed)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
      print(self.sess.run(self.universal_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_a, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_b, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_c, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.contact_a, feed_dict = self.feed))
#       print(self.sess.run(self.contact_b, feed_dict = self.feed))
#       print(self.sess.run(self.contact_c, feed_dict = self.feed))
#       print(self.sess.run(self.contact_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.j, feed_dict = self.feed))
#       print(self.sess.run(self.k, feed_dict = self.feed))
      
  def show_relatedness_weighting(self):
    a = self.sess.run(self.relatedness_a, feed_dict = self.feed)
    b = self.sess.run(self.relatedness_b, feed_dict = self.feed)
    c = self.sess.run(self.relatedness_c, feed_dict = self.feed)
    return a, b, c

  def show_contact_weighting(self):
    a = self.sess.run(self.contact_a, feed_dict = self.feed)
    b = self.sess.run(self.contact_b, feed_dict = self.feed)
    c = self.sess.run(self.contact_c, feed_dict = self.feed)
    return a, b, c












class Model2:
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
    self.relatedness_tensor = tf.placeholder(tf.float64, shape=[samples, 1])
    weight_initer = tf.truncated_normal_initializer(mean=20, stddev=1)
    self.relatedness_a = tf.get_variable(name='relatedness_a', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    self.relatedness_b = tf.get_variable(name='relatedness_b', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    self.relatedness_c = tf.get_variable(name='relatedness_c', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    self.relatedness_weighting = tf.square(self.relatedness_tensor) * self.relatedness_a
    self.relatedness_weighting = self.relatedness_weighting + (self.relatedness_tensor * self.relatedness_b)
    self.relatedness_weighting = self.relatedness_weighting + self.relatedness_c

    self.relatedness_weighting = 1 / self.relatedness_weighting
#     self.relatedness_weighting = tf.sigmoid(self.relatedness_weighting)
    self.k = self.relatedness_weighting
    self.randomly_selected_contact_input = tf.placeholder(tf.float64, [samples, features])
    self.contact_tensor = tf.placeholder(tf.float64, shape=[samples, 1])
    self.contact_a = tf.get_variable(name='contact_a', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.04, stddev=0.01))
    self.contact_b = tf.get_variable(name='contact_b', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.04, stddev=0.01))
    self.contact_c = tf.get_variable(name='contact_c', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.04, stddev=0.01))
    self.contact_weighting = tf.square(self.contact_tensor) * self.contact_a
    self.contact_weighting = self.contact_weighting + (self.contact_tensor * self.contact_b)
    self.contact_weighting = self.contact_weighting + self.contact_c
    self.contact_weighting = 1 / self.contact_weighting
#     self.contact_weighting = tf.sigmoid(self.contact_weighting)    
    self.universal_weighting = tf.get_variable(name='universal_weighting', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.2, stddev=0.01))
#     self.universal_weighting = tf.sigmoid(self.universal_weighting)
    self.weighting_total = self.relatedness_weighting + self.contact_weighting + self.universal_weighting
#     self.weighting_total = self.relatedness_weighting + self.universal_weighting

    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.contact_weighting = self.contact_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias = tf.get_variable(name='universal_bias', dtype=tf.float64, shape=[1, features], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.02))
    self.universal_bias = tf.broadcast_to(self.universal_bias, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) + tf.multiply(self.randomly_selected_contact_input, self.contact_weighting) 
#     self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) 
#     self.prediction = tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting) 
    self.output = tf.placeholder(tf.float64, shape=[samples, features])
    self.total_loss = tf.reduce_sum(tf.log(tf.minimum(1 - tf.abs(self.output - self.prediction), 0.01))) * -1
     
  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    for i in range(steps):
      relatedness_feed = []
      contact_feed = []
      na_contact_feed = []
      random_relatedness_input_feed = []
      random_contact_input_feed = []
      for j in range(self.samples):
        number1 = random.randint(self.samples-2)
        relatedness_feed.append([self.relatedness_array[j][number1]])
        random_relatedness_input_feed.append(self.input_array[j][number1])
        number2 = random.randint(self.samples-2)
        x = self.distance_array[j][number2]
        if np.isnan(x):
          x = 0
          isna = 0
        else:
          isna = 1
        contact_feed.append([x])
        na_contact_feed.append([isna])
        random_contact_input_feed.append(self.input_array[j][number2])
        
      self.feed = {self.relatedness_tensor: relatedness_feed, self.contact_tensor: contact_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array, self.randomly_selected_contact_input: random_contact_input_feed}
      self.sess.run(self.train_step, feed_dict = self.feed)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
#       print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.contact_weighting, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_a, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_b, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_c, feed_dict = self.feed))
      print(self.sess.run(self.k, feed_dict = self.feed))
      print(self.sess.run(self.relatedness_weighting, feed_dict = self.feed))
#       print(self.sess.run(self.contact_a, feed_dict = self.feed))
#       print(self.sess.run(self.contact_b, feed_dict = self.feed))
#       print(self.sess.run(self.contact_c, feed_dict = self.feed))
      print(self.sess.run(self.contact_weighting, feed_dict = self.feed))
# 
#       print(self.sess.run(self.prediction, feed_dict = self.feed))
#       print(self.sess.run(self.universal_bias, feed_dict = self.feed))
      print(self.sess.run(self.universal_weighting, feed_dict = self.feed))
      
        
  def show_relatedness_weighting(self):
    a = self.sess.run(self.relatedness_a, feed_dict = self.feed)
    b = self.sess.run(self.relatedness_b, feed_dict = self.feed)
    c = self.sess.run(self.relatedness_c, feed_dict = self.feed)
    return a, b, c

  def show_contact_weighting(self):
    a = self.sess.run(self.contact_a, feed_dict = self.feed)
    b = self.sess.run(self.contact_b, feed_dict = self.feed)
    c = self.sess.run(self.contact_c, feed_dict = self.feed)
    return a, b, c

class Model1:
  learn_rate = 0.001
  def __init__(self, input_array, output_array, relatedness_array, samples, features):  
    self.sess = tf.Session()
    self.input_array = input_array
    self.output_array = output_array
    self.relatedness_array = relatedness_array
    self.samples = samples
    self.features = features
    self.randomly_selected_relatedness_input = tf.placeholder(tf.float64, [samples, features])
    self.relatedness_tensor = tf.placeholder(tf.float64, shape=[samples, 1])
    weight_initer = tf.truncated_normal_initializer(mean=20, stddev=1)
    self.relatedness_a = tf.get_variable(name='relatedness_a', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=2, stddev=1))
    self.relatedness_b = tf.get_variable(name='relatedness_b', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=2, stddev=1))
    self.relatedness_c = tf.get_variable(name='relatedness_c', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=2, stddev=1))
    self.relatedness_weighting = tf.square(self.relatedness_tensor) * self.relatedness_a
    self.relatedness_weighting = self.relatedness_weighting + (self.relatedness_tensor * self.relatedness_b)
    self.relatedness_weighting = self.relatedness_weighting + self.relatedness_c
    self.relatedness_weighting = 1 / self.relatedness_weighting
    self.universal_weighting = tf.get_variable(name='universal_weighting', dtype=tf.float64, shape=[1], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01))
    self.weighting_total = self.relatedness_weighting + self.universal_weighting
    self.relatedness_weighting = self.relatedness_weighting / self.weighting_total
    self.universal_weighting = self.universal_weighting / self.weighting_total
    self.universal_bias = tf.get_variable(name='universal_bias', dtype=tf.float64, shape=[1, features], initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.02))
    self.universal_bias = tf.broadcast_to(self.universal_bias, [samples, features])
    self.prediction = tf.multiply(self.universal_weighting, self.universal_bias) + tf.multiply(self.randomly_selected_relatedness_input, self.relatedness_weighting)
    self.output = tf.placeholder(tf.float64, shape=[samples, features])
    self.total_loss = tf.reduce_sum(tf.log(1 - tf.abs(self.output - self.prediction))) * -1
    
  def train(self, steps=100): 
    self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)  
    init = tf.initialize_all_variables()
    self.sess.run(init)   
    for i in range(steps):
      relatedness_feed = []
      random_relatedness_input_feed = []
      for j in range(self.samples):
        number = random.randint(self.samples-1)
        relatedness_feed.append([self.relatedness_array[j][number]])
        random_relatedness_input_feed.append(self.input_array[j][number])
      self.feed = {self.relatedness_tensor: relatedness_feed, self.randomly_selected_relatedness_input: random_relatedness_input_feed, self.output: self.output_array}
      self.sess.run(self.train_step, feed_dict = self.feed)
      print("After %d iterations:" % i)
      print(self.sess.run(self.total_loss, feed_dict = self.feed))
      print(self.sess.run(self.prediction, feed_dict = self.feed))
      print(self.sess.run(self.universal_bias, feed_dict = self.feed))
      print(self.sess.run(self.universal_weighting, feed_dict = self.feed))
      
        
  def show_weighting(self):
    a = self.sess.run(self.relatedness_a, feed_dict = self.feed)
    b = self.sess.run(self.relatedness_b, feed_dict = self.feed)
    c = self.sess.run(self.relatedness_c, feed_dict = self.feed)
    return a, b, c
    
    
