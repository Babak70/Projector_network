import tensorflow as tf


def fully1(input_images,sizA,sizB,is_training):

   weights = tf.get_variable("weights", [sizA**2,sizB**2],
        initializer=tf.truncated_normal_initializer(stddev=0.05),trainable=is_training)
  
   biases = tf.get_variable("biases", [sizB**2],
        initializer=tf.constant_initializer(0.0),trainable=is_training)


   
   out1 =(tf.matmul(input_images,weights))
        
   relu1 = (out1+biases)
   


   return (relu1)

def fully2(input_images,sizA,sizB,is_training):

   weights = tf.get_variable("weights2", [sizA**2,sizB**2],
        initializer=tf.truncated_normal_initializer(stddev=0.05),trainable=is_training)
  
   biases = tf.get_variable("biases2", [sizB**2],
        initializer=tf.constant_initializer(0.0),trainable=is_training)


   
   out1 =(tf.matmul(input_images,weights))
        
   relu1 = (out1+biases)
   

   return (relu1)