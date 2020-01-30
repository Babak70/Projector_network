from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import eye_input
import eye_nn_arch_dict as Arch

HEIGHT=eye_input.IMAGE_SIZE
WIDTH=eye_input.IMAGE_SIZE
HEIGHT_label=eye_input.IMAGE_SIZE_label
WIDTH_label=eye_input.IMAGE_SIZE_label
M=HEIGHT


NUMEVAL_SAMPLES=eye_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size_eval',NUMEVAL_SAMPLES,
                            """Number of images to process in a batch of eval == entire data.""")
tf.app.flags.DEFINE_string('data_dir', './fiber10_data',
                           """Path to the CIFAR-10 data directory.""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _activation_summary(x):
  tf.summary.histogram(x.op.name + '/activations', x)
  tf.summary.scalar(x.op.name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):

  with tf.device('/cpu:0'):
#    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype=tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):

  dtype=tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs(NUMTRAIN_SAMPLES):

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')

#  data_dir = os.path.join(FLAGS.data_dir, 'fiber-10-batches-bin')
  data_dir=FLAGS.data_dir
  images, labels = eye_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size,NUMTRAIN_SAMPLES=NUMTRAIN_SAMPLES)
  return images, labels


def inputs(eval_data):

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir=FLAGS.data_dir
  images, labels = eye_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size_eval)

  return images, labels

def corr(logits,labels):   
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  n1=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  dn1=tf.sqrt((tf.reduce_mean(tf.squared_difference(logits, m1))))
  dn2=tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, m2)))
  return tf.divide(n1,dn1*dn2,name="correlation2D")

def corr_train(logits,labels):   
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  n1=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  dn1=tf.sqrt((tf.reduce_mean(tf.squared_difference(logits, m1))))
  dn2=tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, m2)))
  return tf.divide(n1,dn1*dn2,name="correlation2D")



def ssim(logits,labels):   
    
  m1=tf.reduce_mean(logits)
  m2=tf.reduce_mean(labels)
  cov=tf.reduce_mean(tf.multiply(tf.subtract(logits,m1),tf.subtract(labels,m2)))
  sig1=(tf.reduce_mean(tf.squared_difference(logits, m1)))
  sig2=(tf.reduce_mean(tf.squared_difference(labels, m2)))
  
  t1=tf.multiply(2.0,tf.multiply(m1,m2))+0.0001
  t2=tf.multiply(2.0,cov)+0.0009
  t3=tf.square(m1)+tf.square(m2)+0.0001  
  t4=sig1+sig2+0.0009
  
  ssim_val=tf.divide(tf.multiply(t1,t2),tf.multiply(t3,t4))

  return ssim_val

def inference(features,is_training):
  
  tf.summary.image('features', features,max_outputs=3)    
  with tf.variable_scope('generator'): 
      
       mOut1=tf.reshape(features,[-1, HEIGHT*WIDTH])
       mOut2_1=Arch.fully1(mOut1,HEIGHT,HEIGHT_label,is_training)
       mOut2=tf.nn.sigmoid(mOut2_1)       
       mOut4=tf.reshape(mOut2,[-1, HEIGHT_label,WIDTH_label,1]) 
       return mOut4
     

def inference_1(features,is_training):
  
  tf.summary.image('features_s', features,max_outputs=3)   
  with tf.variable_scope('generator_s'): 
      
       mOut1=tf.reshape(features,[-1, HEIGHT*WIDTH])
       mOut2_1=Arch.fully1(mOut1,HEIGHT,HEIGHT_label,is_training)
       mOut2=tf.nn.sigmoid(mOut2_1)                
       mOut4=tf.reshape(mOut2,[-1, HEIGHT_label,WIDTH_label,1])  
       return mOut4


    
def inference_D(features,is_training):
 
  tf.summary.image('features_D', features,max_outputs=3)
  with tf.variable_scope('discriminator'):
      
      mOut1=tf.reshape(features,[-1, HEIGHT_label*WIDTH_label])          
      mOut2_1=Arch.fully1(mOut1,HEIGHT_label,HEIGHT,is_training)
      mOut2=tf.nn.sigmoid(mOut2_1)         
      mOut4=(tf.reshape(mOut2,[-1, HEIGHT,WIDTH,1]))
      return mOut4




def inference_D_1(features,is_training):

  
  tf.summary.image('features_D_s', features,max_outputs=3)
  with tf.variable_scope('discriminator_s'):
      
      mOut1=tf.reshape(features,[-1, HEIGHT_label*WIDTH_label])            
      mOut2_1=Arch.fully1(mOut1,HEIGHT_label,HEIGHT,is_training)
      mOut2=tf.nn.sigmoid(mOut2_1)               
      mOut4=(tf.reshape(mOut2,[-1, HEIGHT,WIDTH,1]))
      return mOut4

def loss(logits, labels):

#  labels = tf.cast(labels, tf.int64)
  loss_mse = tf.reduce_mean(tf.squared_difference(logits, labels,name="sqd"),name="MSE")
  tf.add_to_collection('losses', loss_mse)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')



def _add_loss_summaries(total_loss):

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + '_raw', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step,training_scope,NUMTRAIN_SAMPLES):

  # Variables that affect learning rate.
  num_batches_per_epoch = NUMTRAIN_SAMPLES / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
   
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
   
     opt=tf.train.AdamOptimizer(0.0001)
     apply_gradient_op = opt.minimize(total_loss,global_step=global_step,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=training_scope))


  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)


  return apply_gradient_op
  