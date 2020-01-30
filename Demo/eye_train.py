from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import time
import numpy as np
import tensorflow as tf
import matlab.engine



import eye
from eye_input import HEIGHT
from eye_input import WIDTH
from eye_input import HEIGHT_label
from eye_input import WIDTH_label
from eye_input import NUMEVAL_SAMPLES 
import eye_eval 


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './fiber10_train',
                           """Directory where to write event logs """
                           """and checkpoint000.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 15,
                            """How often to log results to the console.""")


def write_files(NUMTRAIN_SAMPLES):
#    
   train_dataR=np.fromfile('train_dataF.bin',dtype=np.uint8)
   train_data=np.reshape(train_dataR, [NUMTRAIN_SAMPLES,HEIGHT*WIDTH])
   train_data.tofile("./fiber10_matlab/train_dataF.bin")
   train_labelsR=np.fromfile('train_labelsF.bin',dtype=np.uint8)
   train_labels=np.reshape(train_labelsR, [NUMTRAIN_SAMPLES,HEIGHT_label*WIDTH_label])
   train_labels.tofile("./fiber10_matlab/train_labelsF.bin")
   train_data2=np.reshape(np.fromfile('train_dataF_1.bin',dtype=np.uint8),[NUMTRAIN_SAMPLES,HEIGHT*WIDTH])
   train_labels2=np.reshape(np.fromfile('train_labelsF_1.bin',dtype=np.uint8),[NUMTRAIN_SAMPLES,HEIGHT_label*WIDTH_label])
#      
   outdata_train0 = np.concatenate((train_labels,train_labels2), axis = 1)
   outdata_train1 = np.concatenate((train_data,train_data2), axis = 1)
   outdata_train2 = np.concatenate((outdata_train0,outdata_train1), axis = 1)
#   
   outdata_train2.tofile("./fiber10_data/fiber_train_data_3.bin")

def train(step0,step1,step2,step3,NUMTRAIN_SAMPLES):
  """Train for a number of steps."""
    
  write_files(NUMTRAIN_SAMPLES)
  with tf.Graph().as_default():          
    global_step = tf.train.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images_0, labels_0 = eye.distorted_inputs(NUMTRAIN_SAMPLES)


    labels0=tf.split(labels_0, num_or_size_splits=2,axis=3)
    images0=tf.split(images_0, num_or_size_splits=2,axis=3)
    labels=labels0[0]
    labels_1=labels0[1]    
    images=images0[0]
    images_1=images0[1]
  
    tf.summary.image('labels', labels,max_outputs=3)
    tf.summary.image('labels_1', labels_1,max_outputs=3)  
    tf.summary.image('images', images,max_outputs=3)
    tf.summary.image('images_1', images_1,max_outputs=3)

#real part sub_networks  
    with tf.variable_scope('G') as scope:   
     logits = eye.inference(images_1,True)
    
    with tf.variable_scope('D') as scope:       
     logits_D = eye.inference_D(labels,True)
     scope.reuse_variables()
     logits_G = eye.inference_D(logits,True)     
     
#imaginary part sub_networks    
    with tf.variable_scope('G_s') as scope:    
     logits_1 = eye.inference_1(images_1,True)
    
    with tf.variable_scope('D_s') as scope:       
     logits_D_1 = eye.inference_D_1(labels_1,True)
     scope.reuse_variables()
     logits_G_1 = eye.inference_D_1(logits_1,True)
     
     
        
    


    
    tf.summary.image('logits', logits)
    tf.summary.image('logits_D', logits_D)
    r_D=eye.corr(logits_D,images)
    tf.summary.scalar('corr_train_D', r_D)
    tf.summary.image('logits_G', logits_G)
    r_G=eye.corr(logits_G,images_1)
    tf.summary.scalar('corr_train_G', r_G)
       
    tf.summary.image('logits_s', logits_1)
    tf.summary.image('logits_D_s', logits_D_1)
    r_D_s=eye.corr(logits_D_1,images)
    tf.summary.scalar('corr_train_D_s', r_D_s)
    tf.summary.image('logits_G_s', logits_G_1)
    r_G_s=eye.corr(logits_G_1,images_1)
    tf.summary.scalar('corr_train_G_s', r_G_s) 
    
#    constructing losses
    loss=tf.multiply(tf.log(tf.divide(1.0+eye.corr_train(logits_G,images_1),2.0)),-1.0)
    loss_D=(tf.reduce_mean(tf.squared_difference(logits_D, images)))
    
    
    loss_1=tf.multiply(tf.log(tf.divide(1.0+eye.corr_train(logits_G_1,images_1),2.0)),-1.0)
    loss_D_1=(tf.reduce_mean(tf.squared_difference(logits_D_1, images)))
    
   
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('loss_D',loss_D)
    tf.summary.scalar('loss_1',loss_1)
    tf.summary.scalar('loss_D_1',loss_D_1)

        
    train_op_D=eye.train(loss_D, global_step,'D/discriminator',NUMTRAIN_SAMPLES)
    train_op=eye.train(loss, global_step,'G/generator',NUMTRAIN_SAMPLES)  
    train_op_D_1=eye.train(loss_D_1, global_step,'D_s/discriminator_s',NUMTRAIN_SAMPLES)    
    train_op_1=eye.train(loss_1, global_step,'G_s/generator_s',NUMTRAIN_SAMPLES)

      
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    scaffold=tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=10000000000000000000000000),
               tf.train.NanTensorHook(loss_D),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.8)),scaffold=scaffold,save_checkpoint_steps=1000) as mon_sess:
      while not mon_sess.should_stop():
     
        
       GS=mon_sess.run(global_step)       
       
       if  (GS <=step0):     
            mon_sess.run(train_op_D)
            print(GS)
            print('Trainin D_real') 

       elif (GS >step0) and (GS<=step1):                   
           mon_sess.run(train_op)
           print(GS)
           print('Trainin G_real') 
           
           
       elif (GS >step1) and (GS<=step2):          
            mon_sess.run(train_op_D_1)
            print(GS)
            print('Trainin D_imag')
            
       elif (GS >step2) and (GS<=step3):                    
           mon_sess.run(train_op_1)
           print(GS)
           print('Trainin G_imag')        
       else:
           break
       
        
 

def main(argv=None):  

    """
    Central control, trainin, preparing found solutions to be picked up by matlab_script  
    """
    number_of_iterations=1   
    NUMTRAIN_SAMPLES0=20000 
    countee=2
    sum_steps=40000
    Replace_and_Add=0
    if Replace_and_Add==1:
         NUMTRAIN_SAMPLES=NUMTRAIN_SAMPLES0*(countee)            
    else:
         NUMTRAIN_SAMPLES=NUMTRAIN_SAMPLES0
         
     
    eng = matlab.engine.start_matlab()
    eng.addpath("./dividing")            
    eng.eval('eval_divide',nargout=0)        
    eng.quit()     
         
         
         
    
    for ij in range(number_of_iterations):     
    
     if countee==1:
         
      print(countee)
      step0m=10000
      step1m=10000+step0m
      step2m=10000
      step3m=10000+step2m
      step0=step1m*(countee-1)+step0m+(countee-1)*step3m
      step1=step1m*(countee)+(countee-1)*step3m
      step2=step1+step2m
      step3=step1+step3m     
     
     else:

      print(countee)        
      step0m=2500
      step1m=2500+step0m
      step2m=2500
      step3m=2500+step2m        
      step0=step1m*(countee-2)+step0m+(countee-2)*step3m+sum_steps
      step1=step1m*(countee-1)+(countee-2)*step3m+sum_steps
      step2=step1+step2m
      step3=step1+step3m

                              
     countee+=1
     train(step0,step1,step2,step3,NUMTRAIN_SAMPLES)          
     print('done one round of trainig, now prepare solutions to be sent to matlab')
     eye_eval.my_eval(NUMTRAIN_SAMPLES,NUMEVAL_SAMPLES,Replace_and_Add)
     

    
    
if __name__ == '__main__':
 tf.app.run()