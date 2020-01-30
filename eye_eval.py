from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import numpy as np
import tensorflow as tf
import matlab.engine



import eye


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './fiber10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './fiber10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30 * 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once',True,
                         """Whether to run eval only once.""")

from eye_input import HEIGHT
from eye_input import WIDTH

from eye_input import HEIGHT_label
from eye_input import WIDTH_label

from eye_input import NUMEVAL_SAMPLES



def eval_once(saver, summary_writer, loss_eval,logits_1,logits,logits_G_1,logits_G,images_1,summary_op,ccount):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      res_out_images=sess.run(tf.image.convert_image_dtype(images_1, tf.uint8,saturate=True))
      rr_images=np.reshape(res_out_images, [HEIGHT,WIDTH,NUMEVAL_SAMPLES])




      res_out=sess.run(logits)
      rr=np.reshape(res_out, [HEIGHT_label,WIDTH_label,NUMEVAL_SAMPLES])
      rr2=rr*255
      rr3=rr2.astype(np.uint8)      
      res_out_G=sess.run(tf.image.convert_image_dtype(logits_G, tf.uint8,saturate=True))
      rr_G=np.reshape(res_out_G, [HEIGHT,WIDTH,NUMEVAL_SAMPLES])
         

      res_out_1=sess.run(logits_1)
      rr_1=np.reshape(res_out_1, [HEIGHT_label,WIDTH_label,NUMEVAL_SAMPLES])
      rr2_1=rr_1*255
      rr3_1=rr2_1.astype(np.uint8)
      res_out_G_1=sess.run(tf.image.convert_image_dtype(logits_G_1, tf.uint8,saturate=True))
      rr_G_1=np.reshape(res_out_G_1, [HEIGHT,WIDTH,NUMEVAL_SAMPLES])

      print(sess.run(loss_eval))
      print(rr.dtype)
      print(rr.shape) 
      
      
      rr3.tofile("./fiber10_matlab/Output_test_phases_{}.bin".format(int(ccount)))
      rr_G.tofile("./fiber10_matlab/Output_test_phases_G_{}.bin".format(int(ccount)))
      rr_images.tofile("./fiber10_matlab/eval_dataF_{}.bin".format(int(ccount)))            
      rr3_1.tofile("./fiber10_matlab/Output_test_phases_sin_{}.bin".format(int(ccount)))
      rr_G_1.tofile("./fiber10_matlab/Output_test_phases_G_sin_{}.bin".format(int(ccount)))
      
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
#      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def write_files(str1,str2,str3,str4):
    
   
   eval_dataR=np.fromfile(str1,dtype=np.uint8)
   eval_data=np.reshape(eval_dataR, [NUMEVAL_SAMPLES,HEIGHT*WIDTH])
   eval_labelsR=np.fromfile(str2,dtype=np.uint8)
   eval_labels=np.reshape(eval_labelsR, [NUMEVAL_SAMPLES,HEIGHT_label*WIDTH_label])   
   eval_data2=np.reshape(np.fromfile(str3,dtype=np.uint8),[NUMEVAL_SAMPLES,HEIGHT*WIDTH])
   eval_labels2=np.reshape(np.fromfile(str4,dtype=np.uint8),[NUMEVAL_SAMPLES,HEIGHT_label*WIDTH_label])
   
      
   outdata_eval0 = np.concatenate((eval_labels,eval_labels2), axis = 1)
   outdata_eval1 = np.concatenate((eval_data,eval_data2), axis = 1)
   outdata_eval2 = np.concatenate((outdata_eval0,outdata_eval1), axis = 1)
   
   
   outdata_eval2.tofile("./fiber10_data/fiber_test_data_3.bin")


def evaluate(str1,str2,str3,str4,ccount):
    

  write_files(str1,str2,str3,str4)

  with tf.Graph().as_default() as g:
    eval_data = FLAGS.eval_data == 'test' 
    images_0, labels_0 = eye.inputs(eval_data=eval_data)
      
    labels,labels_1=tf.split(labels_0, num_or_size_splits=2,axis=3)
    images,images_1=tf.split(images_0, num_or_size_splits=2,axis=3)
    
    tf.summary.image('eval_labels',labels,max_outputs=3)
    tf.summary.image('eval_images', images,max_outputs=3)
    tf.summary.image('eval_images_1', images_1,max_outputs=3)
    



    with tf.variable_scope('G') as scope:    
     logits = eye.inference(images_1,False)
    
    with tf.variable_scope('D') as scope:        
     logits_D = eye.inference_D(labels,False)
     scope.reuse_variables()
     logits_G = eye.inference_D(logits,False)
        
     
    with tf.variable_scope('G_s') as scope:
     logits_1 = eye.inference_1(images_1,False)
    
    with tf.variable_scope('D_s') as scope:       
     logits_D_1 = eye.inference_D_1(labels_1,False)
     scope.reuse_variables()
     logits_G_1 = eye.inference_D_1(logits_1,False)
    
    

    


    tf.summary.image('eval_logits', logits,max_outputs=3)
    tf.summary.image('eval_logits_D', logits_D,max_outputs=3)
    tf.summary.image('eval_logits_G', logits_G,max_outputs=3)   
    tf.summary.image('eval_logits_s', logits_1,max_outputs=3)
    tf.summary.image('eval_logits_D_s', logits_D_1,max_outputs=3)
    tf.summary.image('eval_logits_G_s', logits_G_1,max_outputs=3)
    
    
    

    loss_eval=tf.multiply(tf.log(tf.divide(1+eye.ssim(logits_G,images_1),2.0)),-1.0)
    loss_eval_D=tf.reduce_mean(tf.squared_difference(logits_D, images))   
    loss_eval_1=tf.multiply(tf.log(tf.divide(1+eye.ssim(logits_G_1,images_1),2.0)),-1.0)
    loss_eval_D_1=tf.reduce_mean(tf.squared_difference(logits_D_1, images))

    
    tf.summary.scalar('loss',loss_eval)
    tf.summary.scalar('loss_D',loss_eval_D)
    tf.summary.scalar('loss_1',loss_eval_1)
    tf.summary.scalar('loss_D_1',loss_eval_D_1)
        
    r2=eye.corr(logits_G,images_1)
    r2_D=eye.corr(logits_D,images)
    ssim=eye.ssim(logits_G,images_1)
    
    tf.summary.scalar('corr_eval', r2)
    tf.summary.scalar('corr_eval_D', r2_D)
    tf.summary.scalar('ssim_eval_D', ssim)
        
    r2_1=eye.corr(logits_G_1,images_1)
    r2_D_1=eye.corr(logits_D_1,images)
    ssim_1=eye.ssim(logits_G_1,images_1)
    
    tf.summary.scalar('corr_eval_s', r2_1)
    tf.summary.scalar('corr_eval_D_s', r2_D_1)
    tf.summary.scalar('ssim_eval_D_s', ssim_1)

    saver = tf.train.Saver()
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g,flush_secs=7200)

    while True:
      eval_once(saver, summary_writer, loss_eval,logits_1,logits,logits_G_1,logits_G,images_1,summary_op,ccount)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def my_eval(NUMTRAIN_SAMPLES,NUMEVAL_SAMPLES,Replace_and_Add):


#  while True:
      print('111111111111111111111111111111111111111111')
      
      for i in range(20):
        
        str1='./dividing/eval_dataF_{}.bin'.format(int(i+1))
        str2='./dividing/eval_labelsF_{}.bin'.format(int(i+1))
        str3='./dividing/eval_dataFf_{}.bin'.format(int(i+1))
        str4='./dividing/eval_labelsFf_{}.bin'.format(int(i+1))
        
        evaluate(str1,str2,str3,str4,i+1)
               
      eng = matlab.engine.start_matlab()
      eng.addpath("./fiber10_matlab")            
      eng.eval('eval_divide_collect_TM',nargout=0)        
      eng.quit()
      return


def main(argv=None):  # pylint: disable=unused-argument
     
      for i in range(1):         
        str1='./eval_dataF.bin'
        str2='./eval_labelsF.bin'
        str3='./eval_dataF.bin'
        str4='./eval_labelsF.bin'
        
        evaluate(str1,str2,str3,str4,i+1)
      eng = matlab.engine.start_matlab()
      eng.addpath("./fiber10_matlab")            
      eng.eval('eval_divide_collect_TM_demo',nargout=0)        
      eng.quit()
      


if __name__ == '__main__':
 tf.app.run()