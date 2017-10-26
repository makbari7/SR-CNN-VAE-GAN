from __future__ import division
import os
import time
import math
from glob import glob

# import prettytensor as pt
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope

import numpy as np
from scipy.stats import threshold
from six.moves import xrange

# used for music (e.g., midi) stuff
#import magenta
import pretty_midi
from music21 import *
import mido
import reverse_pianoroll

# files
import glob
import shutil
from os import listdir
from os.path import isfile, join
from shutil import copyfile

from ops import *
from utils import *

barTime = 2. # 2sec (fixed)

stride=1
filterH=3
filterW=3

FS = 8 # 1/FS=0.125 sec per pitch

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class VAEGAN(object):
  def __init__(self, sess, input_height=88, input_width=int(barTime * FS), batch_size=64, sample_num = 3, output_height=88, output_width=int(barTime * FS),
          z_dim=128, gf_dim=8, df_dim=8,
         gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='Nottingham',
         input_fname_pattern='*.mid', checkpoint_dir='./checkpoint', sample_dir='music'):
    
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.      
      z_dim: (optional) Dimension of dim for Z. [128]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [8]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [8]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of input.
    """

    self.sess = sess    

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width # int(barTime * FS) = 16
    self.output_height = output_height
    self.output_width = output_width # int(barTime * FS) = 16

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.qf_dim = df_dim # encoder

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.q_bn0 = batch_norm(name='q_bn0')
    self.q_bn1 = batch_norm(name='q_bn1')
    self.q_bn2 = batch_norm(name='q_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if self.dataset_name == 'Nottingham':
      self.data_X, self.data_Xp, self.data_y = self.load_Nottingham()
      self.c_dim = self.data_X[0].shape[-1]

    self.build_model()

  def build_model(self):

    input_dims = [self.input_height, self.input_width, self.c_dim]
    
    # encoder
    self.x0 = tf.placeholder(
      tf.float32, [self.batch_size] + input_dims, name='real_inputsP')

    self.x0test = tf.placeholder(
      tf.float32, [1] + input_dims, name='real_inputsPtest')

    self.x = tf.placeholder(
      tf.float32, [self.batch_size] + input_dims, name='real_inputs')
    self.sample_x = tf.placeholder(
      tf.float32, [self.sample_num] + input_dims, name='sample_x')
    
    # encoder
    x0 = self.x0

    x = self.x
    sample_x = self.sample_x

    self.zp = tf.placeholder(tf.float32, [None, self.z_dim], name='zp')    

    ### Encoder
    self.z_mean, self.z_log_sigma_sq = self.Encoder(x0)
    eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
    self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

    self.x0_tilde = self.generator(self.z)
    self.xp_tilde = self.generator(self.zp, reuse=True)

    print(self.x0_tilde.shape)
    print(self.xp_tilde.shape)
    raw_input('shape')

    self.Dx, self.Dx_logits = self.discriminator(x)      
    self.Dx0, self.Dx0_logits = self.discriminator(x0, reuse=True)      

    self.Dx0_tilde, self.Dx0_logits_tilde = self.discriminator(self.x0_tilde, reuse=True)
    self.Dxp_tilde, self.Dxp_logits_tilde = self.discriminator(self.xp_tilde, reuse=True)
    
    self.sampler = self.sampler(self.zp) # zp: will be filled later with random noise

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    ### VAE loss

    #KL_loss / Lprior
    self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

    # Lth Layer Loss - the 'learned similarity measure'  

    self.LL_loss = 0.5 * (      
      tf.reduce_sum(tf.square(self.Dx_logits - self.Dx0_logits_tilde)) #/ (self.input_width*self.input_height)
      )
    
    self.vae_loss = tf.reduce_mean(self.latent_loss + self.LL_loss) / (self.input_width*self.input_height*self.c_dim)

    ### GAN loss
    self.d_loss_real = 0.5 * ( 
      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dx_logits, tf.ones_like(self.Dx)))
    )

    self.d_loss_fake = 0.5 * (
      tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dx0_logits_tilde, tf.zeros_like(self.Dx0_tilde)))
      +tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dxp_logits_tilde, tf.zeros_like(self.Dxp_tilde)))
      )

    self.d_loss = self.d_loss_real + self.d_loss_fake 

    self.g_loss = (0.5 * ( tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dx0_logits_tilde, tf.ones_like(self.Dx0_tilde)))
      +tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dxp_logits_tilde, tf.ones_like(self.Dxp_tilde))))
      +tf.reduce_mean(self.LL_loss / (self.input_width*self.input_height*self.c_dim)))
 
    t_vars = tf.trainable_variables()

    self.q_vars = [var for var in t_vars if 'q_' in var.name]
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]    
    self.vae_vars = self.q_vars+self.g_vars

    self.saver = tf.train.Saver()  

  def train(self, config):
    lr_E = tf.placeholder(tf.float32, shape=[])
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])    

    vae_optim = tf.train.AdamOptimizer(lr_E, beta1=config.beta1) \
              .minimize(self.vae_loss, var_list=self.vae_vars)
    d_optim = tf.train.AdamOptimizer(lr_D, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(lr_G, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):                

      batch_idxs = min(len(self.data_X), config.train_size) // self.batch_size

      for idx in xrange(0, batch_idxs):

        # learning rates
        g_current_lr = 0.0005        
        d_current_lr = 0.0001        
        e_current_lr = 0.0005
                
        batch_z = np.random.normal(0, 1, size=(self.batch_size , self.z_dim))
        batch_inputs = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_inputsP = self.data_Xp[idx*self.batch_size:(idx+1)*self.batch_size]        

        # Update VAE          
        for i in range(2):
          _, summary_str = self.sess.run([vae_optim, self.vae_loss],
            feed_dict={ lr_E: e_current_lr, self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z })
          errVAE = self.vae_loss.eval({ self.x: batch_inputs,self.x0: batch_inputsP, self.zp: batch_z })

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_loss],
          feed_dict={ lr_D: d_current_lr, self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z })
        errD_fake = self.d_loss_fake.eval({ self.x0: batch_inputsP, self.zp: batch_z })
        errD_real = self.d_loss_real.eval({ self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z })
        errD = errD_fake + errD_real

        # Update G network
        for i in range(2):
          _, summary_str = self.sess.run([g_optim, self.g_loss],
            feed_dict={ lr_G: g_current_lr, self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z })
          errG = self.g_loss.eval({ self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z })

        counter += 1
        print("Learning rates: [E: %.8f] [D: %.8f] [G: %.8f]" \
          % (e_current_lr, d_current_lr, g_current_lr))
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, vae_loss: %.8f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errVAE, errD, errG))

        if np.mod(counter, 100) == 0:
          self.generateSamples(sample_dir=config.sample_dir, epoch=epoch, idx=idx)
        if np.mod(counter, 500) == 0:
          self.save(config.checkpoint_dir, counter)

  def test(self, config):
    self.generateSamples(sample_dir=config.sample_dir)

  def generateSamples(self, sample_dir, epoch=0, idx=0):                
    Encoder = self.Encoder(self.x0test,reuse=True, batch_size=1, train=False)    

    for n in range(self.sample_num):
      # get a random sample to start the music with.
      sampleIndex = random.randint(1,self.data_Xp.shape[0])
      x0 = self.data_Xp[sampleIndex:sampleIndex+1]
      if(x0.shape[0]>0):
        music = None

        # saving the generated midi files!
        for i in range(5):              
          if(x0.shape[0]>0):
            bar = np.zeros(shape=[128, self.input_width, self.c_dim])              
            x0 = np.clip(x0,0,1) # force to be [0-1]
            bar[33:121,:,:] = x0[0,:,:,:]*127.0 # [0-127]                          
            bar = bar.astype(int)  

            '''
            # threshdoling (to avoid low values): threshold = average of the nonzero values             
            nonzeroSample = bar[np.nonzero(bar)];
            if(len(nonzeroSample)>0):
              avg = sum(nonzeroSample) / len(nonzeroSample)                      
              bar[bar<avg]=0; 
            '''          

            if music is None:
              music = bar
            else:
              music = np.concatenate((music, bar), axis=1)

            z_mean, z_log_sigma_sq = self.sess.run(Encoder, feed_dict={ self.x0test: x0 })
            eps = np.random.normal(0, 1, size=(1 , self.z_dim))
            sample_z = z_mean + np.sqrt(np.exp(z_log_sigma_sq)) * eps
            sample = self.sess.run(self.sampler, feed_dict={ self.zp: sample_z })
            x0 = sample # the previous generated sample will be the input to the Encoder
                
        if(np.amax(music)>0): # ignore the empty midi samples
          print("\n[Sample]\n")
          des_midi = reverse_pianoroll.piano_roll_to_pretty_midi(music,fs=FS, program=0);        
          des_midi.write(sample_dir+'/train_'+str(epoch)+'_'+str(idx)+'_s'+str(n)+'.mid')            

  def generateSamplesNoise(self, sample_dir, epoch, idx):
    # a placeholder for the random noise for the sampler!            
    Encoder = self.Encoder(self.x0test,reuse=True, batch_size=1, train=False)    

    for n in range(self.sample_num):
      sample_z = np.random.normal(0, 1, size=(1 , self.z_dim))      

      sample = self.sess.run(self.sampler,feed_dict={ self.zp: sample_z })

      music = None

      # saving the generated midi files!
      for i in range(5):             
        bar = np.zeros(shape=[128, self.input_width])              
        sample = np.clip(sample,0,1) # force to be [0-1]
        bar[33:121,:] = sample[0,:,:,0]*127.0 # [0-127]                          
        bar = bar.astype(int)            

        # threshdoling (to avoid low values): threshold = average of the nonzero values             
        #nonzeroSample = sample[np.nonzero(sample)];
        #avg = sum(nonzeroSample) / len(nonzeroSample)                      
        #sample[sample<avg]=0;        

        if music is None:
          music = bar
        else:
          music = np.concatenate((music, bar), axis=1)

        ################
        x0 = sample # the previous generated sample will be the input to the Encoder
        z_mean, z_log_sigma_sq = self.sess.run(Encoder, feed_dict={ self.x0test: x0 })
        eps = np.random.normal(0, 1, size=(1 , self.z_dim))
        sample_z = z_mean + np.sqrt(np.exp(z_log_sigma_sq)) * eps

        sample = self.sess.run(self.sampler,feed_dict={ self.zp: sample_z })
        
      if(np.amax(music)>0): # ignore the empty midi samples
        print("\n[Sample]\n")
        des_midi = reverse_pianoroll.piano_roll_to_pretty_midi(music,fs=FS, program=0);
        des_midi.write(sample_dir+'/train_'+str(epoch)+'_'+str(idx)+'_s'+str(n)+'.mid')            

  def Encoder(self, Xp, y=None, reuse=False, batch_size=64, train=True):                 
    with tf.variable_scope("Encoder") as scope:    
      if reuse:
        scope.reuse_variables()
      with arg_scope([layers.conv2d, layers.conv2d_transpose],
                        activation_fn=tf.nn.elu,
                        normalizer_fn=layers.batch_norm,
                        normalizer_params={'scale': True}
                        ):      
        net = tf.reshape(Xp, [-1, self.input_height, self.input_width, self.c_dim])
        net = layers.conv2d(net, 8, 5, stride=2)
        net = layers.conv2d(net, 16, 5, stride=2)
        net = layers.conv2d(net, 32, 5, stride=2)#, padding='VALID')
        net = layers.flatten(net)
        z_mean = layers.fully_connected(net, self.z_dim, activation_fn=None)   
        z_log_sigma_sq = layers.fully_connected(net, self.z_dim , activation_fn=None)   

        return z_mean, z_log_sigma_sq

  def EncoderOld(self, Xp, y=None, reuse=False, batch_size=64, train=True):                 
    with tf.variable_scope("Encoder") as scope:    
      if reuse:
        scope.reuse_variables()
      net = tf.reshape(Xp, [-1, self.input_height, self.input_width, self.c_dim])
      '''
      net = layers.conv2d(net, 64, filterH, stride=stride)
      net = layers.conv2d(net, 128, filterH, stride=stride)
      net = layers.conv2d(net, 256, filterH, stride=stride)#, padding='VALID')          
      #net = layers.dropout(net, keep_prob=0.9)
      net = layers.flatten(net)
      z_mean = layers.fully_connected(net, self.z_dim, activation_fn=None)   
      z_log_sigma_sq = layers.fully_connected(net, self.z_dim , activation_fn=None)   
      '''
      h0 = lrelu(self.q_bn0(conv2d(net, 8, k_h=5, k_w=5, d_h=2, d_w=2, name='q_h0_conv'), train=train))
      h1 = lrelu(self.q_bn1(conv2d(h0, 16, k_h=5, k_w=5, d_h=2, d_w=2, name='q_h1_conv'), train=train))
      h2 = lrelu(self.q_bn2(conv2d(h1, 32, k_h=5, k_w=5, d_h=2, d_w=2, name='q_h2_conv'), train=train))                  
      z_mean = linear(tf.reshape(h2, [batch_size, -1]), self.z_dim, 'q_m_lin')
      z_log_sigma_sq = linear(tf.reshape(h2, [batch_size, -1]), self.z_dim, 'q_s_lin')

      return z_mean, z_log_sigma_sq

  def discriminator(self, X, y=None, reuse=False):    
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(X, self.df_dim,k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2,k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4,k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8,k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      
      return tf.nn.sigmoid(h4), h4

  def generator(self, z, y=None, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()      

      s_h, s_w = self.output_height, self.output_width                
      s_h2, s_w2 = conv_out_size_same(s_h, stride), conv_out_size_same(s_w, stride)
      s_h4, s_w4 = conv_out_size_same(s_h2, stride), conv_out_size_same(s_w2, stride)
      s_h8, s_w8 = conv_out_size_same(s_h4, stride), conv_out_size_same(s_w4, stride)
      s_h16, s_w16 = conv_out_size_same(s_h8, stride), conv_out_size_same(s_w8, stride)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def sampler(self, z, y=None):    
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()            
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, stride), conv_out_size_same(s_w, stride)
      s_h4, s_w4 = conv_out_size_same(s_h2, stride), conv_out_size_same(s_w2, stride)
      s_h8, s_w8 = conv_out_size_same(s_h4, stride), conv_out_size_same(s_w4, stride)
      s_h16, s_w16 = conv_out_size_same(s_h8, stride), conv_out_size_same(s_w8, stride)

      # project `z` and reshape
      h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [1, s_h8, s_w8, self.gf_dim*4], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [1, s_h4, s_w4, self.gf_dim*2], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [1, s_h2, s_w2, self.gf_dim*1], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [1, s_h, s_w, self.c_dim], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h4')

      return tf.nn.tanh(h4)
 
  def load_Nottingham(self):
    myPath = 'nottingham-dataset/';
    barNum=18798 # total number of bar in the above DS    

    melodies = self.loadTrack(myPath, barNum, 0, self.input_height, self.input_width, FS)            

    X = melodies[1:]    
    Xp = melodies[:-1]
    barNum=barNum-1

    seed = 500#random.randint(1,1000)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Xp)
    
    return X/127., Xp/127., None

  def loadTrack(self, myPath, barNum, trk, height, width, samplingRate):
    allFiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]

    bars = np.zeros(shape=[barNum, height, width, 1], dtype=np.float)    
    
    fileIndex=0
    mi=0
    mb_i=0       

    while fileIndex<len(allFiles) and mb_i<barNum:

        source_midi = pretty_midi.PrettyMIDI(myPath+allFiles[fileIndex],myTrack=trk);
        pianoRoll = source_midi.get_piano_roll(fs=samplingRate, times=None);

        pianoRoll = np.clip(pianoRoll,0,127) # to force element to be in [0-127]
        while mi<pianoRoll.shape[1] and mb_i<barNum:
            
            #[lowPitch , highPitch]
            bar = pianoRoll[33:121,mi:mi+width]            

            # zero padding        
            if(bar.shape[1]<width):                
                bar = np.pad(bar, ((0,0),(0,width-bar.shape[1])), mode='constant', constant_values=0)                            

            bars[mb_i] = bar.reshape(height, width, 1) #m[:,:,np.newaxis]  # np.newaxis: make it 3d
                      
            mb_i+=1
            mi+=width
        fileIndex+=1   
        mi=0
    return bars

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "VAEGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
