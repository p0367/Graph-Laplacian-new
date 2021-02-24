import time

from utils import *

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#import torch
import numpy as np
import imageio
from imageio import imwrite
import requests
from lxml import etree
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.sparse as ss
import random

np.set_printoptions(threshold=np.inf)

import cv2
import scipy.misc

from sklearn.metrics import mean_squared_error

import os

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda


xrange = range
nub = 1


###############
batch_size = 20
output_size = 128
channel = 3
scale = 2

#####

def srcnn(input, is_training=True, output_channels=3):
    
    #conv0 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv0)
    conv1 = tf.layers.conv2d(input, 64, 9, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, 32, 1, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 3, 5, padding='same', activation=None)
    
    
    return input - conv3, conv3

    


#########################################



class denoiser(object):
   
    def __init__(self, sess, input_c_dim=3, sigma=25, batch_size=batch_size):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        # build model
        self.Y_ = tf.placeholder(tf.float32, [batch_size, None, None, self.input_c_dim], name='clean_image') #ground

        #self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name = 'noisy_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        '''original'''
          
        self.X = tf.image.resize(self.Y_ ,(int(output_size/scale) ,int(output_size/scale)), method='bicubic')
        self.X = tf.image.resize(self.X, (output_size,output_size), method='bicubic')
        
        self.para1 = tf.placeholder(tf.float32, name='para1')

        self.Y = srcnn(self.X, is_training=True, output_channels=3)

        #self.Y_b = tf.image.resize(self.X, (output_size,output_size), method='bicubic')
        self.Y_b = self.X
        ##ground
        self.Y_2 = self.Y_   #xxxxxxxx
        #loss
        ####graph
        self.Y0 = self.Y[0]

        #psnr
        self.psnr = tf.norm(tf.image.psnr(self.Y_2, self.Y0, max_val=1.0) , ord=1) / (batch_size)
        #ssim
        self.ssim = tf.norm(tf.image.ssim(self.Y_2, self.Y0, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03) , ord=1) / (batch_size)
        
        #psnr
        self.psnr_b = tf.norm(tf.image.psnr(self.Y_2, self.Y_b, max_val=1.0) , ord=1) / (batch_size)
        #ssim
        self.ssim_b = tf.norm(tf.image.ssim(self.Y_2, self.Y_b, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03) , ord=1) / (batch_size)

        #0leftup↖
        self.k0 = tf.tile(tf.reshape(tf.constant([[1, 0], [0, 0]], tf.float32),[2,2,1,1]),[1,1,1,3])
        self.Yd0 = tf.nn.conv2d(self.Y0, self.k0,strides=[1, 1, 1, 1], padding='VALID')
        self.Yg0 = tf.nn.conv2d(self.Y_2, self.k0,strides=[1, 1, 1, 1], padding='VALID')
        #1rightup↗
        self.k1 = tf.tile(tf.reshape(tf.constant([[0, 1], [0, 0]], tf.float32),[2,2,1,1]),[1,1,1,3])
        self.Yd1 = tf.nn.conv2d(self.Y0, self.k1,strides=[1, 1, 1, 1], padding='VALID')
        self.Yg1 = tf.nn.conv2d(self.Y_2, self.k1,strides=[1, 1, 1, 1], padding='VALID')

        #2leftdown↙
        self.k2 = tf.tile(tf.reshape(tf.constant([[0, 0], [1, 0]], tf.float32),[2,2,1,1]),[1,1,1,3])
        self.Yd2 = tf.nn.conv2d(self.Y0, self.k2,strides=[1, 1, 1, 1], padding='VALID')
        self.Yg2 = tf.nn.conv2d(self.Y_2, self.k2,strides=[1, 1, 1, 1], padding='VALID')

        #3rightdown↘
        self.k3 = tf.tile(tf.reshape(tf.constant([[0, 0], [0, 1]], tf.float32),[2,2,1,1]),[1,1,1,3])
        self.Yd3 = tf.nn.conv2d(self.Y0, self.k3,strides=[1, 1, 1, 1], padding='VALID')
        self.Yg3 = tf.nn.conv2d(self.Y_2, self.k3,strides=[1, 1, 1, 1], padding='VALID')

        #1
        self.Yd01 = tf.math.subtract(self.Yd0,self.Yd1)
        self.Yg01 = tf.math.subtract(self.Yg0,self.Yg1)
        self.Y01 = tf.multiply(tf.math.subtract(self.Yd01,self.Yg01),tf.math.subtract(self.Yd01,self.Yg01))
        #2
        self.Yd02 = tf.math.subtract(self.Yd0,self.Yd2)
        self.Yg02 = tf.math.subtract(self.Yg0,self.Yg2)
        self.Y02 = tf.multiply(tf.math.subtract(self.Yd02,self.Yg02),tf.math.subtract(self.Yd02,self.Yg02))
        #3
        self.Yd03 = tf.math.subtract(self.Yd0,self.Yd3)
        self.Yg03 = tf.math.subtract(self.Yg0,self.Yg3)
        self.Y03 = tf.multiply(tf.math.subtract(self.Yd03,self.Yg03),tf.math.subtract(self.Yd03,self.Yg03))
        #4
        self.Yd12 = tf.math.subtract(self.Yd1,self.Yd2)
        self.Yg12 = tf.math.subtract(self.Yg1,self.Yg2)
        self.Y12 = tf.multiply(tf.math.subtract(self.Yd12,self.Yg12),tf.math.subtract(self.Yd12,self.Yg12))

        self.loss2 = tf.norm(self.Y01+self.Y02+self.Y03+self.Y12,ord=1) * (1/(batch_size*(output_size-1)*(output_size-1)*channel*4))

            


        # real loss
        #MSE_V
        self.loss12 = tf.math.subtract(self.Y[0],self.Y_2)
        self.loss13 = tf.multiply(self.loss12,self.loss12)
        self.loss1 = tf.norm(self.loss13,ord=1) * (1 / (batch_size*output_size*output_size*channel))
        
        self.loss = self.loss1+ self.para1*self.loss2
        

        #zloss
        self.zY = self.Y[0]
        self.zloss12 = self.Y_2 - self.zY
        self.zloss13 = tf.multiply(self.zloss12,self.zloss12)
        self.zloss1 = tf.norm(self.zloss13,ord=1) * (1 / (batch_size*output_size*output_size*channel))
        self.zloss2 = self.loss1 #MSE
        self.zloss3 = self.loss2  #GBS
        self.zzY = self.Y_2

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
                
        self.eva_psnr = 0
        

        #optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        #optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr, name='Gd')
        optimizer = tf.train.MomentumOptimizer(self.lr,momentum=0.9, name='Mm')
        

        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def train(self, data, tdata,ppara1,batch_size, ckpt_dir, epoch,lr, sample_dir, eval_every_epoch=2):
        # assert data range is between 0 and 1
        qm = 'L-'+str(ppara1)

        numBatch = int(data.shape[0] / batch_size)
        tnumBatch = int(tdata.shape[0] / batch_size)
        trBatch = numBatch

        

        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        #tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        


        ii=[]
        aa=[]
        bb=[]

        cc=[]
        dd=[]

        ee=[]
        ff=[]

        pddj1 = []
        pddj2 = []
        pddj3 = []

        pgddj1 = []
        pgddj2 = []
        pgddj3 = []


        lw =[]

        psnr0 = []
        ssim0 = []


        for epoch in xrange(start_epoch, epoch):
            #change learning rate
            if epoch % 1000 ==0 and epoch !=0 :
                lr = lr * 0.5

                
            #MSE
            sumloss=0
            tsumloss=0


            #MAE
            sumsmooth = 0
            tsumsmooth = 0

            #CE
            csumloss = 0
            tcsumloss = 0

            #psnr & ssim
            sumpsnr = 0
            sumssim = 0

            sumpsnr_b = 0
            sumssim_b = 0

            
            count=0
            sj1=range(data.shape[0])
            
            for batch_id in range(numBatch):
                count += 1

                #sj1=range(batch_size*numBatch)
                sjj1=random.sample(sj1,batch_size)

                #w1 = 1
                ground_images = data[sjj1, :, :, :]
                para1 = ppara1
                
                _, loss,summary = self.sess.run([self.train_op, self.loss , merged],
                                                 feed_dict={self.Y_: ground_images, self.lr: lr, self.para1:para1,
                                                            self.is_training: True})
                
                

                                                            
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, count , numBatch, time.time() - start_time, loss))

                
                iter_num += 1
                writer.add_summary(summary, iter_num)

                sj1=[yy for yy in sj1 if yy not in sjj1]
                
                

            
                

#
#
##############  算loss ## train  ##################
# 
#           
            
            
            count=0
            sj2 = range(trBatch*batch_size)
            
            for batch_id in range(trBatch):
                count += 1
                
                sjj2=random.sample(sj2,batch_size)

                ground_images = data[sjj2, :, :, :]
                
                
                loss_train , smooth_train , ce_train  = self.sess.run([self.zloss1,self.zloss2,self.zloss3],feed_dict={self.Y_: ground_images,self.is_training: False})
               
              
                

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, rtloss: %.6f"
                      % (epoch + 1, count , trBatch, time.time() - start_time, loss_train))

                
                tsumloss+=loss_train
                tsumsmooth+=smooth_train
                tcsumloss+=ce_train
               
                sj2=[yy for yy in sj2 if yy not in sjj2]

################for test###########################


            count=0
            print('**************test*****************')
            for batch_id in range(tnumBatch):

                count += 1
                

                ground_images_t = tdata[batch_id *batch_size : (batch_id+1) *batch_size  , :, :, :]
                
                loss_test , smooth_test , ce_test ,psnr1 , ssim1 ,psnr1_b,ssim1_b= self.sess.run([self.zloss1,self.zloss2,self.zloss3,self.psnr,self.ssim,self.psnr_b,self.ssim_b],
                feed_dict={self.Y_: ground_images_t,self.is_training: False})
                
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, count , tnumBatch, time.time() - start_time, loss_test))
                
                sumloss +=loss_test
                sumsmooth +=smooth_test
                csumloss +=ce_test

                sumpsnr += psnr1
                sumssim += ssim1

                sumpsnr_b += psnr1_b
                sumssim_b += ssim1_b

                output_clean_image,output_ground_image  = self.sess.run([self.zY,self.zzY],
                feed_dict={self.Y_: ground_images_t,self.is_training: False})
                
                if batch_id ==0 :
                    ddj1 = output_clean_image[0,10,111,1] #edge
                    ddj2 = output_clean_image[0,7,8,1] #back
                    ddj3 = output_clean_image[0,39,92,1] #inside
                    gddj1 = output_ground_image[0,10,111,1]
                    gddj2 = output_ground_image[0,7,8,1]
                    gddj3 = output_ground_image[0,39,92,1]
                            
                
                if epoch % 500 ==0 and epoch !=0 and batch_id <1:
                    for i in range(batch_size):
                        p18 = output_clean_image[i,:,:,:]
                        p18 = p18 * 255
                        p18 = qiege(p18)  

                        if  os.path.exists('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)) :
                            pass
                        else:
                            os.mkdir('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)) 

                        if  os.path.exists('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)+'/epoch'+str(epoch)) :
                            pass
                        else:
                            os.mkdir('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)+'/epoch'+str(epoch)) 
                        if  os.path.exists('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)+'/xxepoch'+str(epoch)) :
                            pass
                        else:
                            os.mkdir('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)+'/xxepoch'+str(epoch)) 
                        
                        imageio.imwrite('/home1/tei/work1/new-super-resolution/output'+str(nub)+'_'+str(ppara1)+'/epoch'+str(epoch)+'/'+str(batch_size*batch_id + i)+'.png', p18.astype(np.uint8))
                    


            
            #MSE
            tavgloss=tsumloss/trBatch
            avgloss=sumloss/tnumBatch 

            aa.append(tavgloss)
            bb.append(avgloss)
            ii.append(epoch)

            #MSE_V
            tavgsmooth = tsumsmooth/trBatch
            avgsmooth = sumsmooth/tnumBatch
            cc.append(tavgsmooth)
            dd.append(avgsmooth)

            #CE_V
            tcavgloss = tcsumloss/trBatch
            cavgloss = csumloss/tnumBatch
            ee.append(tcavgloss)
            ff.append(cavgloss)


            ##value
            
            pddj1.append(ddj1)
            pddj2.append(ddj2)
            pddj3.append(ddj3)
            pgddj1.append(gddj1)
            pgddj2.append(gddj2)
            pgddj3.append(gddj3)

            #psnr&ssim
            avgpsnr = sumpsnr / tnumBatch
            avgssim = sumssim / tnumBatch
            psnr0.append(avgpsnr)
            ssim0.append(avgssim)

            avgpsnr_b = sumpsnr_b / tnumBatch
            avgssim_b = sumssim_b / tnumBatch
            
            #value
            
            if epoch%5 ==0:
                plt.cla()

                plt.plot(ii,pddj1,color = 'mediumblue', linestyle = "-.",label="v_edge")
                plt.plot(ii,pddj2,color = 'green', linestyle = "-.",label="v_back")
                plt.plot(ii,pddj3,color = 'red', linestyle = "-.",label="v_inside")
                plt.plot(ii,pgddj1,color = 'dodgerblue', linestyle = "-",label="g_edge")
                plt.plot(ii,pgddj2,color = 'yellowgreen', linestyle = "-",label="g_back")
                plt.plot(ii,pgddj3,color = 'salmon', linestyle = "-",label="g_inside")
                #plt.ylim(-0.1,1.1)
                plt.title('value_'+str(nub)+'_'+str(qm), y=-0.15,x=0)
                plt.legend(loc='best')

                plt.xlabel('epoch')
                plt.ylabel('value')
                plt.grid(ii)
                plt.savefig("value_"+str(nub)+'_'+str(ppara1)+".png")
                plt.close('all')

            #loss
            #best #total
            if epoch%5 ==0:
                plt.cla()
                plt.plot(ii,aa,color = 'firebrick', linestyle = "-",label="MSE-train")
                #if epoch < change:
                plt.plot(ii,bb,color = 'lightskyblue',linestyle = "-",label="MSE-test")
                #plt.plot(ii,cc,color = 'darkslategray',linestyle = "--",label="MSE_V-train")
                #plt.plot(ii,dd,color = 'red',linestyle = "--",label="MSE_V-test")
                #if epoch < change:
                plt.plot(ii,ee,color = 'black',linestyle = "-.",label="GBS-train")
                plt.plot(ii,ff,color = 'magenta',linestyle = "-.",label="GBS-test")
                #plt.ylim(-0.02,0.71)
                plt.title('plot_'+str(nub)+'_'+str(qm), y=-0.15,x=0)
                plt.legend(loc='best')

                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.grid(ii)
                plt.savefig("plot_loss_"+str(nub)+'_'+str(ppara1)+".png")
                plt.close('all')
            
            #psnr
            if epoch%5 ==0:
                plt.cla()
                plt.plot(ii,psnr0,color = 'firebrick', linestyle = "-",label="PSNR")
                #plt.ylim(-0.02,0.71)
                plt.title('plot_'+str(nub)+'_'+str(qm), y=-0.15,x=0)
                plt.legend(loc='best')

                plt.xlabel('epoch')
                plt.ylabel('psnr')
                plt.grid(ii)
                plt.savefig("plot_psnr_"+str(nub)+'_'+str(ppara1)+".png")
                plt.close('all')

            #ssim
            if epoch%5 ==0:
                plt.cla()
                plt.plot(ii,ssim0,color = 'firebrick', linestyle = "-",label="SSIM")
                #plt.ylim(-0.02,0.71)
                plt.title('plot_'+str(nub)+'_'+str(qm), y=-0.15,x=0)
                plt.legend(loc='best')

                plt.xlabel('epoch')
                plt.ylabel('ssim')
                plt.grid(ii)
                plt.savefig("plot_ssim_"+str(nub)+'_'+str(ppara1)+".png")
                plt.close('all')


 
            if epoch%100 == 0:
                self.save(iter_num, ckpt_dir)

            if epoch%1 ==0:

                ff1 = open('./plot'+str(nub)+'_'+str(ppara1)+'.txt','a')
                ff1.write('epoch '+str(epoch)+' -MSE_train: '+str(tavgloss)+'\n')
                ff1.write('epoch '+str(epoch)+' -MSE_test: '+str(avgloss)+'\n')
                #ff1.write('epoch '+str(epoch)+' -MSE_V_train: '+str(tavgsmooth)+'\n')
                #ff1.write('epoch '+str(epoch)+' -MSE_V_test: '+str(avgsmooth)+'\n')
                ff1.write('epoch '+str(epoch)+' -CE_V_train: '+str(tcavgloss)+'\n')
                ff1.write('epoch '+str(epoch)+' -CE_V_test: '+str(cavgloss)+'\n')

                ff1.write('epoch '+str(epoch)+' -psnr_test: '+str(avgpsnr)+'\n')
                ff1.write('epoch '+str(epoch)+' -ssim_test: '+str(avgssim)+'\n')

                ff1.write('epoch '+str(epoch)+' -psnr_bulic: '+str(avgpsnr_b)+'\n')
                ff1.write('epoch '+str(epoch)+' -ssim_bulic: '+str(avgssim_b)+'\n')

                ff1.write('epoch '+str(epoch)+' -learning_rate: '+str(lr)+'\n')
                ff1.write('\n')
                ff1.close()



        print("[*] Finish training.")
        

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0
