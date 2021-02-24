import argparse
from glob import glob

import tensorflow as tf

from model_EDSR import denoiser       ##select model here##
from utils import *
import numpy as np
import scipy.misc
import os

datanub = 1   ##select dataset 1/2##

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10001, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--p', dest='p', type=float, default=10, help='p') #Parameter for Graph Laplacian Regularizer
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
args = parser.parse_args()

#sensei, i m here , there is some problem with my micphone

def denoiser_train(denoiser):
    with load_data(filepath='./data'+str(datanub)+'/img_train_pats.npy') as data:    #train  
        with load_data(filepath='./data'+str(datanub)+'/img_test_pats.npy') as tdata:   #test

            data = data.astype(np.float32)  / 255.0# normalize the data to 0-1            
            tdata = tdata.astype(np.float32)  / 255.0 # normalize the data to 0-1            
    
            denoiser.train(data , tdata ,ppara1 = args.p,
            batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr = args.lr, sample_dir=args.sample_dir)

       

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)

        #config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        #sess = tf.compat.v1.InteractiveSession()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'train':
                denoiser_train(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.compat.v1.app.run()
