import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import scipy, scipy.ndimage


def nb_vals(matrix, indices):
    matrix = scipy.array(matrix)
    indices = tuple(scipy.transpose(scipy.atleast_2d(indices)))
    arr_shape = scipy.shape(matrix)
    dist = scipy.ones(arr_shape)
    dist[indices] = 0
    dist = scipy.ndimage.distance_transform_cdt(dist, metric='chessboard')
    nb_indices = scipy.transpose(scipy.nonzero(dist == 1))
    return [matrix[tuple(ind)] for ind in nb_indices]






ii=[]
for i in range(13,21):
    if i<10:
        i = '0'+str(i)
        ii.append(i)
    elif 10<=i and i<100:
        i =str(i)
        ii.append(i)
    else:
        i=str(i)
        ii.append(i)



def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data():
    def __init__(self, filepath='./data/image_train_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        #np.random.shuffle(self.data)
        
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_train_pat.npy'):
    return train_data(filepath=filepath)


def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=3)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')


def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def meromero(im1,bs,ps):
    
    size = im1.shape[1]
    
    a = np.ones((ps,ps))
    xx=[]
    
    for p in range(bs):
        for i in range(ps):
            for ii in range(ps):
                
                ssum = 0
                for iii in range(3):
                    j = im1[p][:,:,iii]
                    #aa = neibor(j,[i,ii])
                    
                    
                    
                    
                    ssum += j[i,ii]
                    
                    #ssum += sum1
                sssum=ssum/3
                sssum=2
            
                a[i][ii] = sssum
        xx.append(a)

    xx = np.array(xx)

    xxx = np.random.randn(36,32,32,3)
    xxx[:,:,:,0] = xx
    xxx[:,:,:,1] = xx
    xxx[:,:,:,2] = xx
    return xxx


def qiege(img):
    for i in range(img.shape[0]):
        for ii in range(img.shape[1]):
            for iii in range(img.shape[2]):
                if img[i,ii,iii]>255:
                    img[i,ii,iii] =255
                elif img[i,ii,iii]<0:
                    img[i,ii,iii] =0
                else:
                    img[i,ii,iii]=img[i,ii,iii]
    return img

def qiege2(img):
    for i in range(img.shape[0]):
        for ii in range(img.shape[1]):
            
            if img[i,ii]>255:
                img[i,ii] =255
            elif img[i,ii]<0:
                img[i,ii] =0
            else:
                img[i,ii]=img[i,ii]
    return img

def check1(img):
    for i in range(img.shape[0]):
        for ii in range(img.shape[1]):
            for iii in range(img.shape[2]):
                if img[i,ii,iii]>255:
                    a = 255
                elif img[i,ii,iii]<0:
                    a = 0
                else:
                    a = 'm'
    return a
        