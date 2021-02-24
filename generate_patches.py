import argparse
import glob
from PIL import Image
import PIL
import random
from utils import *
import cv2
import imageio
from imageio import imwrite
# the pixel value range is '0-255'(uint8 ) of training data

xrange = range


# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data1/test', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data1', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=128, help='patch size')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=10, help='batch size')
args = parser.parse_args()

channel = 3

def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0
    filepaths = sorted(glob.glob(args.src_dir + '/*.png'))
    if isDebug:
        filepaths = filepaths[:10]
    print("number of training data %d" % len(filepaths))
    
    #scales = [1, 1, 1, 1]
    
    # calculate the number of patches
    for i in xrange(len(filepaths)):
        #img = Image.open(filepaths[i]).convert('L') # convert RGB to gray
        #img = Image.open(filepaths[i])
        img = imageio.imread(filepaths[i])
        newsize = (img.shape[0] , img.shape[1] )
        #img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)  # do not change the original img
        img_s = img
        im_h, im_w = img_s.shape[0],img_s.shape[1]
        for x in range(0 , im_h , args.pat_size):
            for y in range(0 , im_w , args.pat_size):
                
                
                count += 1
    print(count)                
    origin_patch_num = count * DATA_AUG_TIMES
    
    #if origin_patch_num % args.bat_size != 0:
    #    numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size
    #else:
    numPatches = origin_patch_num
    print("total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size))
    numPatches = int(numPatches)
    
    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, channel), dtype="uint8")
    #inputs = np.zeros((numPatches, args.pat_size, args.pat_size), dtype="uint8")
    
    count = 0
    # generate patches
    for i in xrange(len(filepaths)):
        #img_s = Image.open(filepaths[i]) #若黑白 则0
        img_s= imageio.imread(filepaths[i])
        #img = Image.open(filepaths[i])
        #print(img_s.shape)
        #newsize = (int(img.shape[0]), int(img.shape[1]))
            # print newsize
        #img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
        #img_s = cv2.resize(img,interpolation=cv2.INTER_CUBIC)

        img_s = np.reshape(np.array(img_s, dtype="uint8"),
                            (img_s.shape[0], img_s.shape[1],channel))  # extend one dimension
            
            
        im_h, im_w, _ = img_s.shape
        #for x in range(0 + args.step, im_h , args.pat_size):
            #for y in range(0 + args.step, im_w , args.pat_size):
        inputs[count, :, :, :] = data_augmentation(img_s[:, :, :], \
                                                    random.randint(0, 0))
        count += 1
    # pad the batch
    #if count < numPatches:
    #    to_pad = numPatches - count
    #    inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
    
    #if not os.path.exists(args.save_dir):
    #    os.mkdir(args.save_dir)
    np.save(os.path.join(args.save_dir, "img_test_pats"), inputs)
    print("size of inputs tensor = " + str(inputs.shape))


if __name__ == '__main__':
    generate_patches()
