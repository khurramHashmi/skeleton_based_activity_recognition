import os
import cv2
import glob
import uuid
import pickle
import random
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from multiprocessing import Pool

SPIXEL = 5
SPATIAL_DIM = 36
TEMPORAL_DIM = 36
STRIDE = 1	# decide how many pseudo images to be created
SKIP = 1	# decide how dense/sparse the skeleton frames are sampled, to build one pseudo image


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


def skel_interpolate(skel_norm):
  fm_num = skel_norm.shape[2]
  skel_dim = skel_norm.shape[0]
  intep_skel = np.zeros([skel_dim, 3, fm_num*2 - 1])
  for ix in range(fm_num*2 - 1):
    if ix % 2 ==0:
      intep_skel[:,:,ix] = skel_norm[:,:,int(ix/2)]
    else:
      intep_skel[:,:,ix] = (skel_norm[:,:,int(ix/2)] + skel_norm[:,:,int(ix/2)+1])/2
  return intep_skel


def velocity(video):

    diff = np.zeros(video.shape)
    for idx in range(video.shape[2]):

        if idx != video.shape[2] - 1:
            diff[:, :, idx] = video[:, :, idx] - video[:, :, idx+1]
        else:
            diff[:, :, idx] = diff[:, :, idx-1]

    return diff

def super_pixel(skel_frame, random_seed, next_frame, compute_diff=True):

  random.seed(random_seed)
  joints_order = np.reshape(random.sample(range(25), 25),(5,5))
  skel_spixel = skel_frame[joints_order]
  skel_spixel2 = next_frame[joints_order]
  diff = np.abs(skel_spixel - skel_spixel2)

  return skel_spixel, diff


def create_img(skel_norm, img_ix):

    # print(img_num)
    skel_arr = np.zeros((SPATIAL_DIM * SPIXEL, TEMPORAL_DIM * SPIXEL, 3), dtype=float)
    diff_arr = np.zeros((SPATIAL_DIM * SPIXEL, TEMPORAL_DIM * SPIXEL, 3), dtype=float)
    compute_diff = True

    for frame_ix in range(TEMPORAL_DIM):
        current_frame = skel_norm[:, :, (img_ix * STRIDE + frame_ix * SKIP)]

        if frame_ix != TEMPORAL_DIM - 1:
            next_frame = skel_norm[:, :, (img_ix * STRIDE + (frame_ix+1) * SKIP)]
        else:
            next_frame = current_frame
            compute_diff = False

        for order_ix in range(SPATIAL_DIM):

            diff, value = super_pixel(current_frame, order_ix, next_frame, compute_diff)
            skel_arr[order_ix * SPIXEL: (order_ix + 1) * SPIXEL,frame_ix * SPIXEL: (frame_ix + 1) * SPIXEL] = value
            diff_arr[order_ix * SPIXEL: (order_ix + 1) * SPIXEL, frame_ix * SPIXEL: (frame_ix + 1) * SPIXEL] = diff

    return skel_arr, diff_arr

def gen(datafiles, labels, process_name, savePath):

    print('process {} handling total {} files'.format(process_name, len(datafiles)))
    image_count = 0

    for count in tqdm(range(len(datafiles))):

        skel_norm = datafiles[count]
        skel_norm = np.reshape(skel_norm, (25, 3, skel_norm.shape[0]))
        ac_id = str(labels[count])

        fm_num = skel_norm.shape[2]

        if fm_num < TEMPORAL_DIM:
          skel_norm = skel_interpolate(skel_norm)
          fm_num = skel_norm.shape[2]
          if fm_num < TEMPORAL_DIM:
              skel_norm = skel_interpolate(skel_norm)
              fm_num = skel_norm.shape[2]

        # find velocity
        # diff = velocity(skel_norm)
        # #diff_arr = create_img(diff, img_ix)
        # diff_img = cv2.normalize(diff, diff_img, 0, 1, cv2.NORM_MINMAX)
        # diff_img = np.array(diff_img * 255, dtype=np.uint8)

        img_num = int((fm_num - TEMPORAL_DIM * SKIP) / STRIDE + 1)
        for img_ix in range(img_num):

            skel_arr, diff_arr = create_img(skel_norm, img_ix)
            skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
            skel_img = np.array(skel_img * 255, dtype=np.uint8)

            diff_img = cv2.normalize(diff_arr, diff_arr, 0, 1, cv2.NORM_MINMAX)
            diff_img = np.array(diff_img * 255, dtype=np.uint8)

            save_file = savePath + '/' + '{:08}_{}'.format(image_count, ac_id) + '.png'
            final_img = np.concatenate((skel_img, diff_img), axis=2)
            pickle.dump(final_img, open(save_file, 'wb'))

        if count % 2000 == 0:
            print('Process {} done with {} files'.format(process_name, count))
            # print(final_img.shape)
#            imageio.imwrite(save_file, diff_img)

# read down sampled train file here
datafiles = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/dsamp_test.p', 'rb'))
print('Total num examples ',len(datafiles))

labels = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/labels_proper_test.p', 'rb'))

from multiprocessing import Process

if __name__ == '__main__':

    datafiles = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/dsamp_train.p', 'rb'))
    print('Total num examples ', len(datafiles))

    labels = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/labels_proper_train.p', 'rb'))
    savePath = './nturgb+d_skeletons_location/'

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    pool = Pool(processes=4)
    start = 0
    increment = 5000

    for i in range(4):
        pool.apply_async(gen, [datafiles[start:start+increment], labels[start:start+increment], str(i), savePath])
        start = start + increment
        if i==3:
            pool.apply_async(gen, [datafiles[start:], labels[start:], str(i), savePath])
    pool.close()
    pool.join()
        # parsed = pool.apply_async(gen, [datafiles[:5000], labels[:10], 'one'])
        # pattern = pool.apply_async(gen, [datafiles[10:20], labels[10:20], 'two'])
