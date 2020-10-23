import os
import cv2
import uuid
import pickle
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

SPIXEL = 5
SPATIAL_DIM = 36
TEMPORAL_DIM = 36
STRIDE = 1  # decide how many pseudo images to be created
SKIP = 1  # decide how dense/sparse the skeleton frames are sampled, to build one pseudo image


def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)


def skel_interpolate(skel_norm):
    fm_num = skel_norm.shape[2]
    skel_dim = skel_norm.shape[0]
    intep_skel = np.zeros([skel_dim, 3, fm_num * 2 - 1])
    for ix in range(fm_num * 2 - 1):
        if ix % 2 == 0:
            intep_skel[:, :, ix] = skel_norm[:, :, int(ix / 2)]
        else:
            intep_skel[:, :, ix] = (skel_norm[:, :, int(ix / 2)] + skel_norm[:, :, int(ix / 2) + 1]) / 2
    return intep_skel


def get_order(random_seed):
    random.seed(random_seed)
    joints_order = np.reshape(random.sample(range(25), 25), (5, 5))
    return joints_order


def create_img(skel_norm, img_ix):
    spatial_arr = None
    spatial_arr_diff = None

    for order_ix in range(SPATIAL_DIM):

        joints_order = get_order(order_ix)
        temporal_arr = None
        temporal_arr_diff = None

        for frame_ix in range(TEMPORAL_DIM):

            current_frame = skel_norm[:, :, (img_ix * STRIDE + frame_ix * SKIP)]
            temporal_arr = current_frame[joints_order] if temporal_arr is None else np.vstack(
                (temporal_arr, current_frame[joints_order]))

            if frame_ix != TEMPORAL_DIM - 1:
                next_frame = skel_norm[:, :, (img_ix * STRIDE + (frame_ix + 1) * SKIP)]
                diff = current_frame[joints_order] - next_frame[joints_order]
            else:
                diff = np.reshape(current_frame, (5, 5, 3))

            temporal_arr_diff = diff if temporal_arr_diff is None else np.hstack((temporal_arr_diff, diff))

        spatial_arr = temporal_arr if spatial_arr is None else np.hstack((spatial_arr, temporal_arr))
        spatial_arr_diff = temporal_arr_diff if spatial_arr_diff is None else np.vstack(
            (spatial_arr_diff, temporal_arr_diff))

    return spatial_arr, spatial_arr_diff


def gen(datafiles, process_name, savePath):
    print('process {} handling total {} files'.format(process_name, len(datafiles)))

    for count in tqdm(range(len(datafiles))):

        skel_norm = datafiles[count]
        skel_norm_data = np.array(skel_norm['input'])
        skel_norm_data = np.reshape(skel_norm_data, (25, 3, skel_norm_data.shape[0]))

        ac_id = str(skel_norm['label'])

        fm_num = skel_norm_data.shape[2]

        if fm_num < TEMPORAL_DIM:
            skel_norm_data = skel_interpolate(skel_norm_data)
            fm_num = skel_norm_data.shape[2]
            if fm_num < TEMPORAL_DIM:
                skel_norm_data = skel_interpolate(skel_norm_data)
                fm_num = skel_norm_data.shape[2]

        img_num = int((fm_num - TEMPORAL_DIM * SKIP) / STRIDE + 1)

        for img_ix in range(img_num):
            skel_arr, velocity_arr = create_img(skel_norm_data, img_ix)
            skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
            skel_img = np.array(skel_img * 255, dtype=np.uint8)

            velocity_img = cv2.normalize(velocity_arr, velocity_arr, 0, 1, cv2.NORM_MINMAX)
            velocity_img = np.array(velocity_img * 255, dtype=np.uint8)

            save_file = savePath + '/' + str(uuid.uuid4()) + '_{}'.format(ac_id) + '.p'
            final_img = np.concatenate((skel_img, velocity_img), axis=2)
            pickle.dump(final_img, open(save_file, 'wb'))

        # if count % 2000 == 0:
        #     print('Process {} done with {} files'.format(process_name, count))
        # print(final_img.shape)


#            imageio.imwrite(save_file, diff_img)


if __name__ == '__main__':

    datafiles = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/trans_train_data.pkl', 'rb'))
    print('Total num examples ', len(datafiles))  # 40091

    savePath = '/home/ahmed/Desktop/dataset_skeleton/temporal_train/'

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    pool = Pool(processes=5)
    start = 0
    increment = 10000

    for i in range(5):
        pool.apply_async(gen, [datafiles[start:start + increment], str(i), savePath])
        start = start + increment
        if i == 4:
            pool.apply_async(gen, [datafiles[start:], str(i), savePath])
    pool.close()
    pool.join()