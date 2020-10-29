import os
import cv2
import uuid
import pickle
import random
import numpy as np
from tqdm import tqdm
import h5py
from multiprocessing import Pool
import argparse
from data_ms import NTUDataLoaders

SPIXEL = 5
SPATIAL_DIM = 36
TEMPORAL_DIM = 36
STRIDE = 1  # decide how many pseudo images to be created
SKIP = 1  # decide how dense/sparse the skeleton frames are sampled, to build one pseudo image


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


def cal_distace(arr_1, arr_2):
    joint_dist = []

    for joint in range(25):
        diff = np.abs(np.array(np.where(arr_1 == joint)) - np.array(np.where(arr_2 == joint)))
        joint_dist.append(np.max(diff))

    return np.sum(joint_dist)


def get_arrangements(num_arrangements):
    total_arrangements = []
    # dist_matrix = np.zeros(shape=(num_arrangements, num_arrangements))
    dist_list = {}

    # get total arrangements
    for i in range(num_arrangements):
        total_arrangements.append(get_order(i))

    # print(total_arrangements)
    for i in range(len(total_arrangements)):
        if i != len(total_arrangements) - 1:
            for j in range(i + 1, len(total_arrangements)):
                dist_list[str(i) + '_' + str(j)] = cal_distace(total_arrangements[i], total_arrangements[j])

    # sort the dict
    sorted_dict = {k: v for k, v in sorted(dist_list.items(), key=lambda item: item[1], reverse=True)}
    top_n = list(sorted_dict.items())[:37]

    top_arrangements = []

    for i in top_n:
        top_arrangements.append(total_arrangements[int(i[0].split('_')[0])])

    # delete extra var
    del sorted_dict
    del top_n
    del dist_list
    del total_arrangements

    return top_arrangements


def create_img(skel_norm, img_ix, cal_velocity=False):
    spatial_arr = None
    # spatial_arr_diff = None

    joint_arrangements = get_arrangements(50)  # get top 36 arrangements

    for order_ix in range(SPATIAL_DIM):

        temporal_arr = None
        # temporal_arr_diff = None

        joints_order = joint_arrangements[order_ix]  # select a arrangement and use it for all frames

        for frame_ix in range(TEMPORAL_DIM):
            current_frame = skel_norm[:, :, (img_ix * STRIDE + frame_ix * SKIP)]
            temporal_arr = current_frame[joints_order] if temporal_arr is None else np.vstack(
                (temporal_arr, current_frame[joints_order]))

            # if cal_velocity:
            #     if frame_ix != TEMPORAL_DIM - 1:
            #         next_frame = skel_norm[:, :, (img_ix * STRIDE + (frame_ix + 1) * SKIP)]
            #         diff = current_frame[joints_order] - next_frame[joints_order]
            #     else:
            #         diff = np.reshape(current_frame, (5, 5, 3))

            # temporal_arr_diff = diff if temporal_arr_diff is None else np.hstack((temporal_arr_diff, diff))

        spatial_arr = temporal_arr if spatial_arr is None else np.hstack((spatial_arr, temporal_arr))
        # spatial_arr_diff = temporal_arr_diff if spatial_arr_diff is None else np.vstack(
        #     (spatial_arr_diff, temporal_arr_diff))

    return spatial_arr  # , spatial_arr_diff


def gen(loader, process_name, savePath):
    # print('process {} handling total {} files'.format(process_name, len(datafiles)))

    # for count in tqdm(range(len(datafiles))):
    for count, (data, target) in enumerate(tqdm(loader)):

        skel_norm_data = data[0].numpy() #datafiles[count]
        # skel_norm_data = np.array(skel_norm['input'])
        skel_norm_data = np.reshape(skel_norm_data, (skel_norm_data.shape[0], 25, 3))
        skel_norm_data = np.moveaxis(skel_norm_data, 0, -1)

        ac_id = str(target[0].numpy())  #str(skel_norm['label'])
        do_inter = True
        # original = skel_norm_data.shape[2]
        # interpolated = False

        while do_inter:
            fm_num = skel_norm_data.shape[2]
            if fm_num < 40:
                skel_norm_data = skel_interpolate(skel_norm_data)
                # interpolated = True
            else:
                do_inter = False
                # if interpolated:
                #     print('size increased from {} to {}'.format(original, fm_num))

        img_num = 5  # int((fm_num - TEMPORAL_DIM * SKIP) / STRIDE + 1)

        for img_ix in range(img_num):
            skel_arr = create_img(skel_norm_data, img_ix)
            skel_img = cv2.normalize(skel_arr, skel_arr, 0, 1, cv2.NORM_MINMAX)
            skel_img = np.array(skel_img * 255, dtype=np.uint8)

            # velocity_img = cv2.normalize(velocity_arr, velocity_arr, 0, 1, cv2.NORM_MINMAX)
            # velocity_img = np.array(velocity_img * 255, dtype=np.uint8)

            save_file = savePath + '/' + str(uuid.uuid4()) + '_{}'.format(ac_id) + '.png'
            # final_img = np.concatenate((skel_img, velocity_img), axis=2)
            # pickle.dump(final_img, open(save_file, 'wb'))
            cv2.imwrite(save_file, skel_img)

        # break
        # if count % 2000 == 0:
        #     print('Process {} done with {} files'.format(process_name, count))

if __name__ == '__main__':

    # print(get_arrangements(100))
    parser = argparse.ArgumentParser(description="Skeleton Classification Training Script")
    parser.add_argument("-m", "--mode", default='train', type=str, help="Train or test mode")
    args = parser.parse_args()

    if args.mode == 'train':

        # datafiles = pickle.load(open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/trans_train_data.pkl', 'rb'))
        path = '/home/ahmed/Desktop/dataset_skeleton/SGN_Data/NTU_CS.h5'

        f = h5py.File(path, 'r')
        datafiles = f['x'][:]
        labels = np.argmax(f['y'][:], -1)
        val_X = f['valid_x'][:]
        val_Y = np.argmax(f['valid_y'][:], -1)
        datafiles = np.concatenate([datafiles, val_X], axis=0)
        labels = np.concatenate([labels, val_Y], axis=0)
        test_features = f['test_x'][:]
        test_data_label = np.argmax(f['test_y'][:], -1)

        data_array = None
        ntu_loaders = NTUDataLoaders(datafiles, labels, 0, seg=30)
        train_loader = ntu_loaders.get_loader(64, 2)

        for i, (data, _) in enumerate(train_loader):
            data_array = data if data_array is None else np.vstack((data_array, data))

        save_path = '/home/ahmed/Desktop/dataset_skeleton/SGN_Data'
        h5file = h5py.File(os.path.join(save_path, 'train.h5'), 'w')
        h5file.create_dataset('x', data=data_array)
        h5file.create_dataset('y', data=labels)
        h5file.close()

        ntu_loaders = NTUDataLoaders(test_features, test_data_label, 0, seg=30)
        train_loader = ntu_loaders.get_loader(64, 2)

        data_array = None
        for i, (data, _) in enumerate(train_loader):
            data_array = data if data_array is None else np.vstack((data_array, data))

        h5file = h5py.File(os.path.join(save_path, 'test.h5'), 'w')
        h5file.create_dataset('x', data=data_array)
        h5file.create_dataset('y', data=test_data_label)
        h5file.close()

        # del f, val_X, val_Y

        # ntu_loaders = NTUDataLoaders(datafiles, labels, 0, seg=30)
        # train_loader = ntu_loaders.get_loader(1, 2)
        #
        # print('Total num examples ', datafiles.shape[0])  # 40091
        # # print('Total num examples ', len(datafiles))  # 40091
        #
        # savePath = '/home/ahmed/Desktop/dataset_skeleton/train/'
        #
        # if not os.path.exists(savePath):
        #     os.makedirs(savePath)
        # gen(train_loader, '1', savePath)
        # pool = Pool(processes=5)
        # start = 0
        # increment = 10000
        #
        # for i in range(5):
        #     pool.apply_async(gen, [datafiles[start:start + increment], labels[start:start+increment], str(i), savePath])
        #     start = start + increment
        #     if i == 4:
        #         pool.apply_async(gen, [datafiles[start:], labels[start:], str(i), savePath])
        # # print('All process have been called')
        # pool.close()
        # pool.join()

    else:

        path = '/home/ahmed/Desktop/dataset_skeleton/SGN_Data/NTU_CS.h5'

        f = h5py.File(path, 'r')
        test_features = f['test_x'][:]
        test_data_label = np.argmax(f['test_y'][:], -1)

        # datafiles = pickle.load(
        #     open('/home/ahmed/Desktop/dataset_skeleton/cross_subject_data/trans_test_data.pkl', 'rb'))
        # print('Total num examples ', len(datafiles))  # 16000-
        #
        savePath = '/home/ahmed/Desktop/dataset_skeleton/test/'

        if not os.path.exists(savePath):
            os.makedirs(savePath)

        ntu_loaders = NTUDataLoaders(test_features, test_data_label, 0, seg=30)
        test_loader = ntu_loaders.get_loader(1, 2)
        del f
        gen(test_loader, '1', savePath)

        # pool = Pool(processes=5)
        # start = 0
        # increment = 4000
        #
        # for i in range(5):
        #     pool.apply_async(gen, [datafiles[start:start + increment], str(i), savePath])
        #     start = start + increment
        #     if i == 4:
        #         pool.apply_async(gen, [datafiles[start:], str(i), savePath])
        # pool.close()
        # pool.join()
