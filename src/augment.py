import pickle
import numpy as np
import cv2
import random
import tqdm
import os
import uuid
import pandas as pd

def add_noise(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col, ch)
        noisy = image + gauss
    return noisy

def rotation(data, alpha_beta=(0,0)):
    # rotate the skeleton around x-y axis
    r_alpha = alpha_beta[0] * np.pi / 180
    r_beta = alpha_beta[1] * np.pi / 180

    rx = np.array([[1, 0, 0],
                   [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                   [0, np.sin(r_alpha), np.cos(r_alpha)]]
                  )

    ry = np.array([
        [np.cos(r_beta), 0, np.sin(r_beta)],
        [0, 1, 0],
        [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
    ])

    r = ry.dot(rx)
    data = data.dot(r)

    return data

def augment(pickle_path, out_path, mode='train'):

    out_path = os.path.join(out_path,  mode + '_images')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if mode == 'train':
        input_features = pickle.load(open(pickle_path + "/dsamp_train.p", "rb"))
        labels = pickle.load(open(pickle_path + "/labels_proper_train.p", "rb"))

    else:
        input_features = pickle.load(open(pickle_path + "/dsamp_test.p", "rb"))
        labels = pickle.load(open(pickle_path + "/labels_proper_test.p", "rb"))

    # label_temp = labels.copy()
    image_list = []
    angle_combinations = [(0,45 ), (30, 30), (90, 0), (45, 90), (120, 120), (45, 45), (15, 15), (45,0), (90,90)]

    for i in tqdm.tqdm(range(len(input_features))):

        if input_features[i].shape[0] < 50:
            inputs = np.zeros((50, 75), dtype=float)
            inputs[:input_features[i].shape[0], :] = np.copy(input_features[i])
            video = inputs
        else:
            video = input_features[i]

        video = np.reshape(video, (50, 25, 3))
        curr_label = labels[i]
        # labels.pop(i)
        # indices = [i for i, x in enumerate(labels) if x == curr_label]
        top_frame = None
        bottom_frame = None

        for count in range(9):

            # apply rotation in pairs
            aug_img = rotation(video, angle_combinations[count])
            if count < 4:
                top_frame = np.hstack((video, aug_img)) if top_frame is None else np.hstack((top_frame, aug_img))
            else:
                bottom_frame = aug_img if bottom_frame is None else np.hstack((bottom_frame, aug_img))
            # idx = random.sample(indices, 1)[0]
            # corresponding_ex = input_features[idx]
            # if corresponding_ex.shape[0] < 50:
            #     inputs = np.zeros((50, 75), dtype=float)
            #     inputs[:corresponding_ex.shape[0], :] = np.copy(corresponding_ex)
            #     corresponding_ex = inputs
            #
            # corresponding_ex = add_noise("gauss",corresponding_ex.reshape((50,25,3)))
            # if j < 3:
            #     video = np.hstack((video, corresponding_ex))
            # else:
            #     new_base = corresponding_ex if new_base is None else np.hstack((new_base, corresponding_ex))

        curr_video = np.vstack((top_frame, bottom_frame))
        curr_video = np.vstack((np.zeros((12, curr_video.shape[1], video.shape[2])), curr_video))
        curr_video = np.vstack((curr_video, np.zeros((13, curr_video.shape[1], video.shape[2]))))
        write_path = os.path.join(out_path, str(uuid.uuid4())+'_'+str(curr_label)+'.jpg')
        cv2.imwrite(write_path, curr_video)
        image_list.append(write_path)
        # labels = label_temp.copy()

    dataframe = pd.DataFrame({'path ': image_list})
    dataframe.to_csv("./csv_data_read/"+mode+'_images.csv', index=False)

    print('Done Generating for ', mode)

pickle_path = '/home/ahmed/Desktop/dataset_skeleton/cross_subject_data'
out_path = '/home/ahmed/Desktop/dataset_skeleton'

augment(pickle_path, out_path, mode='train')