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


def augment(pickle_path, out_path, mode='train'):

    out_path = out_path + mode + '_images/'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if mode == 'train':
        input_features = pickle.load(open(pickle_path + "/dsamp_train.p", "rb"))
        labels = pickle.load(open("labels_proper_train.p", "rb"))

    else:
        input_features = pickle.load(open(pickle_path + "/dsamp_test.p", "rb"))
        labels = pickle.load(open("./labels_proper_test.p", "rb"))

    label_temp = labels.copy()
    features = []
    image_list = []

    for i in tqdm.tqdm(range(len(labels))):

        if input_features[i].shape[0] < 50:
            inputs = np.zeros((50, 75), dtype=float)
            inputs[:input_features[i].shape[0], :] = np.copy(input_features[i])
            input_features[i] = inputs

        video = np.reshape(input_features[i], (50, 25, 3))
        curr_label = labels[i]
        labels.pop(i)
        indices = [i for i, x in enumerate(labels) if x == curr_label]

        new_base = None

        for j in range(7):

            idx = random.sample(indices, 1)[0]

            corresponding_ex = input_features[idx]
            if corresponding_ex.shape[0] < 50:
                inputs = np.zeros((50, 75), dtype=float)
                inputs[:corresponding_ex.shape[0], :] = np.copy(corresponding_ex)
                corresponding_ex = inputs

            corresponding_ex = add_noise("gauss",corresponding_ex.reshape((50,25,3)))

            if j < 3:
                video = np.hstack((video, corresponding_ex))
            # elif j==1:
            #     new_base = corresponding_ex
            else:
                new_base = corresponding_ex if new_base is None else np.hstack((new_base, corresponding_ex))

        img = video
        img = np.vstack((img, new_base))
        # features.append(img)
        # temp = 255 * img
        write_path = os.path.join(out_path, str(uuid.uuid4())+'_'+str(curr_label)+'.jpg')
        cv2.imwrite(write_path, img)
        image_list.append(write_path)
        labels = label_temp.copy()

    dataframe = pd.DataFrame({'path ': image_list})
    dataframe.to_csv("./"+mode+'_images.csv', index=False)

    # pickle.dump(features, open(os.path.join(out_path, 'big_'+mode+'.p'), 'wb'))
    print('Done Generating for ', mode)

pickle_path = '/home/hashmi/Desktop/activity_recognition/transformer_activity/pickles'
out_path = './'

augment(pickle_path, out_path, mode='test')
# labels = pickle.load(open(pickle_path + "/test_lab.p", "rb"))
# temp = []
#
# for j in labels:
#     temp.append(np.argmax(j))
#
# print(temp)
# pickle.dump(temp, open('labels_proper_test.p', 'wb'))