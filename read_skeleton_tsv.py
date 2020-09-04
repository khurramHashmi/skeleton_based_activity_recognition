# MAKING VECTOR FOR EACH SKELETON AS A SEQUENCE TO GIVE INPUT TO TRANSFORMER


import numpy as np
import os
import sys
import pandas as pd
import itertools
import csv

input_dir = "data/train/" # train
input_dir = "data/val/" # val

#iterate over the whole dataset and return the max skeletons
#It is needed to make the size of each video sequence same
def find_max_skeleton(dir):
    max_skel_number = 0
    for filename in os.listdir(dir):
        skeleton_data = np.load(dir + filename, allow_pickle=True).item()
        max_skel_number = len(skeleton_data["nbodys"]) if len(skeleton_data["nbodys"]) > max_skel_number else max_skel_number
    return max_skel_number

frame_count = 0
connecting_joint = {
    "1": [2, 13, 17],
    "2": [21],
    "3": [21, 4],
    "4": [4],
    "5": [21, 6],
    "6": [7],
    "7": [8],
    "8": [22, 23],
    "9": [21, 10],
    "10": [11],
    "11": [12],
    "12": [24, 25],
    "13": [14],
    "14": [15],
    "15": [16],
    "16": [16],
    "17": [18],
    "18": [19],
    "19": [20],
    "20": [20],
    "21": [21],
    "22": [23],
    "23": [23],
    "24": [25],
    "25": [25]
}


file_count = 0

train_list = []

for filename in os.listdir(input_dir):
    video_sequence = []
    # reading skeleton data
    skeleton_data = np.load(input_dir + filename, allow_pickle=True).item()
    skeleton_sequence = []
    # if file_count >= 2:  # Limiting the sequences for now
    #     break
    skeleton_list = []

    #since label for all sequences is same
    output_label = skeleton_data["file_name"][len(skeleton_data["file_name"]) - 2]
    output_label += skeleton_data["file_name"][len(skeleton_data["file_name"]) - 1]
    print(output_label)
    try:
        for frame_count in range(len(skeleton_data["nbodys"])):
            # person_count = 0
            a_vec = []
            for person_count in range(skeleton_data["nbodys"][0]):
                joint_count = 1

                for joint_count in range(1, 26):
                    rgb_body_number = "rgb_body" + str(person_count)
                    connecting_joints = connecting_joint[str(joint_count)]

                    # Calculating distance between two fromes on each joints then take mean
                    x1 = int(skeleton_data[rgb_body_number][frame_count][joint_count - 1][0])
                    y1 = int(skeleton_data[rgb_body_number][frame_count][joint_count - 1][1])
                    a_vec.append(x1)
                    a_vec.append(y1)

                    # Connecting joints code for later use
                    #             for next_joint in connecting_joints:
                    # #                 next_joint = connecting_joint[joint_count]
                    # #                 print(next_joint)
                    #                 x2= int(skeleton_data[rgb_body_number][frame_count][next_joint-1][0])
                    #                 y2= int(skeleton_data[rgb_body_number][frame_count][next_joint-1][1])
                    #                 b_vec.append(x2)
                    #                 b_vec.append(y2)

                    joint_count += 1
                if skeleton_data["nbodys"][0] == 1:
                    a_vec += a_vec

            skeleton_sequence.append(a_vec)

        out_labels = []
        for i in range(find_max_skeleton(input_dir)):
            out_labels.append(int(output_label))

        video_sequence.append(out_labels)

        temp_vec = [0] * 100
        for i in range(len(skeleton_sequence),find_max_skeleton(input_dir)):  # padding 0s vector to the maximum size available
            skeleton_sequence.append(temp_vec)                                # making the video size for each activity same
        video_sequence.append(skeleton_sequence)

        train_list.append(video_sequence)
        # count = count + 1
    except:
        print("no value in the dataset")

    file_count += 1

# print(str(len(train_list[0][1])))
# print(train_list[0][1][0])
# Writing into csv in order to be read as a dataframe later on.
with open('data/val.tsv', 'w') as result_file:
    wr = csv.writer(result_file, quoting=csv.QUOTE_NONE, delimiter="\t")
    for line in train_list:
        wr.writerow((line[0],line[1]))
print("CSV file created with the name of train.csv")

