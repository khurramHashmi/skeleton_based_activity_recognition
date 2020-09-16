# MAKING VECTOR FOR EACH SKELETON AS A SEQUENCE TO GIVE INPUT TO TRANSFORMER
import os
import sys
import csv
import argparse
import itertools
import numpy as np
import pandas as pd

def gen_data(args):

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    # list of ids to be used for training subject wise view.
    training_subjects = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
    ]
    location_row = []
    #iterate over the whole dataset and return the max skeletons
    #It is needed to make the size of each video sequence same
    # def find_max_skeleton(dir):
    #     # dir="data/train/"
    #     max_skel_number = 0
    #     for filename in os.listdir(dir):
    #         skeleton_data = np.load(dir + filename, allow_pickle=True).item()
    #         max_skel_number = len(skeleton_data["nbodys"]) if len(skeleton_data["nbodys"]) > max_skel_number else max_skel_number
    #     return max_skel_number

    max_skel_num = 100
    check_max = 0
    # print(find_max_skeleton(input_dir))
    # input_dir = "./data/cross_view/val/" # val
    # print(find_max_skeleton(input_dir))

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
    count_prob = 0
    path_list = []
    current_length = []

    for filename in os.listdir(args.data_path):

        if file_count >= 500: # 1000 for training
            break
        train_list = []
        video_sequence = []
        # reading skeleton data

        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if args.benchmark=='xsub':
            istraining = (subject_id in training_subjects)

            if args.part == 'train':
                issample = istraining
            elif args.part == 'val':
                issample = not (istraining)
            else:
                raise ValueError()

            if not issample:
                continue # skip the file since its not part of either training or validation

        skeleton_data = np.load(args.data_path + filename, allow_pickle=True).item()
        skeleton_sequence = []

        # if file_count >= 2:  # Limiting the sequences for now
        #     break
        skeleton_list = []
    
        #since label for all sequences is same
        output_label = skeleton_data["file_name"][len(skeleton_data["file_name"]) - 2]
        output_label += skeleton_data["file_name"][len(skeleton_data["file_name"]) - 1]
        current_length.append(len(skeleton_data["nbodys"]))
    
        try:
            for frame_count in range(len(skeleton_data["nbodys"])):
                # person_count = 0
                a_vec = []
                for person_count in range(skeleton_data["nbodys"][0]):
                    joint_count = 1

                    for joint_count in range(1, 26):
                        rgb_body_number = "skel_body" + str(person_count) # now writing skeletons
                        connecting_joints = connecting_joint[str(joint_count)]

                        # Calculating distance between two fromes on each joints then take mean
                        x1 = (skeleton_data[rgb_body_number][frame_count][joint_count - 1][0])
                        y1 = (skeleton_data[rgb_body_number][frame_count][joint_count - 1][1])
                        z1 = (skeleton_data[rgb_body_number][frame_count][joint_count - 1][2])
                        a_vec.append(x1)
                        a_vec.append(y1)
                        a_vec.append(z1)

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
            for i in range(max_skel_num): #instead of calling function again, just replace it with a variable
                out_labels.append(int(output_label))

            video_sequence.append(out_labels)

            if len(skeleton_sequence) > check_max:
                check_max = len(skeleton_sequence)
                print(check_max)

            temp_vec = [0] * 150

            for i in range(len(skeleton_sequence),max_skel_num):  # padding 0s vector to the maximum size available
                skeleton_sequence.append(temp_vec)                 # making the video size for each activity same

            video_sequence.append(skeleton_sequence[0:150])

            train_list.append(video_sequence)
            # count = count + 1

            file_count += 1
            # print(str(len(train_list[0][1])))
            # print(train_list[0][1][0])
            # Writing into csv in order to be read as a dataframe later on.
            write_path = os.path.join(args.out_path, os.path.basename(filename) + '.tsv')
            path_list.append(write_path)

            with open(write_path, 'w') as result_file:
                wr = csv.writer(result_file, quoting=csv.QUOTE_NONE, delimiter="\t")
                for line in train_list:
                    wr.writerow((line[0], line[1]))
            # print("CSV file created with the name of " + os.path.basename(filename) + '.tsv')

        except:
            count_prob +=1
            # print("no value in the dataset")
    
    print("PROBLEM IN FILES : ", str(count_prob))
    df = pd.DataFrame(data={'path':pd.Series(path_list), 'frame_count':current_length})
    df.to_csv(os.path.join('./', args.benchmark+'_'+args.part+'_skel_small.csv'), index=False)

parser = argparse.ArgumentParser(description="Dataset Generator for Skeleton Classification Model")
parser.add_argument("-d", "--data_path",default='./data/raw_npy/', help="Path to folder containing data")
parser.add_argument("-o", "--out_path",default='./data/data_skel/val/', help="Path to create tsv file")
parser.add_argument("-b", "--benchmark", default='xsub', help="Camera view or subject view data generation parameter. xview for camera view, xsub for subject view." )
parser.add_argument("-p", "--part", default='val', help="Create data for train or validation")
args = parser.parse_args()
gen_data(args)