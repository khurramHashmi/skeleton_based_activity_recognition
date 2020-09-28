import re
import os
import csv
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd
from utils import create_dir
from random import randrange


def reduce_sequence(sequence, median_fr):
    
    num_frame_to_drop = len(sequence) - median_fr
    last_frame_to_drop = -1
    
    if num_frame_to_drop >= 2:
        win = 2
    else:
        win = 1
    # print('Total frames to drop: ', num_frame_to_drop)
    while num_frame_to_drop > 0:
        
        random_frame = randrange(len(sequence))
        # print('Frame to drop {} {} last frame to drop {} win {} '.format(random_frame, sequence[random_frame], last_frame_to_drop, win))

        if random_frame+win < len(sequence) and random_frame!=last_frame_to_drop:
            del sequence[random_frame]
            if win>1:
                del sequence[random_frame+1]
            last_frame_to_drop = random_frame+1 
            num_frame_to_drop = num_frame_to_drop - win
            if num_frame_to_drop < 2:
                win = 1
            #print('frames left to drop {} win {} len {}'.format(num_frame_to_drop, win, len(sequence)))
    #print(len(sequence))
    return sequence


def normalize(path='', out_path='./', pickle_output='./', part='train'):

    mean_list = []
    std_list = []
    data = pd.read_csv(path)
    print('Total examples {}'.format(len(data)))
    median_fr = 75 #int(np.median(data['frame_count']))
    print('Medain number of frames {}'.format(median_fr))
    file_path = []
    file_count = 0

    # iterate over data and check for videos length
    for j in tqdm.tqdm(range(len(data))):
        # if file_count > 11:
        #     break
        
        skeleton_data = pd.read_csv(data.iloc[j, 0], delimiter="\t", header=None)
        skeleton_sequence = skeleton_data[1].tolist()[0]
        skeleton_sequence = skeleton_sequence.replace("'", "").split(",")
        train_list = []
        data_source = []
        data_skel=[] # for each skeleton pose
        count = 1
        for d in skeleton_sequence:

            d = re.sub('\D', '', d)
            data_skel.append(int(d))

            if count == 100:
                data_source.append(data_skel)
                data_skel = []
                count = 1
            else:
                count+=1
        # try:
        skeleton_sequence = data_source # 100 * 100
        # print('original: {} {}'.format(len(skeleton_sequence), data.iloc[j, 1]))
        
        #final_skeletons = []
        # for skel in skeleton_sequence:
        # print('frame_count: ', data.iloc[j, 1])
        if len(skeleton_sequence) < median_fr:
            #print('adding')
            # need to add frames as total length has to be equal to median
            # start_pos = 0
            # window = 4
            # iterx = 0
            # while start_pos + window < len(skel):
            #     mean_insert = np.mean(skel[start_pos: start_pos+window])
            #     insert_idx = int(start_pos + window/2)
            #     # print('Iteration {} len {} start_pos {} insert_idx {}'.format(iterx, len(skeleton_sequence), start_pos, insert_idx))
            #     skel.insert(insert_idx, mean_insert)
            #     start_pos = insert_idx + 1
            #     iterx += 1
            # now reduce sequence to median_fr
            # print(skeleton_sequence)
            # reduced_sequence = reduce_sequence(skel, median_fr)
            diff = median_fr - len(skeleton_sequence)
            # copy the starting diff values
            for i in range(diff):
                skeleton_sequence.append(skeleton_sequence[i])
            reduced_sequence = skeleton_sequence
        else:
            #print('dropping')
            reduced_sequence = reduce_sequence(skeleton_sequence, median_fr)
            #print('after reduction: ', len(reduced_sequence))
        assert len(reduced_sequence) == median_fr, print('length not same ')
        # normalize data
        mean_list.append(np.mean(reduced_sequence))
        std_list.append(np.std(reduced_sequence))
        #final_skeletons.append(reduced_sequence)
        labels_ = list(skeleton_data[0])
        labels_ = [int(l) for l in labels_[0][1:-1].split(',')]
        labels_ = [labels_[0]] *  median_fr
        
        train_list.append(labels_)
        train_list.append(reduced_sequence)
        # print('Length {}'.format(len(labels_)))

        # if len(labels_) < median_fr:
        #     temp_labels = [labels_[-1]*median_fr]
        #     labels_.append(temp_labels)
        
        # if len(labels_) > median_fr:
        #     labels_ = labels_[:median_fr]
        
        # print('Length {}'.format(len(labels_)))
        
        assert len(reduced_sequence) == len(labels_), print('labels and data len not same {} {}'.format(len(reduced_sequence), len(labels_)))

        # normalized_data = pd.DataFrame(data={'0': labels_, '1': reduced_sequence})
        # print(normalized_data)
        # normalized_data.to_csv(os.path.join(os.path.basename(out_path, data.iloc[j, 0])), index=False)
        # print(normalized_data.head)
        # frame_count.append(data.iloc[j, 1])
        write_path = os.path.join(out_path, os.path.basename(data.iloc[j,0]))
        file_path.append(write_path)
        with open(write_path, 'w') as result_file:
                wr = csv.writer(result_file, quoting=csv.QUOTE_NONE, delimiter="\t")
                wr.writerow((train_list[0], train_list[1]))
    # except:
    #     print('error in file: ', data.iloc[j, 0])
    #     count_prob +=1
        file_count += 1
    # save std dev and mean as pickle file
    with open(os.path.join(pickle_output, os.path.basename(path).split('_')[0] + '_' + part+'_mean_std.pickle'), 'wb') as handle:
        pickle.dump({'mean': np.mean(mean_list), 'std': np.mean(std_list)}, handle)

    path_df = pd.DataFrame({'path':file_path})
    path_df.to_csv('./' + os.path.basename(path).split('_')[0] + '_'+part+'_norm_rgb.csv', index=False)

parser = argparse.ArgumentParser(description="Normalize dataset")
parser.add_argument("-p", "--part", default='train', help='Run on training data or val data')
parser.add_argument("-d", "--data_path",  default='./xsub_train_int.csv', help="Path to dataset to run script on")
parser.add_argument("-o", "--out_path",  default='./data/data_rgb_new/', help="Path to store files")
parser.add_argument("-po", "--pickle_output",  default='./data/data_rgb_new/', help="Path to store mean and std")

args = parser.parse_args()
out = os.path.basename(args.data_path).split('_')[0] + '_' + args.part+'_norm'
out_path = os.path.join(args.out_path, out)
print('Saving output to ',out_path)
create_dir(out_path)
# import sys
# sys.exit(0)
normalize(args.data_path, out_path, args.pickle_output, args.part)