import os
import tqdm
import pickle
import argparse
import re
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
            # print('frames left to drop {} win {} '.format(num_frame_to_drop, win))
    return sequence


def normalize(path='', out_path='./', pickle_output='./', part='train'):

    mean_list = []
    std_list = []
    data = pd.read_csv(path)
    print(data.head)
    print('Total examples {}'.format(len(data)))
    median_fr = int(np.median(data['frame_count']))
    print(median_fr)
    frame_count = []
    file_path = []
    file_count = 0
    # iterate over data and check for videos length
    for j in tqdm.tqdm(range(len(data))):
        if file_count > 1:
            break
        skeleton_data = pd.read_csv(data.iloc[j, 0], delimiter="\t", header=None)
        print(skeleton_data)

        skeleton_sequence = skeleton_data[1].tolist()[0]
        skeleton_sequence = skeleton_sequence.replace("'", "").split(",")

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
        final_skeletons = []

        for skel in skeleton_sequence:
            if len(skel) < median_fr:
                # need to add frames as total length has to be equal to median
                start_pos = 0
                window = 4
                iterx = 0
                while start_pos + window < len(skel):
                    mean_insert = np.mean(skel[start_pos: start_pos+window])
                    insert_idx = int(start_pos + window/2)
                    # print('Iteration {} len {} start_pos {} insert_idx {}'.format(iterx, len(skeleton_sequence), start_pos, insert_idx))
                    skel.insert(insert_idx, mean_insert)
                    start_pos = insert_idx + 1
                    iterx += 1
                # now reduce sequence to median_fr
                # print(skeleton_sequence)
                reduced_sequence = reduce_sequence(skel, median_fr)
            else:
                reduced_sequence = reduce_sequence(skel, median_fr)

            assert len(reduced_sequence) == median_fr, print('length not same ')
            # normalize data
            mean_list.append(np.mean(reduced_sequence))
            std_list.append(np.std(reduced_sequence))
            final_skeletons.append(reduced_sequence)

        labels_ = list(skeleton_data[0])
        if len(labels_) < median_fr:
            temp_labels = [labels_[-1]*median_fr]
            labels_.append(temp_labels)

        assert len(reduced_sequence) == len(labels_), print('labels and data len not same')

        normalized_data = pd.DataFrame(data={'0': labels_, '1': final_skeletons})
        normalized_data.to_csv(os.path.join(os.path.basename(data.iloc[j, 0]), out_path), index=False)

        frame_count.append(data.iloc[j, 1])
        file_path.append(data.iloc[j, 0])

    # except:
    #     print('error in file: ', data.iloc[j, 0])
    #     count_prob +=1
    file_count += 1
    # save std dev and mean as pickle file
    with open(os.path.join(pickle_output, part+'_mean_std.pickle', 'wb')) as handle:
        pickle.dump({'mean': np.mean(mean_list), 'std': np.mean(std_list)}, handle)

    path_df = pd.DataFrame({'path':file_path, 'frame_count':frame_count})
    path_df.to_csv('./xsub_'+part+'_norm_rgb.csv')

parser = argparse.ArgumentParser(description="Normalize dataset")
parser.add_argument("-p", "--part", default='train', help='Run on training data or val data')
parser.add_argument("-d", "--data_path",  default='./xsub_train_rgb.csv', help="Path to dataset to run script on")
parser.add_argument("-o", "--out_path",  default='./data', help="Path to store files")
parser.add_argument("-po", "--pickle_output",  default='./data', help="Path to store mean and std")

args = parser.parse_args()
out_path = os.path.join(args.out_path, args.part+'_norm')
create_dir(out_path)
normalize(args.data_path, out_path, args.pickle_output, args.part)