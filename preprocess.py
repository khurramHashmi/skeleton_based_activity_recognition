import numpy as np 
import pandas as pd
from random import randrange


def reduce_sequence(sequence, median_fr):

    num_frame_to_drop = len(sequence) - 1 - median_fr                
    last_frame_to_drop = -1
    
    if num_frame_to_drop >= 2:
        win = 2
    else:
        win = 1
    print('Total frames to drop: ', num_frame_to_drop)
    while num_frame_to_drop > 0:
        
        random_frame = randrange(len(sequence))
        print('Frame to drop {} {} last frame to drop {} win {} '.format(random_frame, sequence[random_frame], last_frame_to_drop, win))

        if random_frame+win < len(sequence) and random_frame!=last_frame_to_drop:
            del sequence[random_frame]
            if win>1:
                del sequence[random_frame+1]
            last_frame_to_drop = random_frame+1 
            num_frame_to_drop = num_frame_to_drop - win
            if num_frame_to_drop < 2:
                win = 1
            print('frames left to drop {} win {} '.format(num_frame_to_drop, win))

    
    return sequence


def normalize(path='', median_fr=77, out_path='./'):

    # data = pd.read_csv(path)
    # # iterate over data and check for videos length
    # for j in data['path']:

    #     skeleton_data = pd.read_csv(j, delimiter="\t", header=None)
    count_prob = 0
    try:
        
        skeleton_sequence = [i for i in range(56)]
        #skeleton_sequence = list(skeleton_data[1]) # get coordinates from tsv file

        if len(skeleton_sequence) < median_fr:
            # need to add frames as total length has to be equal to median
            print('Increasing size')
            start_pos = 0
            window = 4
            stride=1
            iterx = 0
            base_len = len(skeleton_sequence)
            while start_pos + window < len(skeleton_sequence):
                mean_insert = np.mean(skeleton_sequence[start_pos: start_pos+window])
                insert_idx = int(start_pos + window/2)
                print('Iteration {} len {} start_pos {} insert_idx {}'.format(iterx, len(skeleton_sequence), start_pos, insert_idx))
                skeleton_sequence.insert(insert_idx, mean_insert)
                start_pos = insert_idx + 1
                iterx += 1
            # now reduce sequence to median_fr 
            print(skeleton_sequence)
            reduced_sequence = reduce_sequence(skeleton_sequence, median_fr)
        else:
            reduced_sequence = reduce_sequence(skeleton_sequence, median_fr)

        assert len(reduced_sequence) == median_fr, print('error in filename '+j)            
        # labels_ = list(skeleton_data[0])
        # if len(labels_)<len(reduced_sequence):
        #     temp_labels = [labels_[-1]*(len(reduced_sequence) - len(labels_))]
        #     labels_.append(temp_labels)

#        assert len(reduced_sequence) == len(labels_), print('labels and data len not same')
        print('REDUCED:', reduced_sequence)
        # normalized_data = pd.DataFrame(data={'0':labels_, '1':reduced_sequence})
        # normalized_data.to_csv(os.path.join(os.path.basename(j), out_path), index=False)

    except:
        count_prob +=1
        # print("no value in the dataset")

normalize()