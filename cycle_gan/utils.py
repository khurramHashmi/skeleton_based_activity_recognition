import os
import sys
import cv2
import errno
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def latex_writer(labels, values, index):

    '''
        Function to create latext table.
    args:
        -labels: a list containing column names or 1D Matrix
        -values: a 2D Matrix containing data
    returns:
        -image file containing confusion matrix
    '''
    # add assert statement here
    df = pd.DataFrame(data=values, columns=labels)
    latex_string = "\\begin{tabular}{l|" + "|".join(["c"] * len(df.columns)) + "}\n"
    latex_string += "\hline \n"
    latex_string += "{} & " + " & ".join([str(x) for x in labels]) + "\\\\\n"
    for i, row in df.iterrows():
        latex_string += str(index[i]) + " & " + " & ".join([str(x) for x in row.values]) + " \\\\\n"
    latex_string += "\hline \n"
    latex_string += "\\end{tabular}"
    
    #latext_string = df.to_latex(index=False, col_space=3, bold_rows=True, caption='Testing')
    return latex_string

def idx_class(classes, preds):
    classes_ = []
    for j in preds:
        classes_.append(classes[j])
    return classes_

def write_to_graph(labels, value,writer,epoch):
    
    writer.add_scalar(labels, value, epoch)

def barplot(x, y, data, axlabel, graph_out): #function generating histograms

    plot = sns.barplot(x=x,y=y, data=data)   
    plot.figure.savefig(os.path.join(graph_out,axlabel+".png"))

def create_dir(path):
    
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def visualize_skeleton(path, pred_label, out_path):
    
    '''
        Function to draw skeletons for visualization purposes
        args:
            -path: list containing path to current batch example files
            -pred_label: list containing labels for current batch examples
            -out_path: path to store visualizations
        returns:
            -images
    '''
    connecting_joint = {
    "1": [2,13,17],
    "2": [21],
    "3": [21,4],
    "4": [4],
    "5": [21,6],
    "6": [7],
    "7": [8],
    "8": [22,23],
    "9": [21,10],
    "10": [11],
    "11": [12],
    "12": [24,25],
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
    raw_file_path = '/netscratch/m_ahmed/datasets/dataset_skeleton/raw_npy'
    
    for filename, label in zip(path, pred_label):
        count_ = 0
        name = os.path.basename(filename[:-4])
        current_path = os.path.join(raw_file_path, name)
        skeleton_data = np.load(current_path, allow_pickle=True).item()
        output_label = skeleton_data["file_name"][len(skeleton_data["file_name"]) - 2]
        output_label += skeleton_data["file_name"][len(skeleton_data["file_name"]) - 1]

        for frame_count in range(len(skeleton_data["nbodys"])):
            print('creating skeleton')
            person_count=0
            #img2 = image
            img2 = np.zeros((1080,1920,3), np.uint8)

            img2.fill(255)
            
            for person_count in range(skeleton_data["nbodys"][0]):
                joint_count = 1
                colorR = 255
                colorG = 127
                colorB = 127
                minx = 1800
                maxx = 0
                miny = 1080
                maxy = 0
                a_vec = []
                b_vec = []
                        
                for joint_count in range(1,25):
        #             print(joint_count)
                    rgb_body_number = "rgb_body" + str(person_count)
                    connecting_joints = connecting_joint[str(joint_count)]
                    
                    # Calculating distance between two fromes on each joints then take mean
                    x1 = int(skeleton_data[rgb_body_number][frame_count][joint_count-1][0])
                    y1 = int(skeleton_data[rgb_body_number][frame_count][joint_count-1][1])
                    
                    if frame_count >=1 : 
                        x1_frame2 = int(skeleton_data[rgb_body_number][frame_count-1][joint_count-1][0])
                        y1_frame2 = int(skeleton_data[rgb_body_number][frame_count-1][joint_count-1][1])
        #                 print(x1,y1)
        #                 print(x1_frame2,y1_frame2)

                        #Calculating euclidean distance between the current frame 
                        #and the previous frame for each joint
                        dist += np.sum((x1 - x1_frame2)**2 + (y1 - y1_frame2)**2)
                        
                        #Now Calculating Consine Similarity
                        a_vec.append(x1)
                        a_vec.append(y1)
                        b_vec.append(x1_frame2)
                        b_vec.append(y1_frame2)

                    #Calculating points to crop the image code here
                    if x1 < minx:
                        minx = x1
                    elif x1 > maxx:
                        maxx = x1
                    if y1 < miny:
                        miny = y1
                    elif y1 > maxy:
                        maxy = y1
                    #Calculating points to crop the image code here ends
            
                    for next_joint in connecting_joints:
        #                 next_joint = connecting_joint[joint_count]
        #                 print(next_joint)
                        x2= int(skeleton_data[rgb_body_number][frame_count][next_joint-1][0])
                        y2= int(skeleton_data[rgb_body_number][frame_count][next_joint-1][1])
                        
                        cv2.line(img2, (x1,y1), (x2,y2), (colorB,colorG,colorR), (10))
                        colorR = 1+next_joint*6
                        colorG = 255-next_joint*7
                        colorB = 1+next_joint*5
                        
                    joint_count+=1

                person_count+=1
                # if frame_count >=1:
                #     print("Euclidean Distance: ",int(dist))
                #     cos_sim = 1 - spatial.distance.cosine(a_vec, b_vec)
                #     print("Cosine Similarity : ",str(cos_sim))
                
            #   Incase if images with distances should be displayed then uncomment this code
                font = cv2.FONT_HERSHEY_SIMPLEX
                minx-=100
                maxx+=100
                miny-=50
                maxy+=50
                cv2.putText(img2,'Predicted Label: '+str(label),(minx+10,miny+30), font, 1,(0,0,255),1)
                cv2.putText(img2,'Ground Truth Label: '+str(output_label), (1080-250, 1920-250), font, 1,(0,0,255),1)
                dist=0
                img2 = img2[miny:maxy, minx:maxx]
                path_to_write = os.path.join(out_path, os.path.basename(filename).split('.')[0]+'_'+str(count_)+'.png')
                count_+=1
                print(path_to_write)
                cv2.imwrite(path_to_write, img2)     # save frame as JPEG file
                #print ('Read a new frame: ', success)
                #count = count + 1

def draw_confusion_matrix(y_true, y_pred, filename, labels, ymap=None, figsize=(25, 25)):

    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.savefig(filename+'.png')

    def _rotation(data, alpha=0, beta=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

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

    def _normal_skeleton(data):
        #  use as center joint
        center_joint = data[0, :, 0, :]

        center_jointx = np.mean(center_joint[:, 0])
        center_jointy = np.mean(center_joint[:, 1])
        center_jointz = np.mean(center_joint[:, 2])

        center = np.array([center_jointx, center_jointy, center_jointz])
        data = data - center

        return data

    def visualize_ex(ex, out_dir, label, seg):

        trunk_joints = [0, 1, 20, 2, 3]
        arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
        leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
        body = [trunk_joints, arm_joints, leg_joints]
        init_vertical = 20
        init_horizon = -45
        x_rotation = None
        y_rotation = None
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(init_vertical, init_horizon)
        plt.ion()
        temp = ex
        coords = np.reshape(temp, (seg,25,3))

        coords = coords[:, :, :,np.newaxis]
        data = np.transpose(coords, (3, 0, 1,2 ))

        # data rotation
        if (x_rotation is not None) or (y_rotation is not None) and False:

            if x_rotation > 180 or y_rotation > 180:
                raise Exception("rotation angle should be less than 180.")

            else:
                print(" Angle : ", x_rotation)
                data = _rotation(data, x_rotation, y_rotation)
        data = _normal_skeleton(data)
        data = data[0]
        # show every frame 3d skeleton
        for frame_idx in range(0, data.shape[0], 5):

            plt.cla()
            plt.title("Frame: {}".format(frame_idx))

            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-0.8, 0.8])

            x = data[frame_idx, :, 0]
            y = data[frame_idx, :, 1]
            z = data[frame_idx, :, 2]

            for part in body:
                x_plot = x[part]
                y_plot = y[part]
                z_plot = z[part]
                ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            if out_dir is not None:
                save_pth = os.path.join(out_dir, '{}_{}.png'.format(label,frame_idx))
                plt.savefig(save_pth)