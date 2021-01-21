from tkinter import ttk
import tkinter as tk
from tkinter import *
import tkinter.font as font
import functools
fp = functools.partial
import json
import os
import numpy as np
import cv2

#For visualization imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.interactive(False)
plt.show(block=False)
# NTU RGB+D 60/120 Action Classes
actions = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down", # problem
    9: "standing up (from sitting position)", # problem
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something", # problem
    25: "reach into pocket",
    26: "hopping (one foot jumping)", # problem
    27: "jump up", # problem
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling", # problem
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person", # problem
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other", # problem
    60: "walking apart from each other", # problem
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person’s stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person’s ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}

class ScrollableFrame(ttk.Frame):
    def __init__(self, container,width, height, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, width=width, height=height)
        canvas.place(relx=0.5, rely=0.5)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )



        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side='left', padx=0, pady=0, anchor='ne')
        scrollbar.pack(side="right", fill="y")



# Thanks to chlutz214 for the usage update:
# if __name__ == "__main__":
#         # Set Up root of app
layers = []
root = tk.Tk()
root.geometry("2400x1000")
root.title("VerticalScrolledFrame Sample")
helv36 = font.Font(family='Helvetica', size=15, weight=font.BOLD)
headings = font.Font(family='Helvetica', size=25, weight=font.BOLD)

layers = []

#reading json file and the numpy skeleton files
path = "/home/hashmi/Desktop/dataset/activity_recognition/ntu_msg3f/xsub/val"
video_path = "/home/hashmi/Desktop/dataset/activity_recognition/NTURGBD-60_120/videos/nturgbd_rgb_s001/nturgb+d_rgb"
with open('result_predictions.json') as json_file:
    result_data = json.load(json_file)

files = os.listdir(path)


#two frames, one with correct class and another with in-correct classes
#Correct class Frame
correct_frame = ScrollableFrame(root, 600,800)
class_heading = tk.Label(correct_frame.scrollable_frame, text="Correct Activity Classification", font=headings, fg="green").pack()

def visualize_skeleton(file):

    ntu_skeleton_bone_pairs = tuple((i - 1, j - 1) for (i, j) in (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
    ))

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(figsize=(10, 35))
    ax = fig.gca(projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    file_class = file.split(".skeleton")[0]
    file_class = int(file_class[len(file_class) - 2] + file_class[len(file_class) - 1])

    def animate(skeleton):
        ax.clear()
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        for i, j in ntu_skeleton_bone_pairs:

            joint_locs = skeleton[:,[i,j]]
            # plot them
            ax.plot(joint_locs[0], joint_locs[1], joint_locs[2], color='blue')

        action_name = actions[file_class]

        plt.title('Frame #{} of 300 from {}\n (Action {}: {})'.format(skeleton_index[0], file.split("/")[-1].split(".skeleton")[0], file_class, action_name))
        skeleton_index[0] += 1
        return ax

    skeleton_index = [0]
    #reading the npyfile to pass skeleton in the animate function
    skeleton = np.load(file, mmap_mode='r')

    ani = FuncAnimation(fig, animate, skeleton[0])
    plt.waitforbuttonpress(0)  # this will wait for indefinite time
    plt.close(fig)
    plt.show()

def display_rgb_video(file):
    '''
        Displaying the RGB video of the selected sample
        Taking input the complete path of video location and file name
    '''
    cap = cv2.VideoCapture(file.split(".skeleton")[0] + "_rgb.avi")

    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow('frame')

def visualize_sample_options(skeleton_file, rgb_file):

    sample_window = tk.Tk()
    sample_window.geometry("500x300")
    sample_window.title("Visualization Option")
    label = tk.Label(sample_window, width="45", font=helv36, fg="blue", text= "how do you want to see the sample?")
    label.config(font=("Courier", 16))
    label.pack()

    skel_button = tk.Button(sample_window, width="45", font=headings, fg="green", text = "Visalize Skeleton", command=fp(visualize_skeleton, os.path.join(path, skeleton_file)))
    skel_button.pack(padx=0, pady=20)
    video_button = tk.Button(sample_window, width="45", font=headings, fg="green", text="Visalize RGB Video", command=fp(display_rgb_video, os.path.join(path, rgb_file)))
    video_button.pack(padx=0, pady=20)


def create_correct_class(num):
    '''
        For inserting samples that are correctly classified
        compare predicted labels with the orignals
    '''
    top = Toplevel()
    layers.append(top)

    top.title("Correct Predictions")
    top.geometry('800x800')
    sample_frame = ScrollableFrame(top, 600, 800)

    for file_mame in files:
        file = file_mame.split(".npy")[0]
        file_class = file.split(".skeleton")[0]
        file_class = int(file_class[len(file_class)-2] + file_class[len(file_class)-1])

        if file_class == num:
            samples_btn = tk.Button(sample_frame.scrollable_frame, width="45", font=helv36, fg="blue", text="Sample : "+file,
                                    command=fp(visualize_sample_options, os.path.join(path,file_mame), os.path.join(video_path,file_mame))).pack()

    sample_frame.pack(side='left', padx=0, pady=20, anchor='n')

# x,y = 5,35
for i in range(1,61):
    btnMyButton = tk.Button(correct_frame.scrollable_frame, text=actions[i], fg="red", font=helv36, width="45", command=fp(create_correct_class, i))
    btnMyButton.pack(pady=5)

correct_frame.pack(side='left', padx=80, pady=80, anchor='ne')

incorrect_frame = ScrollableFrame(root, 600,800)

mis_class_heading = tk.Label(incorrect_frame.scrollable_frame, text="Incorrect Activity Classification", font=headings, fg="green").pack()

def create_incorrect_class(num):
    '''
        For inserting samples that are misclassified
        compare predicted labels with the orignals
    '''

    top = Toplevel()
    layers.append(top)

    top.title("List of Incorrect Predictions")
    top.geometry('800x800')
    sample_frame = ScrollableFrame(top, 800, 800)

    for file_mame in files:
        file = file_mame.split(".npy")[0]
        file_class = file.split(".skeleton")[0]
        file_class = int(file_class[len(file_class)-2] + file_class[len(file_class)-1])

        if file_class == result_data[file]:
            samples_btn = tk.Button(sample_frame.scrollable_frame, width="55", font=helv36, fg="blue", text=file.split(".skeleton")[0] + ": "+actions[file_class],
                                    command=fp(visualize_sample_options, os.path.join(path,file_mame), os.path.join(video_path,file_mame))).pack()

    sample_frame.pack(side='left', padx=0, pady=20, anchor='n')

for i in range(1,61):
    btnMyButton = tk.Button(incorrect_frame.scrollable_frame, text=actions[i], fg="red", font=helv36, width="45", command=fp(create_incorrect_class, i))
    btnMyButton.pack(pady=5)


incorrect_frame.pack(side='left', padx=80, pady=80, anchor='ne')

sample_location = ScrollableFrame(root, 600,800)
mis_class_heading = tk.Label(sample_location.scrollable_frame, text="Misclassified Sample location", font=headings, fg="green").pack()



def display_incorrect_class(num):
    '''
        For inserting samples that are misclassified
        display their misclassified/confused classes
        on the basif of actual class
    '''

    top = Toplevel()
    layers.append(top)

    top.title("List of Incorrect Predictions")
    top.geometry('800x800')
    sample_frame = ScrollableFrame(top, 800, 800)

    for file_mame in files:

        file = file_mame.split(".npy")[0]
        file_class = file.split(".skeleton")[0]
        file_class = int(file_class[len(file_class)-2] + file_class[len(file_class)-1])

        # Trying to find the wrong predicted class of each sample
        if file_class != result_data[file] and file_class == num:
            samples_btn = tk.Button(sample_frame.scrollable_frame, width="55", font=helv36, fg="blue", text=file.split(".skeleton")[0] + ": "+actions[result_data[file]],
                                    command=fp(visualize_sample_options, os.path.join(path,file_mame), os.path.join(video_path,file_mame))).pack()

    sample_frame.pack(side='left', padx=0, pady=20, anchor='n')

for i in range(1,61):
    btnMyButton = tk.Button(sample_location.scrollable_frame, text=actions[i], fg="red", font=helv36, width="45", command=fp(display_incorrect_class, i))
    btnMyButton.pack(pady=5)

sample_location.pack(side='right', padx=80, pady=80, anchor='ne')

# Run mainloop to start app
root.mainloop()

