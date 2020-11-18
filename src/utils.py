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