from PIL import Image
import math
import numpy as np
import collections
from matplotlib import pyplot as plt
import sys
import os
import random
import operator

BLACK = 0
WHITE = 255

def extract(fname):
    im = Image.open(fname)
    pixels = im.load()
    width, height = im.size
    img = []
    for i in range(width):
        line = []
        for j in range(height):
            cpixel = pixels[i,j]
            line.append(cpixel)
        img.append(line)
    return img


# ---- USE THESE METHODS TO DRAW HISTOGRAM /begin ---- #
# count the number of black pixel for each column
def get_vhist(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    hist = []

    for i in range(width):
        for j in range(height):
            if img[i][j] == BLACK:
                hist.append(i)

    bins = width
    plt.hist(hist, bins, [0,bins])
    plt.title('Vertical Histogram for ' + fname)
    plt.show()


# count the number of black pixel for each line
def get_hhist(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    hist = []

    for j in range(height):
        for i in range(width):
            if img[i][j] == BLACK:
                hist.append(j)

    bins = height
    plt.hist(hist, bins, [0,bins])
    plt.title('Horizontal Histogram for ' + fname)
    plt.show()
# ---- USE THESE METHODS TO DRAW HISTOGRAM /end ---- #





# ---- FEATURES EXTRACTION /begin ---- #
# Projection Profiling
def pp(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    pp = []
    # for each column count the number of black pixels
    for i in range(width):
        sum_black = 0
        for j in range(height):
            if img[i][j] == BLACK:
                sum_black += 1

        pp.append(sum_black)

    # normalize
    norm_pp = normalize(pp)
    return norm_pp


# TODO: finish extracting the upper profile
# Upper Profile
def up(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    up = []
    # for each column count the number of white pixels until 1st black pixel is encountered
    # TODO: need to detect where the word begins and where it ends
    for i in range(width):
        sum_white = 0
        j = 0
        while img[i][j] == WHITE and j < height:
            sum_white += 1

        if sum_white == height: # no black pixel encountered => take same value as the last one introduced
            sum_white = up[-1]

        up.append(sum_white)

    # normalize
    norm_up = normalize(up)
    return norm_up

def normalize(x):
    m = np.max(np.array(x))
    for i in range(len(x)):
        x[i] = float(x[i])/float(m)

    return x
# ---- FEATURES EXTRACTION /end ---- #






# ---- DISSIMILARITY COMPUTATION /begin ---- #
# Source: http://nbviewer.ipython.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
# Dynamic Time Warping
def dtw(kw, w, d = lambda x,y: abs(x-y)):
    # Create cost matrix via broadcasting with large int
    kw, w = np.array(kw), np.array(w)
    M, N = len(kw), len(w)
    cost = sys.maxint * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(kw[0], w[0])
    for i in xrange(1, M):
        cost[i, 0] = cost[i-1, 0] + d(kw[i], w[0])

    for j in xrange(1, N):
        cost[0, j] = cost[0, j-1] + d(kw[0], w[j])

    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(1, N):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(kw[i], w[j])

    # Return DTW distance given window
    return cost[-1, -1]
# ---- DISSIMILARITY COMPUTATION /end ---- #







# ------------------------- M A I N -------------------------#

# Keywords
kws = ["O-c-t-o-b-e-r", "s-o-o-n", "t-h-a-t"]
kws_path = "./WashingtonDB/keywords/"
# Words
ws = ["274-05-02", "274-12-04", "273-33-05"]
ws_path = "./WashingtonDB/words/"
# Ground truth
gt_file = "./WashingtonDB/WashingtonDB.txt"
gt = {}

# extract ground-truth in dictionnary for quick search
with open(gt_file) as f:
    cgt = [x.strip('\n ') for x in f.readlines()]
f.close()
for line in cgt:
    key = line.split(' ', 1)[0]
    label = line.split(' ', 1)[1]
    # label = label.split('_', 1)[0]
    gt[key] = label

# print gt


for kw in kws:
    dissimilarity = {}
    for w in ws:
        keyword = kws_path + kw + '.png'
        word = ws_path + w + '.png'
        dist = dtw(pp(keyword), pp(word))
        dissimilarity[w] = dist

    # ~rank list
    res = sorted(dissimilarity.items(), key=lambda x:x[1])
    # res = sorted(dissimilarity, key=dissimilarity.get)
    tp, fn, fp, tn = 0,0,0,0 # false/true positive/negative
    for i in res:
        if gt[i[0]] == kw:
            print "ok"
        else:
            print "not ok"
