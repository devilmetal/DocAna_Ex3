from PIL import Image
import math
import numpy as np
import collections
from matplotlib import pyplot as plt
import sys
import os
import random
import operator
import pickle
BLACK = 0
WHITE = 255
gt = {}

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

# ---- FEATURES EXTRACTION /begin ---- #
# Projection Profiling Columns
def pp_col(fname):
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


# Projection Profiling Lines
def pp_line(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    pp = []
    # for each column count the number of black pixels
    for j in range(height):
        sum_black = 0
        for i in range(width):
            if img[i][j] == BLACK:
                sum_black += 1

        pp.append(sum_black)

    # normalize
    norm_pp = normalize(pp)
    return norm_pp

# Projection Profiling Columns Transition B/W
def pp_col_transition(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    pp = []
    # for each column count the number of black pixels
    for i in range(width):
        transitions = 0
        white = True
        for j in range(height):
            if white:
                if img[i][j] == BLACK:
                    transitions += 1
                    white = False
            else:
                if img[i][j] == WHITE:
                    transitions += 1
                    white = False

        pp.append(transitions)
    return pp

# Projection Profiling Lines Transition B/W
def pp_line_transition(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    pp = []
    # for each column count the number of black pixels
    for j in range(height):
        transitions = 0
        white = True
        for i in range(width):
            if white:
                if img[i][j] == BLACK:
                    transitions += 1
                    white = False
            else:
                if img[i][j] == WHITE:
                    transitions += 1
                    white = False

        pp.append(transitions)
    return pp


# Upper Profile
def up(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    up = []
    # for each column count the number of white pixels until 1st black pixel is encountered
    # TODO: need to detect where the word begins and where it ends
    for i in range(width):
        sum_white = 0
        j=0
        while j < height and img[i][j] == WHITE:
            j+=1
            sum_white += 1
        up.append(sum_white)
    # normalize
    norm_up = normalize(up)
    return norm_up

def lp(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    lp = []
    # for each column count the number of white pixels until 1st black pixel is encountered
    # TODO: need to detect where the word begins and where it ends
    for i in range(width):
        sum_white = 0
        j=height-1
        while j > 0 and img[i][j] == WHITE:
            j-=1
            sum_white += 1
        lp.append(sum_white)
    # normalize
    norm_up = normalize(lp)
    return norm_up

def normalize(x):
    m = np.max(np.array(x))
    for i in range(len(x)):
        x[i] = float(x[i])/float(m)

    return x
# ---- FEATURES EXTRACTION /end ---- #

#compare two custom distances via postal comparison
#OUTPUT : 0 = same (=), + = greater (>), - = less (<)
def compare(dist1,dist2):
    distance = 0
    for i in range(len(dist1[1])):#-1):#because last element is the label
        if dist1[1][i] < dist2[1][i]:
            distance -= 1
        if dist1[1][i] > dist2[1][i]:
            distance +=1
    #print distance
    if distance > 0:
        return 1
    elif distance < 0:
        return -1
    else:
        return 0
# ---- DISSIMILARITY COMPUTATION /end ---- #


# ------------------------- M A I N -------------------------#

# Keywords
kws = ["O-c-t-o-b-e-r", "s-o-o-n", "t-h-a-t"]
kws_path = "./WashingtonDB/keywords/"
# Words
ws = ["274-05-02", "274-12-04", "273-33-05"]
ws_path = "WashingtonDB/_lines/"
crop_path = "WashingtonDB/crops/"
# Ground truth
gt_file = "WashingtonDB/LinesWashington.txt"

#PART 1 Parse all lines to obtain vector feature for each column
features={}
dict_path = './features.dict'
#Check if dict exists
if os.path.isfile(dict_path):
    features = pickle.load(open(dict_path,'rb'))
else:
    for path, subdirs, files in os.walk(ws_path):
        #checking files
        for file in files:
            fname = ws_path+file
            pp = pp_col(fname)
            pp_trans = pp_col_transition(fname)
            upperp = up(fname)
            lowerp = lp(fname)
            vector = [pp,pp_trans,upperp,lowerp]
            features[str(file)]=vector
    #dump dict
    pickle.dump(features,open(dict_path,'wb'))
#PART2 PARSE THE WHOLE DICTIONARY VECTORS WITH A GIVEN KEYWORD FEATURE VECTOR
windows = 30 # 30px width for the sliding windows
for kw in kws:
    fname = kws_path + kw + '.png'
    kw_pp = pp_col(fname)
    kw_pp_trans = pp_col_transition(fname)
    kw_up = up(fname)
    kw_lp = lp(fname)
    array = []
    kw_width = len(kw_pp)
    for key in features:
            width = len(features[key][0])
            for i in range(width):
                if i % windows == 0 and i+kw_width < width:
                    crop_pp = features[key][0][i:i+kw_width]
                    crop_pp_trans = features[key][1][i:i+kw_width]
                    crop_lp = features[key][2][i:i+kw_width]
                    crop_up = features[key][3][i:i+kw_width]
                    dist_pp = sum([abs(x-y) for x, y in zip(crop_pp, kw_pp)])
                    dist_pp_trans = sum([abs(x-y) for x, y in zip(crop_pp_trans, kw_pp_trans)])
                    dist_lp = sum([abs(x-y) for x, y in zip(crop_lp, kw_lp)])
                    dist_up = sum([abs(x-y) for x, y in zip(crop_up, kw_up)])
                    dist=[dist_pp,dist_pp_trans,dist_lp,dist_up]
                    array.append([key,dist])


    #Sorting the array computed
    array.sort(compare)
    print "Ten first hits for keyword "+kw+"."
    print "=========================="
    print " "
    for i in range(10):
        print array[i]
    print " "
