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

# Projection Profiling Columns Ratio
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
        sum_black = sum_black/float(height)
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
                    white = True

        pp.append(transitions)
    return pp
'''
# Upper Profile
def up(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    up = []

    # for each column count the number of white pixels until 1st black pixel is encountered
    # need to detect where the word begins and where it ends
    begin, end = False, False
    begin_i,end_i = 0,0
    for i in range(width):
        for j in range(height):
            if img[i][j] == BLACK and not begin:
                begin_i = i
                begin = True
                break
    for i in reversed(range(width)):
        for j in range(height):
            if img[i][j] == BLACK and not end:
                end_i = i
                end = True
                break

    for i in range(begin_i,end_i):
        sum_white = 0
        j=0
        while j < height and img[i][j] == WHITE:
            j+=1
            sum_white += 1

        if sum_white == height: # no black pixel encountered => take same value as the last one introduced
            sum_white = up[-1]

        up.append(sum_white)

    # normalize
    norm_up = normalize(up)
    return norm_up

# Lower Profile
def lp(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    lp = []
    # for each column count the number of white pixels until 1st black pixel is encountered
    # need to detect where the word begins and where it ends
    begin, end = False, False
    begin_i,end_i = 0,0
    for i in range(width):
        for j in range(height):
            if img[i][j] == BLACK and not begin:
                begin_i = i
                begin = True
                break
    for i in reversed(range(width)):
        for j in range(height):
            if img[i][j] == BLACK and not end:
                end_i = i
                end = True
                break
    for i in range(begin_i,end_i):
        sum_white = 0
        j=height-1
        while j < height and img[i][j] == WHITE:
            j-=1
            sum_white += 1

        if sum_white == height: # no black pixel encountered => take same value as the last one introduced
            sum_white = lp[-1]

        lp.append(sum_white)

    # normalize
    norm_up = normalize(lp)
    return norm_up
'''
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

def compare(dist1,dist2):
    #EUCLIDIAN DISTANCE
    sum1=0
    for i in range(len(dist1[1])):
        sum1+=math.pow(dist1[1][i], 2)
    sum1=math.sqrt(sum1)
    sum2=0
    for i in range(len(dist2[1])):
        sum2+=math.pow(dist2[1][i], 2)
    sum2=math.sqrt(sum2)

    if sum1>sum2:
        return 1
    elif sum1<sum2:
        return -1
    else:
        return 0






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
gt = {}

# extract ground-truth in dictionnary for quick search
with open(gt_file) as f:
    cgt = [x.strip('\n ') for x in f.readlines()]
f.close()
for line in cgt:
    key = line.split(' ', 1)[0]
    label = line.split(' ', 1)[1]
    label = label.split('|')
    # label = label.split('_', 1)[0]
    gt[key] = label

# print gt

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
            print "Encoding file "+ws_path+file
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
windows = 1 # 30px width for the sliding windows
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
                    dist_pp = sum([abs(x-y) for x, y in zip(crop_pp, kw_pp)])/len(crop_pp)
                    dist_pp_trans = sum([abs(x-y) for x, y in zip(crop_pp_trans, kw_pp_trans)])/len(crop_pp_trans)
                    dist_lp = sum([abs(x-y) for x, y in zip(crop_lp, kw_lp)])/len(crop_lp)
                    dist_up = sum([abs(x-y) for x, y in zip(crop_up, kw_up)])/len(crop_up)
                    dist=[dist_pp,dist_pp_trans,dist_lp,dist_up]
                    array.append([key,dist])

    #Sorting the array computed
    array.sort(compare)
    print "Ten first hits for keyword "+kw+"."
    print "=========================="
    print " "
    match = []
    nbr_hits = 150
    while len(match) != nbr_hits:
        elem = array.pop(0)
        if not elem[0] in match:
            print elem
            match.append(elem[0])
#     print " "
    precision, recall, fpr = [],[],[]
    for threshold in range(0,150):
        tp,fn,fp,tn = 0,0,0,0
        for i in range(len(match)):
            m = match[i].split('.', 1)[0]
            if i <= threshold:
                seen = False
                for word in gt[m]:
                    if word == kw and not seen:#keyword appears on the checked line (count line only once)
                        tp += 1
                        seen = True
                if not seen:#keyword is not on the checked line
                    fp += 1
            else:
                seen = False
                for word in gt[m]:
                    if word == kw and not seen:
                        tn += 1
                if not seen:
                    fn += 1
        # TODO: handle case when divided by 0 (should not happen if we take the complete ranked list)
        #print "tn "+str(tn)
        #print "tp "+str(tp)
        #print "fn "+str(fn)
        #print "fp "+str(fp)
        try:
            precision_str = float(tp)/(float(tp)+float(fp))
        except:
            precision_str = 0.0
        try:
            recall_str = float(tp)/(float(tp)+float(fn))
        except:
            recall_str = 0.0
        try:
            fpr_str = float(fp)/(float(fp)+float(tn))
        except:
            fpr_str = 0.0
        print "T"+str(threshold)+" precision= "+str(precision_str)+" recall= "+str(recall_str)+" FPR= "+str(fpr_str)
        precision.append(precision_str)
        recall.append(recall_str)
        fpr.append(fpr_str)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision, 'r')
    plt.show()

    eer_x,eer_y = 0,0
    min_diff = 1000
    for x in fpr:
        for y in precision:
            if abs(x-y) < min_diff:
                min_diff = abs(x-y)
                eer_x,eer_y = x,y
    print "EER= " +str(eer_x)+ "," +str(eer_y)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, precision, 'r', eer_x, eer_y, 'ko')
    plt.show()
    print " "
