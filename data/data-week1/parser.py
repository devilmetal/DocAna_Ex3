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
        if sum_white == height:
            sum_white = int(height/2)
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
        if sum_white == height:
            sum_white = int(height/2)
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






# ---- DISSIMILARITY COMPUTATION /begin ---- #
# Source: http://nbviewer.ipython.org/github/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping/blob/master/K_Nearest_Neighbor_Dynamic_Time_Warping.ipynb
# Dynamic Time Warping
def dtw(ts1, ts2, d = lambda x,y: abs(x-y)):
    # Create cost matrix via broadcasting with large int
    ts1, ts2 = np.array(ts1), np.array(ts2)
    M, N = len(ts1), len(ts2)
    cost = sys.maxint * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts1[0], ts2[0])
    for i in xrange(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts1[i], ts2[0])

    for j in xrange(1, N):
        cost[0, j] = cost[0, j-1] + d(ts1[0], ts2[j])

    # Populate rest of cost matrix within window
    for i in xrange(1, M):
        for j in xrange(1, N):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts1[i], ts2[j])

    # Return DTW distance given window
    return cost[-1, -1]


#compare two custom distances via postal comparison
#OUTPUT : 0 = same (=), + = greater (>), - = less (<)
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
# ---- DISSIMILARITY COMPUTATION /end ---- #












# ------------------------- M A I N -------------------------#

# Keywords
kws = ["O-c-t-o-b-e-r", "s-o-o-n", "t-h-a-t"]
kws_path = "./WashingtonDB/keywords/"
# Words
# ws = ["274-05-02", "274-12-04", "273-33-05"]
ws_path = "WashingtonDB/words/"
# Ground truth
gt_file = "WashingtonDB/WashingtonDB.txt"


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


for kw in kws:
    fname = kws_path + kw + '.png'
    kw_pp = pp_col(fname)
    kw_pp_trans = pp_col_transition(fname)
    kw_up = up(fname)
    kw_lp = lp(fname)
    array = []
    for key in features:
        w_pp = features[key][0]
        w_pp_trans = features[key][1]
        w_up = features[key][2]
        w_lp = features[key][3]
        dist_pp = dtw(kw_pp,w_pp)
        dist_pp_trans = dtw(kw_pp_trans,w_pp_trans)
        dist_lp = dtw(kw_lp,w_lp)
        dist_up = dtw(kw_up,w_up)
        dist=[dist_pp,dist_pp_trans,dist_up,dist_lp]
        array.append([key,dist])

    #Sorting the array computed
    array.sort(compare)
    print "Ten first hits for keyword "+kw+"."
    print "=========================="
    print " "
    match = []
    nbr_hits = 10
    while len(match) != nbr_hits:
        elem = array.pop(0)
        if not elem[0] in match:
            print elem
            match.append(elem[0])
#     print " "
    precision, recall, fpr = [],[],[]
    for threshold in range(0,10):
        tp,fn,fp,tn = 0,0,0,0
        for i in range(len(match)):
            m = match[i].split('.', 1)[0]
            # print m
            if i <= threshold:
                if gt[m] == kw:#match
                    tp += 1
                else:
                    fp += 1
            else:
                if gt[m] == kw:
                    tn += 1
                else:
                    fn += 1
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
        for y in recall:
            if abs(1-x-y) == 0:
                # min_diff = abs(x-y)
                eer_x,eer_y = 1-x,y
    print "EER= " +str(eer_x)+ "," +str(eer_y)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, recall, 'r', eer_x, eer_y, 'ko')
    plt.show()
    print " "
