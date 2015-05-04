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
    # normalize
    norm_pp = normalize(pp)
    return norm_pp


# Upper Profile
def up(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    up = []
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
    # img = extract(fname)
    # width, height = len(img), len(img[0])
    # up = []
    #
    # # for each column count the number of white pixels until 1st black pixel is encountered
    # # need to detect where the word begins and where it ends
    # begin, end = False, False
    # begin_i,end_i = 0,0
    # for i in range(width):
    #     for j in range(height):
    #         if img[i][j] == BLACK and not begin:
    #             begin_i = i
    #             begin = True
    #             break
    # for i in reversed(range(width)):
    #     for j in range(height):
    #         if img[i][j] == BLACK and not end:
    #             end_i = i
    #             end = True
    #             break
    #
    # for i in range(begin_i,end_i):
    #     sum_white = 0
    #     j=0
    #     while j < height and img[i][j] == WHITE:
    #         j+=1
    #         sum_white += 1
    #
    #     if sum_white == height: # no black pixel encountered => take same value as the last one introduced
    #         sum_white = up[-1]
    #
    #     up.append(sum_white)

    # normalize
    norm_up = normalize(up)
    return norm_up

def lp(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    lp = []
    for i in range(width):
        sum_white = 0
        j=height-1
        while j >= 0 and img[i][j] == WHITE:
            j-=1
            sum_white += 1
        if sum_white == height:
            sum_white = int(height/2)
        lp.append(sum_white)
    # normalize
    norm_up = normalize(lp)
    return norm_up
    # img = extract(fname)
    # width, height = len(img), len(img[0])
    # lp = []
    # # for each column count the number of white pixels until 1st black pixel is encountered
    # # need to detect where the word begins and where it ends
    # begin, end = False, False
    # begin_i,end_i = 0,0
    # for i in range(width):
    #     for j in reversed(range(height)):
    #         if img[i][j] == BLACK and not begin:
    #             begin_i = i
    #             begin = True
    #             break
    # for i in reversed(range(width)):
    #     for j in reversed(range(height)):
    #         if img[i][j] == BLACK and not end:
    #             end_i = i
    #             end = True
    #             break
    # for i in range(begin_i,end_i):
    #     sum_white = 0
    #     j=height-1
    #     while j >= 0 and img[i][j] == WHITE:
    #         j-=1
    #         sum_white += 1
    #
    #     if sum_white == height: # no black pixel encountered => take same value as the last one introduced
    #         sum_white = lp[-1]
    #
    #     lp.append(sum_white)
    #
    # # normalize
    # norm_up = normalize(lp)
    # return norm_up

def normalize(x):
    ma = max(x)
    mi = min(x)
    if ma == mi:
        mi = 0
    for i in range(len(x)):
        x[i] = (x[i]-mi)/float(ma-mi)

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
#kws = ["A-r-t-v-s", "d-a-z", "G-r-a-l-s", "k-v-n-e-g-i-n-n-e"]
kws_path = "./WashingtonDB/keywords/"
#kws_path = "./ParzivalDB/keywords/"

# Words
# ws = ["274-05-02", "274-12-04", "273-33-05"]
ws_path = "WashingtonDB/words/"
#ws_path = "ParzivalDB/words/"

# Ground truth
gt_file = "WashingtonDB/WashingtonDB.txt"
#gt_file = "ParzivalDB/ParzivalDB.txt"
gt = {}


# extract ground-truth in dictionnary for quick search
with open(gt_file) as f:
    cgt = [x.strip('\n ') for x in f.readlines()]
f.close()
for line in cgt:
    key = line.split(' ', 1)[0]
    label = line.split(' ', 1)[1]
    gt[key] = label


#PART 1 Parse all lines to obtain vector feature for each column
features={}
dict_path = './features_ppratio.dict'
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
    nbr_hits = 100
    print str(nbr_hits)+" first hits for keyword "+kw+"."
    print "=========================="
    print " "
    match = []
    while len(match) != nbr_hits:
        elem = array.pop(0)
        if not elem[0] in match:
            print elem
            print gt[elem[0].split('.', 1)[0]]
            match.append(elem[0])
    precision, recall, fpr = [],[],[]
    for threshold in range(0,nbr_hits):
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
                    fn += 1
                else:
                    tn += 1
        try:
            precision_str = float(tp)/(float(tp)+float(fp))
            precision_str = float("{0:.2f}".format(precision_str))
        except:
            #should not happend
            precision_str = 0.0
        try:
            #also known as True Positive Rate
            recall_str = float(tp)/(float(tp)+float(fn))
            recall_str = float("{0:.2f}".format(recall_str))
        except:
            recall_str = 0.0
        try:
            #False Positive Rate
            fpr_str = float(fp)/(float(fp)+float(tn))
            fpr_str = float("{0:.2f}".format(fpr_str))
        except:
            fpr_str = 0.0
        print "T"+str(threshold)+" precision= "+str(precision_str)+" recall (TPR)= "+str(recall_str)+" FPR= "+str(fpr_str)
        precision.append(precision_str)
        recall.append(recall_str)
        fpr.append(fpr_str)

    plt.figure(1, figsize=(9, 4))
    plt.subplot(121)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0.0,1.0, 0.0,1.0])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.plot(recall, precision, 'r')

    eer_x,eer_y = 0.0,0.0
    min_diff = 1000.0
    for i in range(len(fpr)):
        if abs(1-fpr[i]-recall[i]) < min_diff:
            min_diff = abs(1-fpr[i]-recall[i])
            eer_x,eer_y = fpr[i],recall[i]
            # print "eer_x="+str(eer_x)+"   eer_y="+str(eer_y)

    avgp = 0.0
    total_match = 0#count when it's a non-match
    for i in range(len(precision)):
        try:
            d_recall = abs(recall[i] - recall[i-1])
        except:
            d_recall = recall[i]
        if d_recall != 0:
            avgp += precision[i]
            total_match += 1

    avgp = avgp/total_match

    print "EER= " +str(eer_x)+ "," +str(eer_y)
    print "AP= " +str(avgp)

    plt.subplot(122)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.axis([0.0,1.0, 0.0,1.0])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.plot(fpr, recall, 'r', eer_x, eer_y, 'ko')

    plt.savefig(kw + "_res_ppratio.png")
    plt.show()
    print " "


    # Output complete ranked list
    #res_file = "WashingtonDB_"+kw+".txt"
    res_file = "ParzivalDB_"+kw+".txt"
    if os.path.isfile(res_file) == False:
        file = open(res_file, "w+")
        for key in array:
            k = key[0].split('.',1)[0]
            file.write(k+"\n")
        file.close
