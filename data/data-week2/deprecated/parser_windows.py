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


#Gives a distance between to given image according to some features.
#fname1 = testInstance, fname2 = trainInstance
def distance(fname1,fname2):
    vector=[]
    #Feature 1 Project Profile per column
    pp1 = pp_col(fname1)
    pp2 = pp_col(fname2)
    feature = dtw(pp1, pp2)
    vector.append(feature)

    #Feature 2 Project Profile per line
    #pp1 = pp_line(fname1)
    #pp2 = pp_line(fname2)
    #feature = dtw(pp1, pp2)
    #vector.append(feature)

    #Feature 3 Project Profile transition B/W  per line
    #pp1 = pp_line_transition(fname1)
    #pp2 = pp_line_transition(fname2)
    #feature = dtw(pp1, pp2)
    #vector.append(feature)

    #Feature 4 Project Profile transition B/W per column
    pp1 = pp_col_transition(fname1)
    pp2 = pp_col_transition(fname2)
    feature = dtw(pp1, pp2)
    vector.append(feature)

    #Feature 5 Upper Profile
    up1 = up(fname1)
    up2 = up(fname2)
    feature = dtw(up1, up2)
    vector.append(feature)

    #Feature 6 Upper Profile
    lp1 = lp(fname1)
    lp2 = lp(fname2)
    feature = dtw(lp1, lp2)
    vector.append(feature)

    #append label of fname2
    #f = fname2.split(".")[0]
    #print fname2
    #l = gt[f] #label
    #vector.append(l)
    # img = pp(file) #feature vector
    return vector


#compare two custom distances via postal comparison
#OUTPUT : 0 = same (=), + = greater (>), - = less (<)
def compare(dist1,dist2):
    distance = 0
    #compare feature 1 : projection profile column

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




def loadSets(dirname, split):
    '''
    Seperate data into a training set and a test set
    '''
    trainingSet = []
    testSet = []

    for file in os.listdir(dirname):
        if file.endswith(".png"):
            f = file.split(".")[0]
            # l = gt[f] #label
            # img = pp(file) #feature vector
            if random.random() < split:
                # img.append(l)
                trainingSet.append(file)
            else:
                testSet.append(file)
    # print trainingSet
    # print "\n\n"
    # print testSet

    return trainingSet, testSet


def getNeighbors(trainingSet, testInstance, k=1):
    dist_matrix = []
    for i in range(len(trainingSet)):
        dist_vec = distance(testInstance, trainingSet[i]) #distance vector (w/ several features)
        dist_matrix.append([trainingSet[i][-1], dist_vec])

    dist_matrix = sorted(dist_matrix, key=methodcaller('compare'))
    # dist_matrix.sort(key=operator.itemgetter(1))
    # print dist_matrix
    neighbors = []
    for i in range(k):
        neighbors.append(dist_matrix[i][0])

    return neighbors


def getVotes(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def knn(trainingSet, testSet, k=1):
    predictions = []
    for i in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[i], k)
        result = getVotes(neighbors)
        predictions.append(result)
    return True









# ------------------------- M A I N -------------------------#

# Keywords
kws = ["O-c-t-o-b-e-r", "s-o-o-n", "t-h-a-t"]
kws_path = "./WashingtonDB/keywords/"
# Words
ws = ["274-05-02", "274-12-04", "273-33-05"]
ws_path = "WashingtonDB/lines/"
crop_path = "WashingtonDB/crops/"
# Ground truth
gt_file = "WashingtonDB/LinesWashington.txt"


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

windows = 30 # 30px width for the sliding windows
for kw in kws:
    dissimilarity = {}
    keyword = kws_path + kw + '.png'
    kwimg = Image.open(keyword)
    kw_size = kwimg.size
    kw_width = kw_size[0]
    array = []
    for path, subdirs, files in os.walk(ws_path):
        #checking files
        for file in files:
            im=Image.open(ws_path+file)
            size = im.size # (width,height) tuple
            width = size[0]
            for i in range(width):
                if i % windows == 0 and i+kw_width < width:
                    crop = crop_path+'win'+str(i)+'_'+str(kw)+'_'+file
                    im.crop((i, 0,i+kw_width, size[1])).save(crop)
                    word = ws_path + file
                    dist = distance(keyword,crop)
                    array.append([word,dist])
    print "keyword "+str(kw)+" done."
    #Sorting the array computed
    array.sort(compare)
    print "Ten first hits for keyword "+kw+"."
    print "=========================="
    print " "
    for i in range(10):
        print array[i]
    print " "
        #for w2 in ws:
        #    if w1!=w2:
        #        word2 = ws_path + w2 + '.png'
        #        dist2 = distance(keyword,word2)
        # dist = dtw(pp(keyword), pp(word))
        #        if not( (w1+':'+w2 in dissimilarity) or (w2+':'+w1 in dissimilarity)):
        #            dissimilarity[w1+':'+w2] = compare(dist1,dist2)

    # ~rank list
    #res = sorted(dissimilarity.items(), key=lambda x:x[1])
    # res = sorted(dissimilarity, key=dissimilarity.get)
    #tp, fn, fp, tn = 0,0,0,0 # false/true positive/negative
    #for i in res:
    #    if gt[i[0]] == kw:
    #        print "ok"
    #    else:
    #        print "not ok"
    #print res


# set1, set2 = loadSets("WashingtonDB/words", 0.75)

# trainSet = [[2, 2, 2, 4], [4, 4, 4, 3]]
# testInstance = [5, 5, 5, 1]
# k = 1
# neighbors = getNeighbors(trainSet, testInstance, 1)
# print(neighbors)
