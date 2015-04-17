from PIL import Image
import math
import numpy as np
import random
import collections
from matplotlib import pyplot as plt
import cv2
from PIL import ImageDraw
from scipy.stats import gaussian_kde

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


# count the number of black pixel for each column
def get_vhist(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    hist = []

    for i in range(width):
        for j in range(height):
            if img[i][j] == BLACK:
                hist.append(i)

    # bins = width
    # plt.hist(hist, bins, [0,bins])
    # plt.title('Vertical Histogram for ' + fname)
    # plt.show()

    return hist


# count the number of black pixel for each line
def get_hhist(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    hist = []

    for j in range(height):
        for i in range(width):
            if img[i][j] == BLACK:
                hist.append(j)

    # bins = height
    # plt.hist(hist, bins, [0,bins])
    # plt.title('Horizontal Histogram for ' + fname)
    # plt.show()

    return hist




# ------------------------- M A I N -------------------------#

# Keywords
imgs = ["./WashingtonDB/keywords/O-c-t-o-b-e-r"]

for img in imgs:
    fname = img+'.png'
    get_hhist(fname)
