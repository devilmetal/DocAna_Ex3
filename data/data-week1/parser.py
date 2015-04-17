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


def get_hist(fname):
    img = extract(fname)
    width, height = len(img), len(img[0])
    vertical = []

    for i in range(width):
        for j in range(height):
            if img[i][j] == BLACK:
                vertical.append(i)

    bins = width
    plt.hist(vertical, bins, [0,bins])
    plt.title('Histogram for ' + fname)
    plt.show()




# ------------------------- M A I N -------------------------#

# Keywords
imgs = ["./WashingtonDB/keywords/O-c-t-o-b-e-r"]

for img in imgs:
    fname = img+'.png'
    get_hist(fname)
