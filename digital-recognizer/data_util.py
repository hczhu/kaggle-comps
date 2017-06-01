import sys

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt, matplotlib.image as mpimg

# %matplotlib inline

def load_data(name, max_rows=None):
    img_dim = 28
    labels_and_imgs = genfromtxt(name, delimiter=',', skip_header=1, max_rows=max_rows)
    print('There are {} instances with {} pixels.'.format(len(labels_and_imgs), len(labels_and_imgs[0])))
    labels = None
    if len(labels_and_imgs[0,:]) == img_dim * img_dim + 1:
        labels = labels_and_imgs[:,0]
        labels_and_imgs = labels_and_imgs[:,1:]
    return labels_and_imgs.reshape(len(labels_and_imgs), img_dim, img_dim), labels

def show_imgs(name, max_rows=None):
    imgs, labels = load_data(name, max_rows)
    print('There {} images.'.format(len(imgs)))
    for i in range(len(imgs)):
        img = imgs[i]
        plt.imshow(img, cmap='gray')
        if labels is not None:
            plt.title(int(labels[i]))
        plt.figure(i)
    plt.show()

if __name__ == "__main__":
    show_imgs(sys.argv[1], int(sys.argv[2]))
