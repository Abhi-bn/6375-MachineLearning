import cv2
import numpy as np
from numpy.core.getlimits import inf
from random import randint


def kmeans(img, k_value):
    kmeans = []
    for i in range(k_value):
        kmeans.append([randint(0, 255), randint(0, 255), randint(0, 255)])
    old_cluster = np.zeros((img.shape[0], img.shape[1]))
    itr = 0
    while True:
        mins = [0] * k_value
        for i in range(len(mins)):
            sum = 0
            for j in range(3):
                sum += (kmeans[i][j] - img[:, :, j]) ** 2
            mins[i] = np.sqrt(sum)

        new_cluster = np.zeros(mins[0].shape)

        for i in range(mins[0].shape[0]):
            for j in range(mins[0].shape[1]):
                pos = 0
                min = inf
                for k, m in enumerate(mins):
                    if m[i, j] < min:
                        min = m[i, j]
                        pos = k
                new_cluster[i, j] = pos

        for i in range(len(mins)):
            val = []
            for j in range(3):
                me = np.nanmean(img[np.where(new_cluster == i)][:, j])
                if np.isnan(me):
                    me = kmeans[i][j]
                val.append(np.round(me))
            kmeans[i] = val
        if (old_cluster == new_cluster).all() or itr == 1000:
            break
        old_cluster = new_cluster
        itr += 1
    return new_cluster, kmeans


def store_disp(name, image, cluster, kmeans, k_value):
    display_image = np.zeros(image.shape)
    for i in range(k_value):
        display_image[np.where(cluster == i)] = kmeans[i]
    cv2.imwrite("output/"+name+str(k_value)+".jpeg", display_image)


imgs = [cv2.imread('homework3/images/Penguins.jpg'), cv2.imread('homework3/images/Koala.jpg')]
name = ['Koala', 'Penguins']
comp = [3]

for c in comp:
    for k in range(4):
        for i, img in enumerate(imgs):
            new_c, centers = kmeans(img, c)
            store_disp(name[i] + "run" + str(k), img, new_c, centers, c)
            print(centers)
