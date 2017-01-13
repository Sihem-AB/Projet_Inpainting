# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy import misc
from scipy.signal import convolve2d
import operator
import matplotlib.image as mpimg
from matplotlib import colors
from skimage import io, color
from scipy.interpolate import *
import numba
from Inpainting import *


cadrant = []

def onselect(eclick, erelease):
    #print "eclick: ", eclick.xdata, " ___ ", eclick.ydata  #x1, y1
    #print "erelease: ", erelease.xdata, " ___ ", erelease.ydata #x2, y2

    cadrant.append(int(eclick.xdata))
    cadrant.append(int(eclick.ydata))
    cadrant.append(int(erelease.xdata))
    cadrant.append(int(erelease.ydata))

    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    # ax.set_ylim(erelease.ydata,eclick.ydata)
    # ax.set_xlim(eclick.xdata,erelease.xdata)
    fig.canvas.draw()



fig = plt.figure()
ax = fig.add_subplot(111)

# filename="plage.jpg"
filename = "licorne.png"
# filename = "ile.jpg"
i = mpimg.imread(filename)

arr = np.asarray(i)
plt_image=plt.imshow(arr)
plt.title("Select the part you want to remove, and close the windows when you have finish")

rs=widgets.RectangleSelector(
    ax, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))

plt.show()

# plt.imshow(i)
# plt.axis('off')
# plt.show()

# Definir le trou initiale (delta omega 0) sense etre selectionne par le user
x1 = cadrant[1]
y1 = cadrant[0]
x2 = cadrant[3]
y2 = cadrant[2]

print x1, " ", y1, " ", x2, " ", y2

for channel in range(i.shape[2]):
    i[x1:x2, y1:y2, channel] = 0.5
    i[x1:x2, y1:y2, channel] = 0.5

plt.imshow(i)
plt.axis('off')
plt.draw()

# im = color.rgb2grey(i[:,:,:3])[:,:].reshape((i.shape[0], i.shape[1], 1))

im = i

inpainting = Inpainting(image = im, pix1 = [x1, y1], pix2 = [x2, y2], patch_size=9, alpha = np.max(i))
new_image = inpainting.region_filling_algorithm()

print "FIN!!!"
plt.imshow(new_image)
plt.axis('off')
plt.show()
plt.pause(1000)
