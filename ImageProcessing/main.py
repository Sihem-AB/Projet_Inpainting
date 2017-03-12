# -*- coding: utf-8 -*-
import matplotlib.widgets as widgets
import matplotlib.image as mpimg
from Inpainting import *

# -------------------------------PARAMETRE UTILISEES  ----------------------------------------
patch_size = 9
gaussian_blur = 0.1

rep = "Images/"
# filename = "Kanizsa_triangle.jpg"
# filename="plage.jpg"
# filename = "licorne.png"
# filename = "ile.jpg"
# filename = "paysage_homme.jpg"
# filename = "chaise.png"
# filename = "orange.jpg"
# filename = "moto.jpg"
# filename = "livre.jpg"
# filename = "renovation.png"
# filename = "texture.gif"
# filename = "statueLiberte.jpg"
filename = "parachutiste.jpg"
# filename = "triangles.jpg"
# filename = "triangles2.jpg"
i = mpimg.imread(rep + filename)

# ------------------------------FIN PARAMETRES -------------------------------------------------------------------------

#  --------------------------GESTION DE LA SELECTION pour le troue -----------------------------------------------------
cadrant = []

def onselect(eclick, erelease):

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

arr = np.asarray(i)
plt_image=plt.imshow(arr)
plt.title("Select the part you want to remove, \n and close the windows when you have finished")

rs=widgets.RectangleSelector(
    ax, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
plt.show()


# Definir le trou initiale (delta omega 0) après que l'utilisateur ait selectionné le rectange
x1 = cadrant[-3]
y1 = cadrant[-4]
x2 = cadrant[-1]
y2 = cadrant[-2]

milieu = patch_size / 2
x1, x2 = max(x1 - milieu, milieu), min(x2+milieu+1, i.shape[0] - milieu-1)
y1, y2 = max(y1 - milieu, milieu), min(y2+milieu+1, i.shape[1] - milieu-1)

# On met le troue dans l'image
for channel in range(i.shape[2]):
    i[x1:x2+1, y1:y2+1, channel] = 1


# ----------------------------FIN DE LA GESTION DE LA SELECTION-------------------------------------------------------


plt.imshow(i)
plt.axis('off')
plt.draw()

im = color.rgb2lab(i[:,:,:3])
inpainting = Inpainting(image = im,
                        pix1 = [x1, y1],
                        pix2 = [x2, y2],
                        patch_size=patch_size,
                        alpha = np.max(im),
                        gaussian_blur=gaussian_blur)
new_image = inpainting.region_filling_algorithm()

print "FIN!!!"

plt.figure()
plt.imshow(i[:,:,:3])
plt.axis('off')
plt.title("Image avant algorithme")
plt.figure()
plt.imshow(color.lab2rgb(new_image))
plt.title("Image apres algorithme : " + str(patch_size))
plt.axis('off')
plt.show()
plt.pause(100000000)
raw_input("Press Enter to continue...")