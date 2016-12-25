# -*- coding: utf-8 -*-
from scipy import misc
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np
import operator
import matplotlib.image as mpimg
from matplotlib import colors
from skimage import io, color
from scipy.interpolate import *

import numba

# image_path = "lena.gif"
# image_path = "Wikipedia.png"
image_path = "licorne.png"
# image_path = "Kaniza_triangle.jpg"
i = mpimg.imread(image_path)
# plt.imshow(i)
# plt.axis('off')
# plt.show()

# Definir le trou initiale (delta omega 0) sense etre selectionne par le user
x1 = 50
x2 = 75
y1 = 80
y2 = 115

for channel in range(i.shape[2]):
    i[x1:x2, y1:y2, channel] = 0.5

# print "contenu du pixel 120 120: channel 0 -> ",i[120, 120, 0]
# print "contenu du pixel 120 120: channel 1 -> ",i[120, 120, 1]
# print "contenu du pixel 120 120: channel 2 -> ",i[120, 120, 2]

#X, Y = np.ogrid[0:nl , 0:nc]

plt.imshow(i)
plt.axis('off')
plt.draw()

def retourne_contour(mask):
    # On va convoluer avec un masque laplacien pour detecter le contour
    masque_laplacien = np.array([[1,1,1], [1,-8,1], [1, 1, 1]])
    after_laplac = convolve2d(mask, masque_laplacien, 'same')
    return [tuple(x) for x in list(np.transpose(after_laplac.nonzero()))]

@numba.jit
def find_best_patch(image, max_patch_center, milieu, mask,nl,nc,cc):
    xp,yp = max_patch_center
    exempl_patch = {}
    for x in range(milieu, nl - milieu):
        for y in range(milieu, nc - milieu):
            if mask[(x, y)] == 0:  # n'appartient pas au troue
                mask_and = ~mask[x-milieu:x+milieu+1,y-milieu:y+milieu+1].astype(bool) & ~mask[xp-milieu:xp+milieu+1,yp-milieu:yp+milieu+1].astype(bool)
                if mask_and.any():
                    diff = image[x-milieu:x+milieu+1,y-milieu:y+milieu+1, :] - image[xp-milieu:xp+milieu+1,yp-milieu:yp+milieu+1, :]
                    exempl_patch[(x,y)] = np.sum(np.power(diff[mask_and],2))

                    # Trouver l'exemplaire dans la region qui n'est pas le trou, qui minimise la distance entre les deux patchs
    print "Recherche du minimale ..."
    print min(exempl_patch.iteritems(), key=operator.itemgetter(1))
    return min(exempl_patch.iteritems(), key=operator.itemgetter(1))[0]




def region_filling_algorithm(image, pix1, pix2,patch_size, alpha):

    if patch_size%2 == 0:
        print ">> Debut..."
        nl, nc = image.shape[0], image.shape[1]

        cc = np.clip(image.shape[2], 1, 3)

        # cc = 1
        # if len(image.shape) == 3:
        #     cc = 3

        patch_area = patch_size*patch_size
        milieu = (patch_size) / 2



        # initialisation du trou, les coordonnees des pixels composant le trou
        print ">> Initialisation du trou..."
        gamma_t = []
        for x in range(pix1[0], pix2[0]):
            for y in range(pix1[1], pix2[1]):
                gamma_t.append((x,y))


        # Initialisation des pixels contours, garder les coordonnees des pixels
        print ">> Initialisation des contours"
        delta_gamma_t = []
        for x in range(pix1[0], pix2[0]+1):
                delta_gamma_t.append((x,pix1[1]))
        for x in range(pix1[0], pix2[0]+1):
                delta_gamma_t.append((x,pix2[1]))
        for y in range(pix1[1], pix2[1]+1):
                delta_gamma_t.append((pix1[0],y))
        for y in range(pix1[1], pix2[1]+1):
                delta_gamma_t.append((pix2[0],y))


        # initialisation de Cp
        print ">> Initialisation de C(p)"
        C = np.ones((nl, nc))
        C[pix1[0]:pix2[0]+1, pix1[1]:pix2[1]+1] = 0

        D = np.zeros((nl,nc))

        # We start by creating a mask : on met 1 dans le contour du trou t 0 au reste
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)
        mask[pix1[0]:pix2[0], pix1[1]:pix2[1]] = 1

        cpt = 0
        #while(len(delta_gamma_t)!=0):
        plt.ion()
        while(len(delta_gamma_t) > 0):
            print ">> Debut du while..., len(delta_gamma_t) = ", len(delta_gamma_t)
            cpt = cpt + 1
            # Compute prioreties
            print ">> Calcul des P(p)"
            P = {}



            # On calcule maintenant le gradient du masque. Cela va nous permettre d'avoir le vecteur unitaire de la normal
            grad_mask = np.gradient(mask)
            grad_maskx, grad_masky = grad_mask[0], grad_mask[1]

            # normalisation du gradient
            grad_maskx, grad_masky = grad_maskx / np.max(grad_maskx), grad_masky / np.max(grad_masky)

            # On calcule le gradientde l'image (cela va servir pour le calcul de D(p))
            gradx = np.zeros((image.shape[:2]))
            grady = gradx
            for channel in range(cc):
                grad_isophote = np.gradient(image[:, :, channel])
                gradx += grad_isophote[0]
                grady += grad_isophote[1]
            gradx /= 3
            grady /= 3

            # gradx /= np.max(gradx)
            # grady /= np.max(grady)

            # On fait une rotation de 90 degré du gradient pour que le gradient suivant le contour(sinon il est perpendiculaire au contour)
            tmp = gradx
            gradx = -grady
            grady = tmp

            plt.figure(1)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title("mask")
            #
            # plt.figure()
            # plt.imshow(grad_maskx, cmap='gray')
            # plt.axis('off')
            # plt.title("grad_maskx")
            #
            # plt.figure()
            # plt.imshow(grad_masky, cmap='gray')
            # plt.axis('off')
            # plt.title("grad_masky")

            print "shape : ", image.shape

            plt.figure(0)
            plt.imshow(image)
            plt.axis("off")
            plt.title("image")
            plt.show()

            plt.figure(2)
            plt.imshow(C)
            plt.axis("off")
            plt.title("C")
            plt.show()

            plt.figure(3)
            plt.imshow(D+np.average(image, axis=2))
            plt.axis("off")
            plt.title("D")
            plt.show()

            # plt.figure(4)
            # plt.imshow(gradx, cmap='gray')
            # plt.axis("off")
            # plt.title("gradx")
            # plt.show()
            #
            # plt.figure(5)
            # plt.imshow(grady, cmap='gray')
            # plt.axis("off")
            # plt.title("grady")
            # plt.show()


            plt.pause(0.001)

            for p in delta_gamma_t:
                # P(p) = C(p) * D(p)

                # C(p) la somme des C(q) pour tout q qui appartient au patch mais pas au trou
                # somme = 0
                # min_x, max_x = max(p[0]-milieu, 0), min(p[0]+milieu, nl)
                # min_y, max_y = max(p[1] - milieu, 0), min(p[1] + milieu, nc)
                # for x in range(min_x, max_x):
                #     for y in range(min_y, max_y):
                #         if mask[(x,y)] == 0: # dans le troue
                #             somme = somme + C[x][y]
                #
                # C[p] = somme/((max_x-min_x) * (max_y-min_y))
                P[p] = C[p]

                ## Calculer D(p) ...

                # print grad_maskx[p], " ", grad_masky[p]
                N = np.array([grad_maskx[p], grad_masky[p]])
                isophote = np.array([gradx[p], grady[p]])
                D[p] = abs(N[0]*isophote[0] + N[1]*isophote[1])/alpha
                P[p] *= abs(N[0]*isophote[0] + N[1]*isophote[1])/alpha

                print "C[p] : ", C[p], " D[p] : ", D[p], "N : ", N



            # Chercher le patch ayant le plus grand Pp
            print P
            max_patch_center = max(P.iteritems(), key=operator.itemgetter(1))[0]
            print ">> Patch au plus grand P(p) calculé..."

            min_exempl_center = find_best_patch(image, max_patch_center, milieu, mask, nl, nc, cc)
            print ">> exemple de distance minimale calculé..."

            # On met à jour le masque
            xp,yp = max_patch_center
            print "xp, yp : ", max_patch_center
            xq,yq = min_exempl_center
            print "xq, yq :", min_exempl_center, " mask : ", mask[min_exempl_center]

            plt.figure(6)
            plt.imshow(image[xp - milieu:xp + milieu, yp - milieu:yp + milieu])
            plt.axis("off")
            plt.title("meilleur patch a remplir")
            plt.show()

            plt.figure(7)
            plt.imshow(image[xq - milieu:xq + milieu, yq - milieu:yq + milieu])
            plt.axis("off")
            plt.title("meilleur aptche trouvee")
            plt.show()

            plt.pause(0.001)

            # On met à jour le masque
            for i in range(-milieu, milieu+1):
                for j in range(-milieu, milieu+1):
                    if mask[(xp+i, yp+j)] == 1:
                        mask[(xp + i, yp + j)] = mask[(xq+i, yq+j)].copy()
                        image[xp + i, yp + j,:] = image[xq+i, yq+j,:].copy()

            # mask[xp-milieu:xp+milieu, yp-milieu:yp+milieu] = mask[xq-milieu:xq+milieu, yq-milieu:yq+milieu].copy()

            print ">> pixels copiés dans le trou..."
            # Update C(p) pour tous les pixels appartenant au patch actuel et le trou

            delta_gamma_t = retourne_contour(mask)
            delta_gamma_t = [x for x in delta_gamma_t if mask[x]==1]


            for x in range(xp-milieu, xp+milieu+1):
                for y in range(yp-milieu, yp+milieu+1):
                    if mask[(x,y)] == 1:
                        C[x,y] = 0
                        for i in range(x - milieu, x + milieu+1):
                            for j in range(y - milieu, y + milieu+1):
                                if mask[i,j] == 0:
                                    C[x,y] += C[i,j]
                        C[x,y] /= patch_area



            print ">> MAJ de C(p) effectuée..."

            print ">> MAJ du trou effectuée..."

            print ">> MAJ des nouveaux contours du trou effectuée..."
            print delta_gamma_t
            print "\n"
    else:
        raise Exception('Patch_size doit etre impair.')

    return image

# im = color.rgb2grey(i[:,:,:3])[:,:].reshape((i.shape[0], i.shape[1], 1))
im = i

new_image = region_filling_algorithm(image = im, pix1 = [x1, y1], pix2 = [x2, y2], patch_size=14, alpha = np.max(i))
print "FIN!!!"
plt.imshow(new_image)
plt.axis('off')
plt.show()
plt.pause(1000)