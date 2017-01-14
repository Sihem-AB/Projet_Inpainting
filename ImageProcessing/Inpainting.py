import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from scipy import misc
from scipy.signal import convolve2d
import scipy.ndimage as ndimage
import operator
import matplotlib.image as mpimg
from matplotlib import colors
from skimage import io, color
from scipy.interpolate import *
import numba
import random
from numpy.lib.stride_tricks import as_strided
import copy

import pyximport
pyximport.install()

# from fastSSD import *

class Inpainting():
    def __init__(self, image, pix1, pix2,patch_size, alpha):
        self.image = image
        self.pix1 = pix1
        self.pix2 = pix2
        self.patch_size = patch_size
        self.alpha = alpha

    """
        Input :
            coutour (list of coordinate)
            mask (1 inside the hole, 0 otherwise)
        Output:
            normal compared to the gradient in each point

    """
    def normal_contour(self, contour, mask):
        # We put line of the contour in black, otherwise it's white
        # mask_countour = np.ones(mask.shape)
        # for p in contour:
        #     mask_countour[p] = 0
        #
        # mask_countour = ndimage.gaussian_filter(mask_countour, sigma=1, order=0)
        #
        grad_mask = np.gradient(mask)
        grad_maskx, grad_masky = grad_mask[0], grad_mask[1]

        # normalisation du gradient
        grad_maskx, grad_masky = grad_maskx / np.max(grad_maskx), grad_masky / np.max(grad_masky)

        return grad_maskx, grad_masky

    """
        Input :
            mask (1 inside the hole, 0 otherwise)
        Output :
            The contour of the hole (list of coordinate)
    """
    def retourne_contour(self, mask):
        # On va convoluer avec un masque laplacien pour detecter le contour
        masque_laplacien = np.array([[1,1,1], [1,-8,1], [1, 1, 1]])
        after_laplac = convolve2d(mask, masque_laplacien, 'same')
        contour = [tuple(x) for x in list(np.transpose(after_laplac.nonzero()))]

        contour = [x for x in contour if mask[x] == 1]



        if not contour: #If the laplacien doesnt detect any contour, it can always remain some piwels who have not been filled yet
            contour = [tuple(x) for x in list(np.transpose(mask.nonzero()))]

        return contour


    # @numba.jit
    def find_best_patch(self, image, max_patch_center, milieu, mask,nl,nc,cc):
        xp,yp = max_patch_center

        patch = image[xp - milieu:xp + milieu + 1, yp - milieu:yp + milieu + 1,:].astype(float)

        # y = as_strided(image,
        #                shape=(image.shape[0] - milieu + 1,
        #                       image.shape[1] - milieu + 1,cc) +
        #                      (milieu, milieu),
        #                strides=image.strides * 2)
        #
        # mask_and = ~mask.astype(bool) & ~mask.astype(bool)
        #

        # Compute the sum of squared differences using broadcasting.
        # ssd = ((y - patch) ** 2 * mask_and).sum(axis=-1).sum(axis=-1)

        exempl_patch = {}
        voisinage_x = image.shape[0] / 2
        voisinage_y = image.shape[1] /2

        print "voisinage_x : ", voisinage_x

        xmin,xmax = max(xp - voisinage_x, milieu), min(xp+voisinage_x, nl - milieu)
        ymin, ymax = max(yp - voisinage_y, milieu), min(yp + voisinage_y, nc - milieu)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                if mask[(x, y)] == 0:  # n'appartient pas au troue
                    mask_and = ~mask[x-milieu:x+milieu+1,y-milieu:y+milieu+1].astype(bool) & ~mask[xp-milieu:xp+milieu+1,yp-milieu:yp+milieu+1].astype(bool)
                    if mask_and.any():  # Si les deux patchs ont au moins un pixel en commun qui n'appartient pas au toroue
                        somme = 0.0
                        for c in range(cc):
                            diff = image[x - milieu:x + milieu + 1, y - milieu:y + milieu + 1, c].astype(float) \
                                   - patch[:,:,c]
                            somme += np.sum(diff[mask_and] * diff[mask_and])
                        exempl_patch[(x,y)] = somme / float(np.sum(mask_and))

                        # Trouver l'exemplaire dans la region qui n'est pas le trou, qui minimise la distance entre les deux patchs
        print "Recherche du minimale ..."
        # exemplar_sorted = sorted(exempl_patch.iteritems(), key=operator.itemgetter(1))
        # cpt = 0
        # for i in range(1,len(exemplar_sorted)):
        #     if exemplar_sorted[i][1] != exemplar_sorted[i-1][1]:
        #         break
        #     cpt += 1
        #
        # best_exemplar = random.randint(0, cpt)
        # # print min(exempl_patch.iteritems(), key=operator.itemgetter(1))
        # return exemplar_sorted[best_exemplar][0]
        return min(exempl_patch.iteritems(), key=operator.itemgetter(1))[0]


    """
    Main function to execute the inpainting algorithm
    """
    def region_filling_algorithm(self):
        image = self.image
        pix1 = self.pix1
        pix2 = self.pix2
        patch_size = self.patch_size
        alpha = self.alpha

        # The patch size must be an odd numbers
        if patch_size%1 == 0:
            print ">> Debut..."
            nl, nc = image.shape[0], image.shape[1]

            # We take into account than the image can be gray
            cc = 1
            if len(image.shape) == 3:
                cc = image.shape[2]
            patch_area = patch_size*patch_size
            milieu = (patch_size) / 2 # Will simplify the writing

            # Initialisation des pixels contours du troue, garder les coordonnees des pixels
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

            # Initialisation de Cp
            # We put 1 outside the hole, else : 0
            print ">> Initialisation de C(p)"
            C = np.ones((nl, nc))
            C[pix1[0]:pix2[0]+1, pix1[1]:pix2[1]+1] = 0

            D = np.zeros((nl,nc))

            # We start by creating a mask : on met 1 dans le trou t 0 au reste
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float)
            mask[pix1[0]:pix2[0], pix1[1]:pix2[1]] = 1

            cpt = 0
            plt.ion()
            while(len(delta_gamma_t) > 0):
                print ">> Debut du while..., len(delta_gamma_t) = ", len(delta_gamma_t)
                cpt = cpt + 1
                # Compute prioreties
                print ">> Calcul des P(p)"
                P = {}


                # On calcule maintenant le gradient du masque. Cela va nous permettre d'avoir le vecteur unitaire de la normal
                # grad_mask = np.gradient(mask)
                # grad_maskx, grad_masky = grad_mask[0], grad_mask[1]

                # normalisation du gradient
                # grad_maskx, grad_masky = grad_maskx / np.max(grad_maskx), grad_masky / np.max(grad_masky)

                grad_maskx, grad_masky = self.normal_contour(delta_gamma_t, mask)

                # On calcule le gradient de l'image (cela va servir pour le calcul de D(p))
                image_smooth = ndimage.gaussian_filter(image, sigma=0.1, order=0)
                gradx = np.zeros((image.shape[:2]))
                grady = gradx
                for channel in range(cc):
                    grad_isophote = np.gradient(image_smooth[:, :, channel])
                    gradx += grad_isophote[0]
                    grady += grad_isophote[1]
                gradx /= cc
                grady /= cc

                gradx /= np.max(gradx)
                grady /= np.max(grady)

                # We make a rotation of 90 degree of the gradient because othewise the gradient is perpendicular to the contour
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
                plt.imshow(color.lab2rgb(image))
                plt.axis("off")
                plt.title("image")
                plt.show()

                plt.figure(2)
                plt.imshow(C)
                plt.axis("off")
                plt.title("C")
                plt.show()

                plt.figure(3)
                plt.imshow(D)
                plt.axis("off")
                plt.title("D")
                plt.show()

                plt.figure(4)
                plt.imshow(gradx, cmap='gray')
                plt.axis("off")
                plt.title("gradx")
                plt.show()

                plt.figure(5)
                plt.imshow(grady, cmap='gray')
                plt.axis("off")
                plt.title("grady")
                plt.show()


                plt.pause(0.0000001)

                for p in delta_gamma_t:
                    # calcul de C[p]
                    Cp = 0
                    for x in range(p[0] - milieu, p[0] + milieu + 1):
                        for y in range(p[1] - milieu, p[1] + milieu + 1):
                            if mask[(x, y)] == 0:
                                Cp += C[(x,y)]
                    Cp /= patch_area


                    P[p] = Cp

                    ## Calculer D(p) ...
                    N = np.array([grad_maskx[p], grad_masky[p]])
                    # On choisit la valeur du plus grand gradient dans le patch
                    mask_patch = mask[p[0]-milieu:p[0]+milieu+1, p[1]-milieu:p[1]+milieu+1].astype(bool)
                    gradx_p = np.max(gradx[p[0]-milieu:p[0]+milieu+1, p[1]-milieu:p[1]+milieu+1] * ~mask_patch)
                    grady_p = np.max(grady[p[0]-milieu:p[0]+milieu+1, p[1]-milieu:p[1]+milieu+1] * ~mask_patch)
                    isophote = np.array([gradx_p, grady_p])
                    D[p] = abs(N[0]*isophote[0] + N[1]*isophote[1])/alpha +0.001
                    P[p] *= D[p]

                    print "C[p] : ", C[p], " D[p] : ", D[p], "N : ", N



                # Chercher le patch ayant le plus grand Pp
                print P
                max_patch_center = max(P.iteritems(), key=operator.itemgetter(1))[0]
                print ">> Best P[p] computed"

                min_exempl_center = self.find_best_patch(image, max_patch_center, milieu, mask, nl, nc, cc)
                print ">> Minimum distance computed"


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
                plt.title("meilleur patch trouvee")
                plt.show()

                plt.pause(0.001)

                # We update the mask
                for i in range(-milieu, milieu+1):
                    for j in range(-milieu, milieu+1):
                        if mask[(xp+i, yp+j)] == 1:
                            mask[(xp + i, yp + j)] = copy.deepcopy(mask[(xq+i, yq+j)])
                            image[xp + i, yp + j,:] = copy.deepcopy(image[xq+i, yq+j,:])

                print ">> Pixels copied inside the hole"

                # We compute the new contour of the hole
                delta_gamma_t = self.retourne_contour(mask)

                # Update C(p) for all the pixels inside the current patch and inside the hole
                for x in range(xp-milieu, xp+milieu+1):
                    for y in range(yp-milieu, yp+milieu+1):
                        if mask[(x,y)] == 1:
                            C[x,y] = 0
                            for i in range(x - milieu, x + milieu+1):
                                for j in range(y - milieu, y + milieu+1):
                                    if mask[i,j] == 0:
                                        C[x,y] += C[i,j]
                            C[x,y] /= patch_area

        else:
            raise Exception('Patch_size doit etre impair.')

        return image