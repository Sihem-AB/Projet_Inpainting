# -*- coding: utf-8 -*-
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import operator
import cv2

from scipy.interpolate import *

image_path = "licorne.png"
i = misc.imread(image_path)
# plt.imshow(i)
# plt.axis('off')
# plt.show()

# Definir le trou initiale (delta omega 0) sense etre selectionne par le user
x1 = 50
x2 = 75
y1 = 80
y2 = 115
i[x1:x2, y1:y2, 0] = 0
i[x1:x2, y1:y2, 1] = 0
i[x1:x2, y1:y2, 2] = 0
i[x1:x2, y1:y2, 3] = 0


# print "contenu du pixel 120 120: channel 0 -> ",i[120, 120, 0]
# print "contenu du pixel 120 120: channel 1 -> ",i[120, 120, 1]
# print "contenu du pixel 120 120: channel 2 -> ",i[120, 120, 2]

#X, Y = np.ogrid[0:nl , 0:nc]

plt.imshow(i)
plt.axis('off')
plt.draw()


def region_filling_algorithm(image, pix1, pix2,patch_size, alpha):

    if patch_size%2 != 0:
        print ">> Debut..."
        nl, nc, cc = image.shape
        patch_area = patch_size*patch_size
        milieu = (patch_size - 1) / 2



        # initialisation du trou, les coordonnees des pixels composant le trou
        print ">> Initialisation du trou..."
        gamma_t = []
        for x in range(pix1[0], pix2[0]):
            for y in range(pix1[1], pix2[1]):
                gamma_t.append((x,y))


        # Initialisation des pixels contours, garder les coordonnees des pixels
        print ">> Initialisation des contours"
        delta_gamma_t = []
        for x in range(pix1[0], pix2[0]):
                delta_gamma_t.append((x,pix1[1]))
        for x in range(pix1[0], pix2[0]):
                delta_gamma_t.append((x,pix2[1]))
        for y in range(pix1[1]+1, pix2[1]-1):
                delta_gamma_t.append((pix1[0],y))
        for y in range(pix1[1]+1, pix2[1]-1):
                delta_gamma_t.append((pix2[0],y))


        # initialisation de Cp
        print ">> Initialisation de C(p)"
        C = np.ones((nl, nc))
        C[pix1[0]:pix1[1], pix2[0]:pix2[1]] = 0

        cpt = 0
        #while(len(delta_gamma_t)!=0):
        while(cpt < 20):
            print ">> Debut du while..., len(delta_gamma_t) = ", len(delta_gamma_t)
            cpt = cpt + 1
            # Compute prioreties
            print ">> Calcul des P(p)"
            P = {}

            # We start by creating a mask : on met 1 dans le contour du trou t 0 au reste
            mask = np.zeros((image.shape[0], image.shape[1]))
            for p in delta_gamma_t:
                mask[p] = 1

            for p in delta_gamma_t:
                # P(p) = C(p) * D(p)

                # C(p) la somme des C(q) pour tout q qui appartient au patch mais pas au trou
                somme = 0
                for x in range(p[0]-milieu, p[0]+milieu):
                    for y in range(p[1]-milieu, p[1]+milieu):
                        if (x,y) not in gamma_t:
                            somme = somme + C[x][y]

                C[p[0], p[1]] = somme/patch_area
                P[p] = C[p[0], p[1]]

                print P

                ## Calculer D(p) ...
                # On commence par calculer le gradient(en utilisant le filtre sobel)
                sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
                # On calcule de l'angle
                angle = cv2.phase(sobelx, sobely)

                # On fait une interpolation du contour pour pouvoir calculer la normale
                # delta_sorted = sorted(delta_gamma_t, key=lambda x: x[0])
                # interpolator = KroghInterpolator([x[0] for x in delta_sorted], [x[1] for x in delta_sorted])
                # derivate_interp = misc.derivative(interpolator, p[0])
                # # Calcul du vecteur unitaire normale
                # # tangente en un pt : y = f(a) +f'(a)(x-a)
                # yt = p[1] +derivate_interp
                # vector_tanget = (1,yt-p[1])  #p0+1-p0
                # n = (-vector_tanget[1], vector_tanget[0])
                # print n
                # P[p] += abs(sobelx[p[0], p[1],0] * n[0] +  sobely[p[0], p[1],0] * n[1])

                # On calcule maintenant le gradient du masque. Cela va nous permettre d'avoir le vecteur unitaire de la normal
                grad = np.gradient(mask)
                gradx, grady = grad[0], grad[1]
                print gradx.shape
                print gradx[p], " ", grady[p]



            # Chercher le patch ayant le plus grand Pp
            print P
            max_patch_center = max(P.iteritems(), key=operator.itemgetter(1))[0]
            print ">> Patch au plus grand P(p) calculé..."


            # Trouver l'exemplaire dans la region qui n'est pas le trou, qui minimise la distance entre les deux patchs
            max_patch_pixels = [] #les pixels dans le patch qui n'appartiennent pas au trou!
            for x in range(max_patch_center[0]-milieu, max_patch_center[0]+milieu):
                row = []
                for y in range(max_patch_center[1]-milieu, max_patch_center[1]+milieu):
                    if (x,y) not in gamma_t:
                        row.append(1)
                    else:
                        row.append(0)
                max_patch_pixels.append(row)

            exempl_patch = {}
            for x in range(milieu, nl-milieu):
                for y in range(milieu, nc-milieu):
                    if (x,y) not in gamma_t and x%patch_size==0 and y%patch_size==0:
                        moy = []
                        for i in range(0,milieu):
                            for j in range(0, milieu):
                                if max_patch_pixels[i][j] == 1:
                                    dist = 0
                                    for channel in range(3):
                                        dist = dist + (image[x-i, y-j, channel] - image[max_patch_center[0]-i, max_patch_center[1]-j, channel])^2
                                        dist = dist + (image[x + i, y + j, channel] - image[max_patch_center[0] + i , max_patch_center[1] + j, channel]) ^ 2
                                        dist = dist + (image[x + i, y - j, channel] - image[max_patch_center[0] + i, max_patch_center[1] - j, channel]) ^ 2
                                        dist = dist + (image[x - i, y + j, channel] - image[max_patch_center[0] - i, max_patch_center[1] + j, channel]) ^ 2
                                        moy.append(dist)
                        exempl_patch[(x,y)] = sum(moy)/len(moy)

            min_exempl_center = max(exempl_patch.iteritems(), key=operator.itemgetter(1))[0]

            print ">> exemple de distance minimale calculé..."

            # copier les pixels de l'exemplaire sur le patch du trou
            for i in range(0, milieu):
                for j in range(0, milieu):
                    image[max_patch_center[0] - i , max_patch_center[1] - j] = image[min_exempl_center[0] - i , min_exempl_center[1] - j]
                    image[max_patch_center[0] - i , max_patch_center[1] + j] = image[min_exempl_center[0] - i , min_exempl_center[1] + j]
                    image[max_patch_center[0] + i , max_patch_center[1] + j] = image[min_exempl_center[0] + i , min_exempl_center[1] + j]
                    image[max_patch_center[0] + i , max_patch_center[1] - j] = image[min_exempl_center[0] + i , min_exempl_center[1] - j]
            print ">> pixels copiés dans le trou..."
            # Update C(p) pour tous les pixels appartenant au patch actuel et le trou

            for p in max_patch_pixels:
                if p in gamma_t:
                    somme = 0
                    for x in range(p[0] - milieu, p[0] + milieu):
                        for y in range(p[1] - milieu, p[1] + milieu):
                            if (x, y) not in gamma_t:
                                somme = somme + C[x][y]
                    C[x][y] = somme / patch_area
            print ">> MAJ de C(p) effectuée..."
        # update gamma_t: enlever les pixels qui n'appartiennent plus au trou de gamma_t
            new_gamma_t = []
            for p in gamma_t:
                if p not in max_patch_pixels:
                    new_gamma_t.append(p)

            gamma_t = new_gamma_t

            print ">> MAJ du trou effectuée..."
        # update delta_gamma_t: identifier les nouveaux contours
            delta_gamma_t = []
            dict_x = {}
            dict_y = {}
            for p in gamma_t:
                if p[0] in dict_x:
                    dict_x[p[0]].append(p[1])
                else:
                    dict_x[p[0]] = [p[1]]

                if p[1] in dict_y:
                    dict_y[p[1]].append(p[0])
                else:
                    dict_y[p[1]] = [p[0]]

            for x in dict_x:
                max_y = max(dict_x[x])
                min_y = min(dict_x[x])
                delta_gamma_t.append((x,max_y))
                delta_gamma_t.append((x,min_y))

            for y in dict_y:
                max_x = max(dict_y[y])
                min_x = min(dict_y[y])
                delta_gamma_t.append((y,max_x))
                delta_gamma_t.append((y,min_x))

            print ">> MAJ des nouveaux contours du trou effectuée..."
            print delta_gamma_t
            print "\n"
    else:
        raise Exception('Patch_size doit etre impair.')

    return image

print "Début"
new_image = region_filling_algorithm(image = i, pix1 = [x1, y1], pix2 = [x2, y2], patch_size= 5, alpha = 255)
print "FIN!!!"
plt.imshow(new_image)
plt.axis('off')
plt.show()
