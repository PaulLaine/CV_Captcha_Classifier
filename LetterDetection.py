import cv2 as cv
import diplib as dip
import numpy as np
from matplotlib import pyplot as plt
from captcha.image import ImageCaptcha
import Functions as fct
import os
from PIL import Image

main_path = 'C:/Users/plain/PycharmProjects/CaptchaProject/'

def letterDetection(dataset, folder):

    
    try:
        os.mkdir(os.path.join(os.getcwd(), folder))
    except:
        pass
    os.chdir(os.path.join(os.getcwd(), folder))

    numberLetters = 5

    image_count = np.zeros(36)
    list_char = fct.createList()

    for png_path in range(len(dataset)):
        img = cv.imread(dataset[png_path], 0)
        #cv.imshow("img", img)

        code = os.path.split(dataset[png_path])
        code = code[1][:5]

        ret, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,8))
        #cv.imshow("thresh", thresh)

        morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, np.ones((3, 4), np.uint8))
        #cv.imshow("morph", morph)
        dil = cv.dilate(morph, np.ones((2,2), np.uint8), iterations = 1)
        #cv.imshow("dil", dil)
        gauss = cv.GaussianBlur(dil, (1,1), 0)
        #cv.imshow("gauss", gauss)


        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours, hierarchy = cv.findContours(gauss, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        #cv.drawContours(img, contours, -1, (0,255,0), 3)
        #cv.imshow('contours', img)
        #print("nb contours :" +  str(len(contours)))


        #print("nb contours trouvés : " + str(len(contours)))
        cpt = 0
        listContours = []

        #for i in range(len(contours)):
        for i in range(1, len(contours)):
            #print(i)
            #print("nb CONTOURS : " + str(len(contours[i])))
            #print(contours[i])
            mini_y = min(contours[i][:, 0, 1])
            maxi_y = max(contours[i][:, 0, 1])
            mini_x = min(contours[i][:, 0, 0])
            maxi_x = max(contours[i][:, 0, 0])
            if( (len(contours[i]) > 4) and ( maxi_y-mini_y > 11 ) and ( maxi_x-mini_x > 11 ) ):
                listContours.append((i))
                cpt += 1
                cv.drawContours(img, contours[i], -1, (0, 255, 0), 0)
                #cv.imshow('contours', img)


        #print("cpt : " + str(cpt))
        #cv.imshow('final contours', img)

        listImages = []
        listAbsImages = []
        cpt = 0
        for i in range(len(listContours)):
            if (hierarchy[0][listContours[i]][3] == 0):
                x, y, width, height = cv.boundingRect(contours[listContours[i]])
                #print("largeur : " + str(width))
                if (width < 35):
                    roi = gauss[y:y + height, x:x + width]
                    listAbsImages.append(x)
                    listImages.append(roi)
                    cpt += 1
                elif (width < 65):
                    roi1 = gauss[y:y + height, x:x + int(width / 2)]
                    roi2 = gauss[y:y + height, x + int(width / 2):x + width]
                    listAbsImages.append(x)
                    listAbsImages.append(x + width / 2)
                    listImages.append(roi1)
                    listImages.append(roi2)
                    cpt += 2
                elif (width < 95):
                    roi1 = gauss[y:y + height, x:x + int(width / 3)]
                    roi2 = gauss[y:y + height, x + int(width / 3):x + int(2 * width / 3)]
                    roi3 = gauss[y:y + height, x + int(2 * width / 3):x + width]
                    listAbsImages.append(x)
                    listAbsImages.append(x + int(width / 3))
                    listAbsImages.append(x + int(2 * width / 3))
                    listImages.append(roi1)
                    listImages.append(roi2)
                    listImages.append(roi3)
                    cpt += 3
                else:
                    roi1 = gauss[y:y + height, x:x + int(width / 4)]
                    roi2 = gauss[y:y + height, x + int(width / 4):x + int(2 * width / 4)]
                    roi3 = gauss[y:y + height, x + int(2 * width / 4):x + int(3 * width / 4)]
                    roi4 = gauss[y:y + height, x + int(3 * width / 4):x + width]
                    listAbsImages.append(x)
                    listAbsImages.append(x + int(width / 4))
                    listAbsImages.append(x + int(2 * width / 4))
                    listAbsImages.append(x + int(3 * width / 4))
                    listImages.append(roi1)
                    listImages.append(roi2)
                    listImages.append(roi3)
                    listImages.append(roi4)
                    cpt += 4

        if (cpt < numberLetters):
            images2Add = numberLetters - cpt
            new_im = np.zeros([28, 28], dtype=np.uint8)
            new_im.fill(255)
            for i in range(images2Add):
                listImages.append(new_im)
                listAbsImages.append(280 - images2Add + i)
        elif (cpt > numberLetters):
            images2Supp = cpt - numberLetters
            for i in range(images2Supp):
                meanValues = [listImages[i].mean(axis=0).mean(axis=0) for i in range(len(listImages))]
                maxiWhite = max(meanValues)
                maxiWhiteIndex = meanValues.index(maxiWhite)
                del listImages[maxiWhiteIndex]
                del listAbsImages[maxiWhiteIndex]


        for i in range(len(listImages)):
            mini = min(listAbsImages)
            index = listAbsImages.index(mini)
            label = list_char.index(code[i])
            image_count[label] += 1
            cv.imwrite(str(label) + "_" + str(int(image_count[label])) + ".png", listImages[index])
            del listAbsImages[index]
            del listImages[index]
            
    os.chdir(main_path)

    #cv.waitKey(0)