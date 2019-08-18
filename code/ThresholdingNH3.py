import codecs
import cv2
import numpy as np
from matplotlib import pyplot as plt

# filename = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/Tiff2D/HL-60_in_collagen_8bit_t004_z084.tif'
# test_img = cv2.imread(filename)

def CompareThreshold(img):
    
    # img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_Gray = img
    img_GaussBlur = cv2.GaussianBlur(img_Gray,(5,5),0)
    img_MedBlur = cv2.medianBlur(img_Gray,5)
    ret1, img_Simple = cv2.threshold(img_GaussBlur,60,255,cv2.THRESH_BINARY)
    

    img_Adaptive = cv2.adaptiveThreshold(img_MedBlur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
    ret3, img_Otsu = cv2.threshold(img_GaussBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    method_Name = ['Src','MedBlur','GaussBlur','Simple thresholding','Adaptive thresholding','Otsu\'s thresholding']
    images = [img_Gray,img_MedBlur,img_GaussBlur,img_Simple,img_Adaptive,img_Otsu]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(method_Name[i])
        plt.xticks([]),plt.yticks([])
    
    # plt.show()
    plt.ion
    plt.pause(0.1)
    plt.clf
    
    

# CompareThreshold(test_img)

# input('')

