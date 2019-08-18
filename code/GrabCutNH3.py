import codecs
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''test'''
########################################
# filename = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/Tiff2D/HL-60_in_collagen_8bit_t004_z080.tif'
# filename_mask = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/mask/mask_80_.tif'

# # img_GaussBlur = cv2.GaussianBlur(img_Gray,(5,5),0)

# test_img = cv2.imread(filename)f
# mask_img = cv2.imread(filename_mask)
########################################
''''''
def GrabCut_NH3(img_src,img_mask):

    # img_Gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    img_Gauss = cv2.GaussianBlur(img_src,(5,5),0)
    # mask_img = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
    mask_img = img_mask
    mask_img_label = mask_img.copy()
    for row in range(mask_img.shape[0]):
        for col in range(mask_img.shape[1]):
            if mask_img[row][col] == 255 :
                mask_img_label[row][col] = 3
            else:
                mask_img_label[row][col] = 0


    cv2.imshow('src',img_src)
    cv2.imshow('mask',mask_img_label*255)

    # mask = np.zeros(test_img.shape[:2],np.float64)


    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    '''not sure'''
    rect = (0,0,200,200) 
    cv2.grabCut(img_Gauss,mask_img_label,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)#cv2.GC_INIT_WITH_MASK
    mask = np.where((mask_img_label==2)|(mask_img_label==0),0,1).astype('uint8')
    cv2.imshow('mask_new',mask*255)

    # cv2.grabCut(test_img,mask_img_label,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask_img_label==2)|(mask_img_label==0),0,1).astype('uint8')
    img = img_src*mask[:,:,np.newaxis]
    cv2.imshow('img_new',img)
    # plt.imshow(img),plt.colorbar(),plt.show()
    cv2.waitKey(50)

# input('')