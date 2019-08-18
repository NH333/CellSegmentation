# /*
#  * @Author: Anqi Chen 
#  * @Date: 2019-08-17 22:49:54 
#  * @Last Modified by: Anqi Chen
#  * @Last Modified time: 2019-08-17 23:11:30
#  */

# /*
# TODO:hello world!
# */

import numpy as np
import glob as gb
import os
import matplotlib.pyplot as plt
import cv2
# import ToVector
from matplotlib import pyplot as plt

import ThresholdingNH3
import WatershedNH3
import GrabCutNH3

import sys


sys.path.append('./')

num_time = 4
num_z = 111
filename = []
filename_mask = []
slices = 141

list_idex = []
dict_file = {}
for i in range(slices):
    list_idex.append(('%d')%(i+1))


img_path = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/Tiff2D'
imgMask_path = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/mask'

'''read img_src '''

for time in range(4,num_time+1):
    for z in range(1,slices+1):
        if z<10:
                tmp_name = ('HL-60_in_collagen_8bit_t00%d_z00%d.tif')%(time,z)
        elif z<100:
                tmp_name = ('HL-60_in_collagen_8bit_t00%d_z0%d.tif')%(time,z)
        else:
                tmp_name = ('HL-60_in_collagen_8bit_t00%d_z%d.tif')%(time,z)

        result_name = os.path.join(img_path,tmp_name)
        filename.append(result_name)



'''read img_mask'''

for z in range(1,slices+1):
    tmp_mask_name = ('mask_%d_.tif')%(z)
    result_mask_name = os.path.join(imgMask_path,tmp_mask_name)
    filename_mask.append(result_mask_name)


for i in range(0, len(list_idex)):#len(list_idex)
    dict_file[i]=(filename[i], filename_mask[i])


mask = np.zeros((293,297),np.uint8)
mask[50:150,100:250] = 255

# plt.figure(figsize=(8,6),dpi=80)
# plt.ion()
count = 1
print('please chose a method:\nA.Threshold B.Water C.GrabCut ')
chose_str = input('input: ')
for i in range(46,118):
    
    

#     if chose_str == 'A' or

    file = dict_file[i][0]
    file_mask = dict_file[i][1]
    
#     plt.cla()

    img = cv2.imread(file)
    img_mask = cv2.imread(file_mask)
    img_mask = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
    img_Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_Gray_Normal =cv2.normalize(img_Gray,None,255,0,cv2.NORM_MINMAX,cv2.CV_8UC1)

    


    if chose_str == 'A': #阈值方法
        ThresholdingNH3.CompareThreshold(img_Gray_Normal)
        # break
    
    elif chose_str == 'B':#分水岭方法
        # tmp_filename_water = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/Image1/Water/'+('Water_%d_.png')%(count)

        #     tmp_filename_mask = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/Image1/mask/'+('mask_%d_.tif')%(count)
        #     cv2.imwrite(tmp_filename_mask)
        #tmp_img = cv2.imread(tmp_filename_water)

        result_img = WatershedNH3.WaterNH3(img)
        # tmp_filename_mask = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/Image1/mask/'+('mask_%d_.tif')%(count)
        # cv2.imwrite(tmp_filename_mask,result_img[5])
   

    else: #GrabCut方法     
        ###先算一下分水岭得到的mask，如果数量很少就continue#####
        #####################################################
        count_mask = 0
        for row in range(img_mask.shape[0]):
            for col in range(img_mask.shape[1]):
                if img_mask[row][col] != 0:
                    count_mask=count_mask+1
        print(('The num of mask is:%d ')%(count_mask))
        ######--not sure--#######
        if count_mask < 3000:
                continue
        ####################################################
        GrabCutNH3.GrabCut_NH3(img,img_mask)



    count = count+1

    
    '''直方图计算'''
#     img_Hist = cv2.calcHist([img_Gray_Normal],[0],None,[256],[0,256])
#     masked_img_Hist = cv2.calcHist([img_Gray_Normal],[0],mask,[256],[0,256])

#     masked_img = cv2.bitwise_and(img_Gray_Normal,img_Gray_Normal,mask=mask)

#     plt.subplot(221),plt.imshow(img_Gray_Normal,'gray')
#     plt.subplot(222),plt.imshow(masked_img,'gray')
#     plt.subplot(223),plt.plot(img_Hist)
#     plt.subplot(224),plt.plot(masked_img_Hist)
    
# #     plt.plot(img_Hist)
#     plt.xlim([0,256])




#     plt.show()

#     cv2.waitKey(60)
#     plt.show()
plt.ioff()
plt.show()

cv2.destroyAllWindows()

input('')