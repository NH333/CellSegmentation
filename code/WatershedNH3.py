import codecs
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Segmenter(object):
   def __init__(self):
      self._mask_32S = None
      self._waterImg = None
# 将掩膜转化为CV_32S
   def setMark(self, mask):
      self._mask_32S = np.int32(mask)
# 进行分水岭操作
   def waterProcess(self, img):
      self._waterImg = cv2.watershed(img, self._mask_32S)
# 获取分割后的8位图像
   def getSegmentationImg(self):
      segmentationImg = np.uint8(self._waterImg)
      return segmentationImg
# 处理分割后图像的边界值
   def getWaterSegmentationImg(self):
      waterSegmentationImg = np.copy(self._waterImg)
      waterSegmentationImg[self._waterImg == -1] = 1
      waterSegmentationImg = np.uint8(waterSegmentationImg)
      return waterSegmentationImg
# 将分水岭算法得到的图像与源图像合并 实现抠图效果
   def mergeSegmentationImg(self,src_img,waterSegmentationImg, isWhite = False):
      _, segmentMask = cv2.threshold(waterSegmentationImg, 250, 1, cv2.THRESH_BINARY)
      segmentMask = cv2.cvtColor(segmentMask, cv2.COLOR_GRAY2BGR)
      mergeImg = cv2.multiply(src_img, segmentMask)
      if isWhite is True:
         mergeImg[mergeImg == 0] = 255
      return mergeImg

# def getBoundingRect(img, pattern):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     x, y, w, h = cv2.boundingRect(contours[1])
#     cv2.rectangle(pattern, (x, y), (x + w, y + h), (0, 0, 200), 2)
#     return pattern


     

def WaterNH3(img):
    # filename = 'C:/Users/NH3/Desktop/python_ex/Segmentation/image/image1/Tiff2D/HL-60_in_collagen_8bit_t004_z084.tif'
    # test_img = cv2.imread(filename)
    test_img = img
    # markers = cv2.watershed(test_img,markers) #test_image:待处理图像，8位; markers:生成的掩模，32位单通道图像
    img_Gray = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    img_Blur = cv2.blur(img_Gray,(3,3))
    ret,img_Threshold = cv2.threshold(img_Blur,12,255,cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img_Front = cv2.morphologyEx(img_Threshold,cv2.MORPH_OPEN,kernel1)

    # 获取背景图片
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_Dilate = cv2.dilate(img_Threshold, kernel2, iterations=4)
    ret, img_BackGroung = cv2.threshold(img_Dilate, 1, 128, cv2.THRESH_BINARY)

    # 合成掩膜
    img_Mask = cv2.add(img_Front, img_BackGroung)
    mySegmenter = Segmenter()
    mySegmenter.setMark(img_Mask)

    # 进行分水岭操作 并获得分割图像
    mySegmenter.waterProcess(test_img)
    waterSegmentationImg = mySegmenter.getWaterSegmentationImg()
    outputImgWhite = mySegmenter.mergeSegmentationImg(test_img,waterSegmentationImg,True)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    img_Dilate = cv2.dilate(waterSegmentationImg, kernel3)
    ret, img_Water = cv2.threshold(img_Dilate, 130, 255, cv2.THRESH_BINARY)

    # img_Bound = getBoundingRect(img_Water,test_img)


    img_Name = ['Src','Blur','Threshold','Front','BackGround','Mask','Water',]
    img_Method = [img_Gray,img_Blur,img_Threshold,img_Front,img_BackGroung,img_Mask,img_Water]
    
    
    

   
    for i in range(7):
        plt.subplot(2,4,i+1),plt.imshow(img_Method[i],'gray')
        plt.title(img_Name[i])
        plt.xticks([]),plt.yticks([])
    
    plt.ion
    plt.pause(0.01)
    plt.clf()
    return img_Method
   #  return img_Mask
    # plt.show()
    # plt.get_current_fig_manager().pygame.display.toggle_fullscreen()
   #  plt.ion
   #  plt.pause(0.01)
   #  plt.clf()
    # plt.close()



# input('')
