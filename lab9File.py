import cv2
import numpy as np
from matplotlib import pyplot as plt

#Variables
nrows = 3
ncols = 4

KernelSizeWidth = 3
KernelSizeHeight = 3

KernelSizeWidth1 = 13
KernelSizeHeight1 = 13

cannyThreshold = 100
cannyParam2 = 200

cannyParams1 = 80
cannyParams2 = 160
#==========================================================================
#Define Variables

img = cv2.imread('GMIT.jpg',)
myImg = cv2.imread('Image2.jpg',)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgIn = gray

gray2 = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)

imgOut = cv2.GaussianBlur(imgIn,(KernelSizeWidth, KernelSizeHeight),0)
sobelHorizontal = cv2.Sobel(imgIn,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(imgIn,cv2.CV_64F,0,1,ksize=5) # y dir
sobelBoth = sobelHorizontal + sobelVertical

canny = cv2.Canny(imgIn,cannyThreshold,cannyParam2)

canny2 = cv2.Canny(gray2,cannyParams1,cannyParams2)

#kernel = np.ones((5,5),np.float32)/25
#dst = cv2.filter2D(img,-1,kernel

#cv2.imshow('Original image',img)
#cv2.imshow('Gray image', gray) 

#====================================================================================
#Printing images

#Original image
plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

#GrayScale image
plt.subplot(nrows, ncols,2),plt.imshow(gray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

#3x3 blurred Image
plt.subplot(nrows, ncols,3),plt.imshow(imgOut, cmap = 'gray')
plt.title('3X3 Blurred'), plt.xticks([]), plt.yticks([])

#13 X 13 blurred image
imgOut = cv2.GaussianBlur(imgIn,(KernelSizeWidth1, KernelSizeHeight1),0)
plt.subplot(nrows, ncols,4),plt.imshow(imgOut, cmap = 'gray')
plt.title('13 X 13 Blurred'), plt.xticks([]), plt.yticks([])

#Sobel horizontal
plt.subplot(nrows, ncols,5),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Horizontal'), plt.xticks([]), plt.yticks([])

#sobel verical
plt.subplot(nrows, ncols,6),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Veritical'), plt.xticks([]), plt.yticks([])

#sobel Both
plt.subplot(nrows, ncols,7),plt.imshow(sobelBoth, cmap = 'gray')
plt.title('Both sobels'), plt.xticks([]), plt.yticks([])

#Canny
plt.subplot(nrows, ncols,8),plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

#My image
plt.subplot(nrows, ncols,9),plt.imshow(myImg, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

#My image
plt.subplot(nrows, ncols,9),plt.imshow(myImg, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

#My image
plt.subplot(nrows, ncols,10),plt.imshow(canny2, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

cv2.imshow('Gray image', canny2) 

#==================================================
#DISPLAY IMAGES

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()