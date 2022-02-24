# use opencv to open iamge - open source computer vision library
from turtle import hideturtle
import cv2
from cv2 import COLOR_RGB2BGR
import numpy as np

""" 
Steps:

1) open image using cv2 library
2) convert color image to gray
3) reduce noise - find average pixel using kernel and smooth out #convert image to gray scale (gradient by pixel) -> easier to commute than diff color pixels
4) idenitfy edges in image - find sharp change in pixel or gradient aka color # edge detection - identify boundaries of images where there is sharp change of image/color
5)
"""

# read image from file - return mulit-dimension pi array w/ each pixel
image = cv2.imread('test_image.jpg')

# copy our array to new variable to make gray scale coloring
lane_image = np.copy(image)

# convert image from one color to gray using "cvtColor"
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# reduce noise + smooth out using 5 x 5 kernel
blur = cv2.GaussianBlur(gray, (5,5), 0)

# render image - name of window (result) where image shown
# cv2.imshow('result', image) //shows color image
# cv2.imshow('result', gray)

# displays image for amt of 1000 secs
cv2.waitKey(1000)

# use math. function to spot the edges in pixel aka gradient AKA Canny()
