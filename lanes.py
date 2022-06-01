# use opencv to open iamge - open source computer vision library
from sys import displayhook
from turtle import hideturtle
import cv2
from cv2 import COLOR_RGB2BGR
import numpy as np
import matplotlib.pyplot as plt

""" 
Steps:
1) open image using cv2 library
2) convert color image to gray
3) reduce noise - find average pixel using kernel and smooth out #convert image to gray scale (gradient by pixel) -> easier to commute than diff color pixels
4) idenitfy edges in image and outline - find sharp change in pixel or gradient aka color # edge detection - identify boundaries of images where there is sharp change of image/color
5) define the lane shape and crop 
6) find and connect the lane shape to straight lines
7) blend / combine orignal image with lane line image 
"""

def make_coordinates(image, line_paramters):
    slope, intercept = line_paramters

    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    # contains coordinates of left / right lanes
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    # convert image from one color to gray using "cvtColor"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # reduce noise + smooth out using 5 x 5 kernel
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # use math. function to spot/trace the edges in pixel aka gradient AKA Canny(image, low_threshold, high_threshold)
    canny = cv2.Canny(blur, 50, 150) 
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) #change from 2d to 1d
            cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10) # draw on image from (x1,y1) and (x2,y2)
    return line_image

def region_of_interest(image):
    # height of the image
    height = image.shape[0]

    # declare lane shape using coordinates of plot
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])

    # another image that is just black/white base on the shape of lane
    mask = np.zeros_like(image)

    # fill in the polygon / shape of lane
    cv2.fillPoly(mask, polygon, 255)

    # take bitwise of both images - result in outlining the orginial lane base on polygin
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


# read image from file - return mulit-dimension pi array w/ each pixel
image = cv2.imread('test_image.jpg')

# copy our array to new variable to make gray scale coloring
lane_image = np.copy(image)

""""
# outline / trace the edges in image
canny_image = canny(lane_image)

# outline only the lanes in mask
cropped_image = region_of_interest(canny_image)

# detect straight lines from crop image / lanes using HoughLinesP(image, pixel, radians, threshold, placeholder array, length of line that accept to output, max distance of pixel to connect lines) 
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
average_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, average_lines)

# combine 2 image: add pixels to blend both image using addWeighted(image1, weight to multiply pixel, image2, weight to mulitply pixel, gamma value)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# render image - name of window (result) where image shown
cv2.imshow('result', combo_image)

# displays image for amt of 1000 secs
cv2.waitKey(10000)

# render plot of image + display images for 1000 sec
    #plt.imshow(canny)
    #plt.show()


"""
"""
Video Capture
"""

cap = cv2.VideoCapture('test2.mp4')

while(cap.isOpened()):
    _, frame = cap.read()

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
