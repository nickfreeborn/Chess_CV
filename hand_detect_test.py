import cv2
import numpy as np
import imutils
import os

def __addHandBoundingBoxes(image):
    inverted = cv2.bitwise_not(image.copy())
    hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    lower = np.array([135, 6, 91])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    green_squares_mask = mask.copy()

    cv2.imwrite('hand_detect_imgs/1green.jpg', green_squares_mask)
    cv2.imwrite('hand_detect_imgs/1notgreen.jpg', cv2.bitwise_not(green_squares_mask.copy()))
    # cv2.imshow('green', green_squares_mask)

    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 81])
    upper = np.array([255, 43, 255])
    mask = cv2.inRange(hsv.copy(), lower, upper)
    mask = 255-mask
    white_squares_mask = mask.copy()

    cv2.imwrite('hand_detect_imgs/2white.jpg', white_squares_mask)
    # cv2.imshow('white', white_squares_mask)

  
    image_final = image.copy()
    image_final = cv2.bitwise_and(image_final, image_final, mask=green_squares_mask)
    image_final = cv2.bitwise_and(image_final, image_final, mask=white_squares_mask)

    cv2.imwrite('hand_detect_imgs/3and.jpg', image_final)
    # cv2.imshow('and', image_final)

  
    inverted = cv2.bitwise_not(image_final.copy())
    hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    target = image.copy()
    lower = np.array([76, 87, 50])
    upper = np.array([255, 255, 255])
    white_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

    cv2.imwrite('hand_detect_imgs/31white_pieces.jpg', white_pieces_mask)


    lower = np.array([0, 0, 159])
    upper = np.array([55, 255, 255])
    black_pieces_mask = cv2.inRange(hsv.copy(), lower, upper)

    cv2.imwrite('hand_detect_imgs/32black_pieces.jpg', black_pieces_mask)

    hand_is_detected, hand_contours = __hand_detected(image_final, white_pieces_mask, black_pieces_mask)
    # if hand_is_detected:
    #   self.__drawHand(target, hand_contours)

    return (target, hand_is_detected)

def __hand_detected(no_houses_frame, white_pieces_mask, black_pieces_mask):
    """
    return `True` or `False` if hand is detected
    """
    white_pieces_mask = 255-white_pieces_mask
    black_pieces_mask = 255-black_pieces_mask

    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=white_pieces_mask)
    cv2.imwrite('hand_detect_imgs/4and_white_pieces.jpg', no_houses_frame)
    # cv2.imshow('no_white_piece', no_houses_frame)
    no_houses_frame = cv2.bitwise_and(no_houses_frame, no_houses_frame, mask=black_pieces_mask)
    cv2.imwrite('hand_detect_imgs/5and_black_pieces.jpg', no_houses_frame)

    # cv2.imshow('image', no_houses_frame)

    # convert image to gray scale
    gray = cv2.cvtColor(no_houses_frame, cv2.COLOR_BGR2GRAY)

    # This is the threshold level for every pixel.
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('hand_detect_imgs/6final_thresh.jpg', thresh)

    # cv2.imshow('thresh', thresh)
    thresh = cv2.erode(thresh, None, iterations=8)
    cv2.imwrite('hand_detect_imgs/7erode.jpg', thresh)

    # cv2.imshow('image', thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts is not None and len(cnts) > 0:
        
        cnt = max(cnts, key=cv2.contourArea)
        return (True, cnt)
    else:
        return (False, None)
    
# path
path = "chess_tracking/assets/videos/hand_frame2.png"

# Reading an image in default mode
image = cv2.imread(path)

# Using cv2.imshow() method
# Displaying the image
# cv2.imshow('image', image)
# print(__addHandBoundingBoxes(image))
__addHandBoundingBoxes(image)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()