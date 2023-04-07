import cv2
import numpy as np

SQUARE_SIDE = 3.175

img = None
imgPoints = []
imgShape = ()
imgNum = -1

def onMouseClick(event, x, y, *_):
    global img, imgPoints, imgShape
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 20, (255, 255, 255), -1)
        imgPoints.append(np.array([x, imgShape[0] - 1 - y]))