import cv2
from cornerDetector import CornerDetector

window = {
    "type": "gaussian",
    "size": 5,
}
cornerDetector = CornerDetector(window, 0.04, 0.01, 2)
img = cv2.imread("Dataset/1/image 0.jpg")
dst = cornerDetector.find(img)
print(((dst > 0) * 1).sum())
dst = cv2.dilate(dst,None)
img[dst > 0] = [0, 0, 255]
cv2.imwrite("harris_our.png", img)