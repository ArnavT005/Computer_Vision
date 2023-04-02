import cv2
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

img = None
imgPoints = []
imgShape = ()
imgNum = -1

def onMouseClick(event, x, y, flags, param):
    global img, imgPoints, imgShape
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 20, (255, 255, 255), -1)
        imgPoints.append(np.array([x, imgShape[0] - 1 - y]))
    
def findHomography2D(src, dst):
    numPoints = len(src)
    A = np.array([[dst[0][1, 0] * src[0][0, 0], dst[0][1, 0] * src[0][1, 0], - dst[0][0, 0] * src[0][0, 0], - dst[0][0, 0] * src[0][1, 0]]])
    for i in range(1, numPoints):
        A = np.concatenate((A, np.array([[dst[i][1, 0] * src[i][0, 0], dst[i][1, 0] * src[i][1, 0], - dst[i][0, 0] * src[i][0, 0], - dst[i][0, 0] * src[i][1, 0]]])), axis=0)
    _, _, vh = np.linalg.svd(A)
    homography = vh[-1].reshape((2, 2))
    if homography[1, 1] != 0:
        homography /= homography[1, 1]
    return homography

def findIAC(hPts, vPts):
    numPoints = len(hPts)
    A = np.array([[vPts[0][0, 0] * hPts[0][0, 0], vPts[0][0, 0] * hPts[0][1, 0] + vPts[0][1, 0] * hPts[0][0, 0], vPts[0][0, 0] + hPts[0][0, 0], vPts[0][1, 0] * hPts[0][1, 0], vPts[0][1, 0] + hPts[0][1, 0], 1.0]])
    for i in range(1, numPoints):
        A = np.concatenate((A, np.array([[vPts[i][0, 0] * hPts[i][0, 0], vPts[i][0, 0] * hPts[i][1, 0] + vPts[i][1, 0] * hPts[i][0, 0], vPts[i][0, 0] + hPts[i][0, 0], vPts[i][1, 0] * hPts[i][1, 0], vPts[i][1, 0] + hPts[i][1, 0], 1.0]])), axis=0)
    _, _, vh = np.linalg.svd(A)
    w = vh[-1]
    W = np.array([[w[0], w[1], w[2]], [w[1], w[3], w[4]], [w[2], w[4], w[5]]])
    if W[2, 2] != 0:
        W /= W[2, 2]
    return W

def getOrthogonalDirections():
    global img, imgPoints, imgNum
    squareSide = 3.175
    realPoints = []
    for i in range(6):
        realPoints.append(np.array([[squareSide * i], [1.0]]))
    cv2.namedWindow("Chessboard")
    cv2.setMouseCallback("Chessboard", onMouseClick)
    while True:
        cv2.imshow("Chessboard", img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow("Chessboard")
    cv2.imwrite(f"Dataset/Calibration/calibration_{imgNum}.jpg", img)
    horizontalImgPoints, verticalImgPoints = [np.array([[0.0], [1.0]])], [np.array([[0.0], [1.0]])]
    previousDistance = 0.0
    for i in range(1, 6):
        distance = np.sqrt(np.sum(((imgPoints[i] - imgPoints[i - 1]) ** 2)))
        horizontalImgPoints.append(np.array([[distance + previousDistance], [1.0]]))
        previousDistance += distance
    horizontalHomography = findHomography2D(realPoints, horizontalImgPoints)
    horizontalPoint = horizontalHomography @ np.array([[1.0], [0.0]])
    horizontalDistance = horizontalPoint[0, 0] / horizontalPoint[1, 0]
    if imgPoints[1][0] == imgPoints[0][0]:
        horizontalSlope = np.inf
    else:
        horizontalSlope = (imgPoints[1][1] - imgPoints[0][1]) / (imgPoints[1][0] - imgPoints[0][0])
    horizontalConstant = imgPoints[0][1] - horizontalSlope * imgPoints[0][0]
    horizontalVanishingPoint = np.array([[0.0], [0.0], [1.0]])
    if imgPoints[1][0] >= imgPoints[0][0]:
        horizontalVanishingPoint[0, 0] = imgPoints[0][0] + horizontalDistance / np.sqrt(1 + horizontalSlope ** 2)
        horizontalVanishingPoint[1, 0] = horizontalSlope * horizontalVanishingPoint[0, 0] + horizontalConstant
    else:
        horizontalVanishingPoint[0, 0] = imgPoints[0][0] - horizontalDistance / np.sqrt(1 + horizontalSlope ** 2)
        horizontalVanishingPoint[1, 0] = horizontalSlope * horizontalVanishingPoint[0, 0] + horizontalConstant
    previousDistance = 0.0
    for i in range(7, 12):
        distance = np.sqrt(np.sum(((imgPoints[i] - imgPoints[i - 1]) ** 2)))
        verticalImgPoints.append(np.array([[distance + previousDistance], [1.0]]))
        previousDistance += distance
    verticalHomography = findHomography2D(realPoints, verticalImgPoints)
    verticalPoint = verticalHomography @ np.array([[1.0], [0.0]])
    verticalDistance = verticalPoint[0, 0] / verticalPoint[1, 0]
    if imgPoints[7][0] == imgPoints[6][0]:
        verticalSlope = np.inf
    else:
        verticalSlope = (imgPoints[7][1] - imgPoints[6][1]) / (imgPoints[7][0] - imgPoints[6][0])
    verticalConstant = imgPoints[6][1] - verticalSlope * imgPoints[6][0]
    verticalVanishingPoint = np.array([[0.0], [0.0], [1.0]])
    if imgPoints[7][0] >= imgPoints[6][0]:
        verticalVanishingPoint[0, 0] = imgPoints[6][0] + verticalDistance / np.sqrt(1 + verticalSlope ** 2)
        verticalVanishingPoint[1, 0] = verticalSlope * verticalVanishingPoint[0, 0] + verticalConstant
    else:
        verticalVanishingPoint[0, 0] = imgPoints[6][0] - verticalDistance / np.sqrt(1 + verticalSlope ** 2)
        verticalVanishingPoint[1, 0] = verticalSlope * verticalVanishingPoint[0, 0] + verticalConstant    
    return horizontalVanishingPoint, verticalVanishingPoint

def getCameraIntrinsicMatrix(horizontalVanishingPoints, verticalVanishingPoints):
    IAC = findIAC(horizontalVanishingPoints, verticalVanishingPoints)
    K = np.linalg.inv(np.linalg.cholesky(IAC).T)
    K /= K[2, 2]
    return K

def main():
    global img, imgPoints, imgShape, imgNum
    horizontalVanishingPoints, verticalVanishingPoints = [], []
    if args.load:
        with open("CameraModel/intrinsic.pkl", "rb") as fp:
            K = pickle.load(fp)
    elif args.debug:
        with open("CameraModel/horizontal.pkl", "rb") as fp:
            horizontalVanishingPoints = pickle.load(fp)
        with open("CameraModel/vertical.pkl", "rb") as fp:
            verticalVanishingPoints = pickle.load(fp)
        breakpoint()
        K = getCameraIntrinsicMatrix(horizontalVanishingPoints, verticalVanishingPoints)
        with open("CameraModel/intrinsic.pkl", "wb") as fp:
            pickle.dump(K, fp)
    else:
        for i in range(20):
            img = cv2.imread(f"Dataset/Chessboard/chessboard_{i}.jpg")
            imgPoints = []
            imgShape = img.shape
            imgNum = i
            print(f"Choose calibration points in image: {i} ...")
            horizontalVanishingPoint, verticalVanishingPoint = getOrthogonalDirections()
            print(f"Vanishing points computation: SUCCESS")
            horizontalVanishingPoints.append(horizontalVanishingPoint)
            verticalVanishingPoints.append(verticalVanishingPoint)
        with open("CameraModel/horizontal.pkl", "wb") as fp:
            pickle.dump(horizontalVanishingPoints, fp)
        with open("CameraModel/vertical.pkl", "wb") as fp:
            pickle.dump(verticalVanishingPoints, fp)
        K = getCameraIntrinsicMatrix(horizontalVanishingPoints, verticalVanishingPoints)
        with open("CameraModel/intrinsic.pkl", "wb") as fp:
            pickle.dump(K, fp)
    breakpoint()

if __name__ == "__main__":
    main()