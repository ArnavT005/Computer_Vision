import cv2
import numpy as np

import globals

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
    W = np.array([[vh[-1][0], vh[-1][1], vh[-1][2]], [vh[-1][1], vh[-1][3], vh[-1][4]], [vh[-1][2], vh[-1][4], vh[-1][5]]])
    if W[2, 2] != 0:
        W /= W[2, 2]
    return W

def getOrthogonalDirections():
    imgNum = globals.imgNum
    imgPoints = globals.imgPoints
    realPoints = []
    for i in range(6):
        realPoints.append(np.array([[globals.SQUARE * i], [1.0]]))
    cv2.namedWindow("Chessboard")
    cv2.setMouseCallback("Chessboard", globals.onMouseClick)
    while True:
        cv2.imshow("Chessboard", globals.img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow("Chessboard")
    cv2.imwrite(f"Dataset/Calibration/Chessboard/calibration_{imgNum}.jpg", globals.img)
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

def getRotationAndTranslation(K_inv):
    imgNum = globals.imgNum
    realPoints = []
    for i in range(5):
        for j in range(5):
            realPoints.append(np.array([[(i + 2) * globals.SQUARE], [(j + 2) * globals.SQUARE], [1.0]]))
    cv2.namedWindow("TableTop")
    cv2.setMouseCallback("TableTop", globals.onMouseClick)
    while True:
        cv2.imshow("TableTop", globals.img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow("TableTop")
    cv2.imwrite(f"Dataset/Calibration/TableTop/calibration_{imgNum}.jpg", globals.img)
    imgPoints = [K_inv @ np.append(imgPoint, 1.0).reshape((3, 1)) for imgPoint in globals.imgPoints]
    partialMatrix, _ = cv2.findHomography(np.array(realPoints), np.array(imgPoints))
    partialMatrix /= np.linalg.norm(partialMatrix[:, 0])
    r1, r2, t = partialMatrix[:, 0], partialMatrix[:, 1], partialMatrix[:, 2]
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    R = np.concatenate((r1.reshape((3, 1)), r2.reshape((3, 1)), r3.reshape((3, 1))), axis=-1)
    if np.linalg.det(R) > 0:
        R = np.concatenate((r1.reshape((3, 1)), r2.reshape((3, 1)), -r3.reshape((3, 1))), axis=-1)
    return R, t.reshape((3, 1))