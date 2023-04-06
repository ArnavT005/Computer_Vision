import cv2
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load_K", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--load_RT", default=False, action=argparse.BooleanOptionalAction)

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
    cv2.imwrite(f"Dataset/Calibration/Chessboard/calibration_{imgNum}.jpg", img)
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
    global img, imgPoints, imgNum
    squareSide = 3.175
    realPoints = []
    for i in range(5):
        for j in range(5):
            realPoints.append(np.array([[(i + 2) * squareSide], [(j + 2) * squareSide], [1.0]]))
    cv2.namedWindow("TableTop")
    cv2.setMouseCallback("TableTop", onMouseClick)
    while True:
        cv2.imshow("TableTop", img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow("TableTop")
    cv2.imwrite(f"Dataset/Calibration/TableTop/calibration_{imgNum}.jpg", img)
    imgPoints = [K_inv @ np.append(imgPoint, 1.0).reshape((3, 1)) for imgPoint in imgPoints]
    partialMatrix, _ = cv2.findHomography(np.array(realPoints), np.array(imgPoints))
    partialMatrix /= np.linalg.norm(partialMatrix[:, 0])
    r1, r2, t = partialMatrix[:, 0], partialMatrix[:, 1], partialMatrix[:, 2]
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    R = np.concatenate((r1.reshape((3, 1)), r2.reshape((3, 1)), r3.reshape((3, 1))), axis=-1)
    if np.linalg.det(R) > 0:
        R = np.concatenate((r1.reshape((3, 1)), r2.reshape((3, 1)), -r3.reshape((3, 1))), axis=-1)
    return R, t.reshape((3, 1))

def getCubeMesh():
    squareSide = 3.175
    vertices = [
        np.array([2 * squareSide, 2 * squareSide, 0, 1.0]).reshape((4, 1)),
        np.array([6 * squareSide, 2 * squareSide, 0, 1.0]).reshape((4, 1)),
        np.array([6 * squareSide, 6 * squareSide, 0, 1.0]).reshape((4, 1)),
        np.array([2 * squareSide, 6 * squareSide, 0, 1.0]).reshape((4, 1)),
        np.array([2 * squareSide, 2 * squareSide, 4 * squareSide, 1.0]).reshape((4, 1)),
        np.array([6 * squareSide, 2 * squareSide, 4 * squareSide, 1.0]).reshape((4, 1)),
        np.array([6 * squareSide, 6 * squareSide, 4 * squareSide, 1.0]).reshape((4, 1)),
        np.array([2 * squareSide, 6 * squareSide, 4 * squareSide, 1.0]).reshape((4, 1))
    ]
    triangles = [
        [0, 3, 2],
        [0, 2, 1],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
        [4, 5, 6],
        [4, 6, 7]
    ]
    faceNormals = [
        np.array([0, 0, -1]).reshape((3, 1)),
        np.array([0, 0, -1]).reshape((3, 1)),
        np.array([0, -1, 0]).reshape((3, 1)),
        np.array([0, -1, 0]).reshape((3, 1)),
        np.array([1, 0, 0]).reshape((3, 1)),
        np.array([1, 0, 0]).reshape((3, 1)),
        np.array([0, 1, 0]).reshape((3, 1)),
        np.array([0, 1, 0]).reshape((3, 1)),
        np.array([-1, 0, 0]).reshape((3, 1)),
        np.array([-1, 0, 0]).reshape((3, 1)),
        np.array([0, 0, 1]).reshape((3, 1)),
        np.array([0, 0, 1]).reshape((3, 1))
    ]
    return vertices, triangles, faceNormals

def crossProduct2D(a, b):
    return a[0] * b[1] - a[1] * b[0]

def orientCounterClockwise(pos):
    ab = pos[1] - pos[0]
    ac = pos[2] - pos[0]
    if crossProduct2D(ab, ac) < 0:
        return [0, 2, 1]
    else:
        return [0, 1, 2]

def inTriangle(x, y, pos):
    point = np.array([x, y]).reshape((2, 1))
    ap, ab = point - pos[0], pos[1] - pos[0]
    bp, bc = point - pos[1], pos[2] - pos[1]
    cp, ca = point - pos[2], pos[0] - pos[2]
    if crossProduct2D(ab, ap) < 0 or crossProduct2D(bc, bp) < 0 or crossProduct2D(ca, cp) < 0:
        return False
    return True

def distance(point, pos):
    return np.abs(point[0] * (pos[0][1] - pos[1][1]) + point[1] * (pos[1][0] - pos[0][0]) + pos[0][0] * pos[1][1] - pos[0][1] * pos[1][0])

def phi(index, point, pos):
    sidePos = []
    for i in range(3):
        if i != index:
            sidePos.append(pos[i])
    return distance(point, sidePos) / distance(pos[index], sidePos)

def rasterize(posWorld, pos, depth, normal, depthBuffer, img, lightSources, ka, kd):
    minX, maxX = int(min([pos[0][0], pos[1][0], pos[2][0]])), int(max([pos[0][0], pos[1][0], pos[2][0]]) + 1)
    minY, maxY = int(min([pos[0][1], pos[1][1], pos[2][1]])), int(max([pos[0][1], pos[1][1], pos[2][1]]) + 1)
    order = orientCounterClockwise(pos)
    posWorld = [posWorld[i] for i in order]
    pos = [pos[i] for i in order]
    depth = [depth[i] for i in order]
    count = 0
    for i in range(minX, maxX):
        for j in range(minY, maxY):
            if inTriangle(i, j, pos):
                count += 1
                weights = [phi(0, (i, j), pos), phi(1, (i, j), pos), phi(2, (i, j), pos)]
                depth_ = 1.0 / (weights[0] / depth[0] + weights[1] / depth[1] + weights[2] / depth[2])
                if depth_ >= depthBuffer[imgShape[0] - 1 - j][i]:
                    continue
                posWorld_ = (weights[0] * posWorld[0] / depth[0] + weights[1] * posWorld[1] / depth[1] + weights[2] * posWorld[2] / depth[2]) * depth_
                diffuseColor = 0
                for lightSource in lightSources:
                    lightVector_ = (lightSource - posWorld_) / np.linalg.norm(lightSource - posWorld_)
                    diffuseColor += kd * max(0, np.dot(lightVector_, normal))
                depthBuffer[imgShape[0] - 1 - j][i] = depth_
                img[imgShape[0] - 1 - j][i] = 255 * (ka + diffuseColor / len(lightSources)) * np.array([1, 1, 1])
    
def main():
    global img, imgPoints, imgShape, imgNum
    horizontalVanishingPoints, verticalVanishingPoints = [], []
    if args.load_K:
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
    for i in range(4):
        img = cv2.imread(f"Dataset/TableTop/tabletop_{i}.jpg")
        imgCopy = img.copy()
        imgPoints = []
        imgShape = img.shape
        imgNum = i
        if args.load_RT:
            with open(f"CameraModel/extrinsic_{i}.pkl", "rb") as fp:
                P = K @ pickle.load(fp)
        else:
            print(f"Choose calibration points in image: {i} ...")
            R, t = getRotationAndTranslation(np.linalg.inv(K))
            print(f"Rotation and translation computation: SUCCESS")
            with open(f"CameraModel/extrinsic_{i}.pkl", "wb") as fp:
                pickle.dump(np.concatenate((R, t), axis=-1), fp)
            P = K @ np.concatenate((R, t), axis=-1)
        verticesWorld, triangles, faceNormals = getCubeMesh()
        verticesCamera = [np.linalg.inv(K) @ P @ vertex for vertex in verticesWorld]
        verticesDepth = [vertex[2, 0] for vertex in verticesCamera]
        verticesImage = [P @ vertex for vertex in verticesWorld]
        verticesImage = [vertex[:2, :] / vertex[2, 0] for vertex in verticesImage]
        depthBuffer = np.inf * np.ones(imgShape[:2])
        lightSources = [np.array([-100, -100, 250]), np.array([100, 50, 0])]
        ka = np.array([0, 0, 0.4])
        kd = np.array([0, 0, 0.6])
        for index, (triangle, normal) in enumerate(zip(triangles, faceNormals)):
            i0, i1, i2 = triangle
            posWorld = [verticesWorld[i0][:3, 0], verticesWorld[i1][:3, 0], verticesWorld[i2][:3, 0]]
            pos = [verticesImage[i0], verticesImage[i1], verticesImage[i2]]
            depth = [verticesDepth[i0], verticesDepth[i1], verticesDepth[i2]]
            rasterize(posWorld, pos, depth, normal, depthBuffer, imgCopy, lightSources, ka, kd)
            print(f"Rasterized triangle {index}...")
        cv2.imwrite(f"Dataset/Output/AR_{i}.jpg", imgCopy)
        print(f"Image augmentation complete for scene {i}\n")

if __name__ == "__main__":
    main()