import cv2
import numpy as np

import globals

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
    edge = []
    for i in range(3):
        if i != index:
            edge.append(pos[i])
    return distance(point, edge) / distance(pos[index], edge)

def rasterize(img, lightVectors, ka, kd, pos, normal, depth, depthBuffer):
    minX, maxX = int(min([pos[0][0], pos[1][0], pos[2][0]])), int(max([pos[0][0], pos[1][0], pos[2][0]]) + 1)
    minY, maxY = int(min([pos[0][1], pos[1][1], pos[2][1]])), int(max([pos[0][1], pos[1][1], pos[2][1]]) + 1)
    order = orientCounterClockwise(pos)
    pos = [pos[i] for i in order]
    depth = [depth[i] for i in order]
    for i in range(minX, maxX):
        for j in range(minY, maxY):
            if inTriangle(i, j, pos):
                weights = [phi(k, (i, j), pos) for k in range(3)]
                depth_ = 1.0 / (weights[0] / depth[0] + weights[1] / depth[1] + weights[2] / depth[2])
                if depth_ >= depthBuffer[globals.imgShape[0] - 1 - j][i]:
                    continue
                diffuseColor = 0
                for lightVector in lightVectors:
                    diffuseColor += kd * max(0, np.dot(lightVector / np.linalg.norm(lightVector), normal))
                depthBuffer[globals.imgShape[0] - 1 - j][i] = depth_
                img[globals.imgShape[0] - 1 - j][i] = 255 * (ka + diffuseColor / len(lightVectors)) * np.array([1, 1, 1])
    
def rasterizeH(img, lightVectors, ka, kd, pos, normal):
    diffuseColor = 0
    for lightVector in lightVectors:
        diffuseColor += kd * max(0, np.dot(lightVector / np.linalg.norm(lightVector), normal))
    color = 255.0 * (ka + diffuseColor / len(lightVectors)) * np.array([1, 1, 1])
    pts = np.stack([p.T[0] for p in pos])
    pts[:, 1] = globals.imgShape[0] - 1 - pts[:, 1]
    cv2.fillPoly(img, np.int32([pts]), color)