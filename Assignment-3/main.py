import cv2
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load_K", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--load_RT", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--augment", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pyramid", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--color", type=str, default="red")
parser.add_argument("--heuristic", default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

import mesh
import globals
import rasterizer
import calibration
    
def main():
    horizontalVanishingPoints, verticalVanishingPoints = [], []
    if args.load_K:
        with open("CameraModel/intrinsic.pkl", "rb") as fp:
            K = pickle.load(fp)
    else:
        for i in range(20):
            globals.img = cv2.imread(f"Dataset/Chessboard/chessboard_{i}.jpg")
            globals.imgPoints = []
            globals.imgShape = globals.img.shape
            globals.imgNum = i
            print(f"Choose calibration points in image: {i} ...")
            horizontalVanishingPoint, verticalVanishingPoint = calibration.getOrthogonalDirections()
            print(f"Vanishing points computation: SUCCESS")
            horizontalVanishingPoints.append(horizontalVanishingPoint)
            verticalVanishingPoints.append(verticalVanishingPoint)
        with open("CameraModel/horizontal.pkl", "wb") as fp:
            pickle.dump(horizontalVanishingPoints, fp)
        with open("CameraModel/vertical.pkl", "wb") as fp:
            pickle.dump(verticalVanishingPoints, fp)
        K = calibration.getCameraIntrinsicMatrix(horizontalVanishingPoints, verticalVanishingPoints)
        with open("CameraModel/intrinsic.pkl", "wb") as fp:
            pickle.dump(K, fp)
    if not args.augment:
        return
    for i in range(4):
        globals.img = cv2.imread(f"Dataset/TableTop/tabletop_{i}.jpg")
        imgCopy = globals.img.copy()
        globals.imgPoints = []
        globals.imgShape = globals.img.shape
        globals.imgNum = i
        if args.load_RT:
            with open(f"CameraModel/extrinsic_{i}.pkl", "rb") as fp:
                P = K @ pickle.load(fp)
        else:
            print(f"Choose calibration points in image: {i} ...")
            R, t = calibration.getRotationAndTranslation(np.linalg.inv(K))
            print(f"Rotation and translation computation: SUCCESS")
            with open(f"CameraModel/extrinsic_{i}.pkl", "wb") as fp:
                pickle.dump(np.concatenate((R, t), axis=-1), fp)
            P = K @ np.concatenate((R, t), axis=-1)
        if args.pyramid:
            verticesWorld, triangles, normalsWorld = mesh.getPyramidMesh()
            lightVectors = [np.array([-50, -100, 100])]
        else:
            verticesWorld, triangles, normalsWorld = mesh.getCubeMesh()
            lightVectors = [np.array([-50, -100, 100]), np.array([100, 50, 50]), np.array([-50, 100, 100])]
        verticesCamera = [np.linalg.inv(K) @ P @ vertex for vertex in verticesWorld]
        verticesDepth = [vertex[2, 0] for vertex in verticesCamera]
        verticesImage = [P @ vertex for vertex in verticesWorld]
        verticesImage = [vertex[:2, :] / vertex[2, 0] for vertex in verticesImage]
        normalsTransform = np.linalg.inv(np.concatenate((np.linalg.inv(K) @ P, np.array([[0, 0, 0, 1]])), axis=0).T)
        normalsCamera = [normalsTransform @ np.concatenate((normal, np.array([[0]])), axis = 0) for normal in normalsWorld]
        normalsCamera = [normal / np.linalg.norm(normal) for normal in normalsCamera]
        triangleCentres = [(verticesCamera[triangle[0]] + verticesCamera[triangle[1]] + verticesCamera[triangle[2]]) / 3.0 for triangle in triangles]
        raysCamera = [-centre / np.linalg.norm(centre) for centre in triangleCentres]
        visibility = [np.dot(raysCamera[i].T[0], normalsCamera[i][:3, :].T[0]) >= 0 for i in range(len(triangles))]
        if args.color == "blue":
            ka = np.array([0.3, 0, 0])
            kd = np.array([0.7, 0, 0])
        elif args.color == "green":
            ka = np.array([0, 0.3, 0])
            kd = np.array([0, 0.7, 0])
        else:
            ka = np.array([0, 0, 0.3])
            kd = np.array([0, 0, 0.7])
        depthBuffer = np.inf * np.ones(globals.imgShape[:2])
        for index in range(len(triangles)):
            if args.heuristic and not visibility[index]:
                continue
            i0, i1, i2 = triangles[index]
            pos = [verticesImage[i0], verticesImage[i1], verticesImage[i2]]
            if args.heuristic:
                rasterizer.rasterizeH(imgCopy, lightVectors, ka, kd, pos, normalsWorld[index])
            else:
                depth = [verticesDepth[i0], verticesDepth[i1], verticesDepth[i2]]
                rasterizer.rasterize(imgCopy, lightVectors, ka, kd, pos, normalsWorld[index], depth, depthBuffer)            
                print(f"Rasterized triangle {index}...")
        if args.pyramid:
            cv2.imwrite(f"Dataset/Output/pyramid_{i}.jpg", imgCopy)
        else:
            cv2.imwrite(f"Dataset/Output/cube_{i}.jpg", imgCopy)
        print(f"Image augmentation complete for scene {i}")

if __name__ == "__main__":
    main()