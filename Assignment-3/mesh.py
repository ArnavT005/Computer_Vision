import numpy as np

import globals

def getCubeMesh():
    vertices = [
        np.array([2 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([2 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([2 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 4 * globals.SQUARE_SIDE, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 4 * globals.SQUARE_SIDE, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 4 * globals.SQUARE_SIDE, 1.0]).reshape((4, 1)),
        np.array([2 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 4 * globals.SQUARE_SIDE, 1.0]).reshape((4, 1))
    ]
    triangles = [[0, 3, 2], [0, 2, 1], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]]
    normals = [
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
    return vertices, triangles, normals

def getPyramidMesh():
    vertices = [
        np.array([2 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([6 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([2 * globals.SQUARE_SIDE, 6 * globals.SQUARE_SIDE, 0, 1.0]).reshape((4, 1)),
        np.array([4 * globals.SQUARE_SIDE, 4 * globals.SQUARE_SIDE, 2 * globals.SQUARE_SIDE, 1.0]).reshape((4, 1)),
    ]
    triangles = [[0, 3, 2], [0, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    normals = [
        np.array([0, 0, -1]).reshape((3, 1)),
        np.array([0, 0, -1]).reshape((3, 1)),
        np.array([0, -1, 1]).reshape([3, 1]) / np.sqrt(2),
        np.array([1, 0, 1]).reshape([3, 1]) / np.sqrt(2),
        np.array([0, 1, 1]).reshape([3, 1]) / np.sqrt(2),
        np.array([-1, 0, 1]).reshape([3, 1]) / np.sqrt(2)
    ]
    return vertices, triangles, normals