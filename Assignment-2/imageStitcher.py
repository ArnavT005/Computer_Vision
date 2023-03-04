import cv2
import numpy as np
from random import sample

class ImageStitcher:
    def __init__(self, proximity_radius, neighbourhood_radius, sample_count):
        """Initializes image stitcher module.
        
        Function parameters:\\
        proximity_radius:     int: proximity radius used for matching only nearby points\\
        neighbourhood_radius: int: neighbourhood radius used for SSD calculation during matching\\
        sample_count:         int: number of interest points that are matched between two images
        
        Returns nothing.
        """
        self.proximity_radius = proximity_radius
        self.neighbourhood_radius = neighbourhood_radius
        self.sample_count = sample_count

    def match(self, img, mask):
        """Matches interest points of image A with interest points of image B.
        
        Function parameters:\\
        img:  list: list of two images (A and B)\\
        mask: list: list of two interest points mask (A and B) 
        
        Returns tuple of list of matching points in image A and image B
        """
        img_A, img_B = img
        mask_A, mask_B = mask[0].copy(), mask[1].copy()
        match_A, match_B = [], []
        points_A = sample(tuple(np.transpose(np.where(mask_A))), self.sample_count)
        p, n = self.proximity_radius, self.neighbourhood_radius
        for point_A in points_A:
            proximity_mask = np.zeros_like(mask_B, dtype=bool)
            proximity_mask[point_A[0]-p:point_A[0]+p+1, point_A[1]-p:point_A[1]+p+1] = True
            proximity_mask = np.logical_and(proximity_mask, mask_B)
            points_B = tuple(np.transpose(np.where(proximity_mask)))
            min_SSD = np.inf
            match_point = np.array([-1, -1])
            for point_B in points_B:
                SSD = ((img_B[point_B[0]-n:point_B[0]+n+1, point_B[1]-n:point_B[1]+n+1] - img_A[point_A[0]-n:point_A[0]+n+1, point_A[1]-n:point_A[1]+n+1]) ** 2).sum()
                if SSD < min_SSD:
                    min_SSD = SSD
                    match_point = point_B
            if match_point[0] > -1:
                match_A.append(np.array((point_A[1], point_A[0])))
                match_B.append(np.array((match_point[1], match_point[0])))
                mask_B[match_point[0], match_point[1]] = False
        return match_A, match_B
    
    def getAffineTransform2D(self, src, dest):
        """Computes affine transformation from one set of points to another (2D).
        
        Function parameters:\\
        src:  list: source (domain) list of points (2D)\\
        dest: list: destination (range) list of points (2D)
        
        Returns affine transformation matrix (2 by 3) taking src to dest.
        """
        source = np.array(src)
        destination = np.array(dest)
        return cv2.estimateAffine2D(source, destination, method=cv2.RANSAC)[0]
