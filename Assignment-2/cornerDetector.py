import cv2
import numpy as np
from skimage.feature import peak_local_max

class CornerDetector:
    def __init__(self, window, kappa, thresh, top_k, nms_radius):
        """Initializes corner detector module (Harris).
        
        Function parameters:\\
        window:     dict: window parameters (type and size (square))\\
        kappa:     float: Harris response parameter\\
        thresh:    float: Harris response threshold fraction\\
        top_k:       int: Harris response threshold count (top k)\\
        nms_radius:  int: Non-Maximum Suppression radius
        
        Returns nothing.
        """
        self.window_type = window["type"]
        self.window_size = window["size"]
        self.kappa = kappa
        self.thresh = thresh
        self.top_k = top_k
        self.nms_radius = nms_radius
    
    def find(self, img):
        """Finds interest points (corners) in given image.
        
        Function parameters:\\
        img: array[R, C, 3]: input RGB image
        
        Returns list of interest points (corners).
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        del_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        del_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        img_del_x = cv2.filter2D(img_gray.astype(np.float32), -1, del_x_filter)
        img_del_y = cv2.filter2D(img_gray.astype(np.float32), -1, del_y_filter)
        img_del_x_del_x = img_del_x ** 2
        img_del_x_del_y = img_del_x * img_del_y
        img_del_y_del_y = img_del_y ** 2
        if self.window_type == "gaussian":
            w_img_del_x_del_x = cv2.GaussianBlur(img_del_x_del_x, (self.window_size, self.window_size), 0)
            w_img_del_x_del_y = cv2.GaussianBlur(img_del_x_del_y, (self.window_size, self.window_size), 0)
            w_img_del_y_del_y = cv2.GaussianBlur(img_del_y_del_y, (self.window_size, self.window_size), 0)
        else:
            avg_filter = np.ones((self.window_size, self.window_size), dtype=np.int32)
            w_img_del_x_del_x = cv2.filter2D(img_del_x_del_x, -1, avg_filter)
            w_img_del_x_del_y = cv2.filter2D(img_del_x_del_y, -1, avg_filter)
            w_img_del_y_del_y = cv2.filter2D(img_del_y_del_y, -1, avg_filter)
        det_harris = w_img_del_x_del_x * w_img_del_y_del_y - w_img_del_x_del_y ** 2
        trace_harris = w_img_del_x_del_x + w_img_del_y_del_y
        # Harris and Stephens (1988)
        response_harris = det_harris - self.kappa * (trace_harris ** 2)
        threshold_mask = np.where(np.unique(-response_harris, return_inverse=True)[1] < self.top_k, 1, 0).reshape(response_harris.shape)
        threshold_harris = threshold_mask * response_harris
        threshold_mask = (threshold_harris > self.thresh * threshold_harris.max()) * 1
        threshold_harris = threshold_mask * threshold_harris
        # Non-Maximum Suppression
        max_indices = peak_local_max(threshold_harris, min_distance=self.nms_radius)
        max_mask = np.zeros_like(threshold_harris, dtype=bool)
        max_mask[tuple(max_indices.T)] = True
        return max_mask