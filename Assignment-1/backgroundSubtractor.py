import numpy as np
import cv2

class BackgroundSubtractor:
    # Initializes background subtractor module (Mixture of Gaussians)
    #   K:            int: number of gaussians in the mixture
    #   alpha:      float: learning rate
    #   T:          float: cumulative threshold for filtering background distributions
    #   num_rows:     int: number of rows in image
    #   num_cols:     int: number of columns in image
    #   num_channels: int: number of channels in image
    def __init__(self, K, alpha, T, initial_data):
        # set model parameters
        self.K = K
        self.alpha = alpha
        self.T = T
        self.initial_weight = initial_data["weight"]
        self.initial_variance = initial_data["variance"]
        self.initial_image = initial_data["image"]
        self.num_rows, self.num_cols = self.initial_image.shape[:2]
        if len(self.initial_image.shape) > 2:
            self.num_channels = self.initial_image.shape[2]
        else:
            self.num_channels = 1
            self.initial_image = self.initial_image.reshape([self.num_rows, self.num_cols, self.num_channels])
        # initialize MOG parameters
        self.mean = np.zeros((self.num_rows, self.num_cols, self.K, self.num_channels), dtype=np.float32)
        self.mean[:, :, self.K - 1] = self.initial_image.copy()
        self.mean_square = np.zeros((self.num_rows, self.num_cols, self.K), dtype=np.float32)
        self.mean_square[:, :, self.K - 1] = np.linalg.norm(self.initial_image, axis=2) ** 2
        self.weight = np.zeros((self.num_rows, self.num_cols, self.K), dtype=np.float32)
        self.weight[:, :, self.K - 1] = 1
        self.variance = np.full((self.num_rows, self.num_cols, self.K), self.initial_variance, dtype=np.float32)

    # Trains background subtractor module on given dataset
    #   dataset:         list: list of images (except first one)
    #   output_folder: string: path to save images
    def train(self, dataset, output_folder):
        for idx, image in enumerate(dataset):
            if self.num_channels == 1:
                image = image.reshape([self.num_rows, self.num_cols, self.num_channels])
            match_found = np.full((self.num_rows, self.num_cols), False, dtype=bool)
            for k in reversed(range(self.K)):
                match_locations = np.where(np.logical_and(np.linalg.norm(image - self.mean[:, :, k], axis=-1) < 2.5 * np.sqrt(self.variance[:, :, k]), np.logical_not(match_found)))
                if not match_locations[0].any():
                    continue
                match_found[match_locations] = True
                weight_copy = self.weight[match_locations].copy()
                weight_copy = (1 - self.alpha) * weight_copy
                weight_copy[:, k] += self.alpha
                self.weight[match_locations] = weight_copy.copy()
                rho = self.alpha * self.gaussian_pdf(image[match_locations], self.mean[match_locations][:, k], self.variance[match_locations][:, k])
                rho = rho.reshape(rho.shape[0], 1)
                mean_copy = self.mean[match_locations].copy()
                mean_copy[:, k] = (1 - rho) * mean_copy[:, k] + rho * image[match_locations]
                self.mean[match_locations] = mean_copy.copy()
                rho = rho.reshape(rho.shape[0],)
                variance_copy = self.variance[match_locations].copy()
                variance_copy[:, k] = (1 - rho) * variance_copy[:, k] + rho * (np.linalg.norm(image[match_locations] - self.mean[match_locations][:, k], axis=-1) ** 2)
            no_match_locations = np.where(np.logical_not(match_found))
            sort_weights = np.argsort(self.weight[no_match_locations], axis=-1)
            self.weight[no_match_locations] = np.take_along_axis(self.weight[no_match_locations], sort_weights, axis=-1)
            weight_copy = self.weight[no_match_locations].copy()
            weight_copy[:, 0] = self.initial_weight
            sum_weights = np.sum(weight_copy, axis=-1)
            sum_weights = sum_weights.reshape(sum_weights.shape[0], 1)
            weight_copy /= sum_weights
            self.weight[no_match_locations] = weight_copy.copy()
            mean_copy = self.mean[no_match_locations].copy()
            for k in range(self.num_channels):
                mean_copy[:, :, k] = np.take_along_axis(mean_copy[:, :, k], sort_weights, axis=-1)
            mean_copy[:, 0] = image[no_match_locations].copy()
            self.mean[no_match_locations] = mean_copy.copy()
            self.variance[no_match_locations] = np.take_along_axis(self.variance[no_match_locations], sort_weights, axis=-1)
            variance_copy = self.variance[no_match_locations].copy()
            variance_copy[:, 0] = self.initial_variance
            self.variance[no_match_locations] = variance_copy.copy()
            sort_weights_by_std = np.argsort(self.weight / np.sqrt(self.variance), axis=-1)
            self.weight = np.take_along_axis(self.weight, sort_weights_by_std, axis=-1)
            mean_copy = self.mean.copy()
            for k in range(self.num_channels):
                mean_copy[:, :, :, k] = np.take_along_axis(mean_copy[:, :, :, k], sort_weights_by_std, axis=-1)
            self.mean = mean_copy.copy()
            self.variance = np.take_along_axis(self.variance, sort_weights_by_std, axis=-1)
            foreground = self.find_foreground(image)
            filtered_foreground = self.filter_foreground(foreground)
            cv2.imwrite(output_folder + "out" + str(0).zfill(6) + ".png", filtered_foreground)
    
    # Computes multi-dimensional gaussian pdf (scalar covariance matrix)
    #   X:        ndarray[B, num_channels]: input vector
    #   mean:     ndarray[B, num_channels]: mean vector
    #   variance:              ndarray[B,]: variance along a component
    def gaussian_pdf(self, X, mean, variance):
        return np.exp(-0.5 * (np.linalg.norm(X - mean, axis=-1) ** 2) / variance) / ((np.sqrt(2 * np.pi * variance)) ** self.num_channels)
    
    # Finds pixels corresponding to foreground
    #   image:  ndarray[R, C, num_channels]: input image
    def find_foreground(self, image):
        foreground = np.zeros((self.num_rows, self.num_cols), dtype=np.uint8)
        match_found = np.full((self.num_rows, self.num_cols), False, dtype=bool) 
        cumulative_sum = np.zeros((self.num_rows, self.num_cols), dtype=np.float32)
        for k in reversed(range(self.K)):
            match_locations = np.where(np.logical_and(np.logical_and(np.linalg.norm(image - self.mean[:, :, k], axis=-1) < 2.5 * np.sqrt(self.variance[:, :, k]), np.logical_not(match_found)), cumulative_sum < self.T))
            match_found[match_locations] = True
            cumulative_sum += self.weight[:, :, k].copy()
        no_match_locations = np.where(np.logical_not(match_found))
        foreground[no_match_locations] = 255
        return foreground
    
    def filter_foreground(self, foreground):
        numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8, cv2.CV_32S)
        filtered_foreground = np.zeros((foreground.shape), dtype=np.uint8)
        for i in range(1, numLabels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # if w > 150:
            #     continue
            # if area < 900 and (area / (w * h)) < 0.5:
            #     continue
            # if area < 200:
            #     continue
            filtered_foreground += (labels == i).astype("uint8") * 255
        return filtered_foreground


        

                    


