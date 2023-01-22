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
    def __init__(self, K, alpha, T, num_rows, num_cols, num_channels, initial_variance, initial_image):
        # set model parameters
        self.K = K
        self.alpha = alpha
        self.T = T
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels
        self.initial_variance = initial_variance
        self.initial_image = initial_image
        # initialize MOG parameters
        self.mean = np.zeros((self.num_rows, self.num_cols, self.K, self.num_channels), dtype=np.float32)
        self.mean[:, :, 0] = self.initial_image
        self.mean_square = np.zeros((self.num_rows, self.num_cols, self.K), dtype=np.float32)
        self.mean_square[:, :, 0] = np.linalg.norm(self.initial_image, axis=2) ** 2
        self.weight = np.zeros((self.num_rows, self.num_cols, self.K), dtype=np.float32)
        self.weight[:, :, 0] = 1
        self.variance = np.full((self.num_rows, self.num_cols, self.K), self.initial_variance, dtype=np.float32)

    # Trains background subtractor module on given dataset
    #   dataset:         list: list of images (except first one)
    #   output_folder: string: path to save images
    def train(self, dataset, output_folder):
        for idx, image in enumerate(dataset):
            print("Training iteration:", idx + 1)
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    matching_gaussian = -1
                    for k in range(self.K):
                        if np.linalg.norm(image[i][j] - self.mean[i][j][k]) < 2.5 * np.sqrt(self.variance[i][j][k]):
                            matching_gaussian = k
                            break
                    # no match, initialize new gaussian
                    if matching_gaussian == -1:
                        least_weight_gaussian = np.argmin(self.weight[i][j])
                        self.weight[i, j, :] = (1 - self.alpha) * self.weight[i, j, :]
                        self.weight[i, j, least_weight_gaussian] += self.alpha
                        self.mean[i, j, least_weight_gaussian] = image[i][j]
                        self.mean_square[i, j, matching_gaussian] = np.linalg.norm(image[i][j]) ** 2
                        self.variance[i, j, least_weight_gaussian] = self.initial_variance
                    # match found, update existing gaussian
                    else:
                        self.weight[i, j, :] = (1 - self.alpha) * self.weight[i, j, :]
                        self.weight[i, j, matching_gaussian] += self.alpha
                        rho = self.alpha * self.gaussian_pdf(image[i][j].copy(), self.mean[i, j, matching_gaussian].copy(), self.variance[i, j, matching_gaussian].copy())
                        self.mean[i, j, matching_gaussian] = (1 - rho) * self.mean[i, j, matching_gaussian] + rho * image[i][j]
                        self.mean_square[i, j, matching_gaussian] = (1 - rho) * self.mean_square[i, j, matching_gaussian] + rho * (np.linalg.norm(image[i][j]) ** 2)
                        self.variance[i, j, matching_gaussian] = (1 - rho) * self.variance[i, j, matching_gaussian] + rho * np.abs(self.mean_square[i, j, matching_gaussian] - (np.linalg.norm(self.mean[i, j, matching_gaussian]) ** 2))
                        if self.variance[i, j, matching_gaussian] < 1e-8:
                            self.variance[i, j, matching_gaussian] = 1e-8
                        # self.variance[i, j, matching_gaussian] = (1 - rho) * self.variance[i, j, matching_gaussian] + rho * (np.linalg.norm(image[i][j] - self.mean[i, j, matching_gaussian]) ** 2)
            # self.find_background()
            # self.inference([image], output_folder)
        
    
    # Computes multi-dimensional gaussian pdf (scalar covariance matrix)
    #   X:        vector: input vector
    #   mean:     vector: mean vector
    #   variance: scalar: variance along a component
    def gaussian_pdf(self, X, mean, variance):
        X = np.reshape(X, (self.num_channels, 1))
        mean = np.reshape(mean, (self.num_channels, 1))
        covariance = variance * np.eye(self.num_channels)
        return np.exp(-0.5 * ((X - mean).T @ np.linalg.inv(covariance) @ (X - mean))) / ((np.sqrt(2 * np.pi) * variance) ** self.num_channels)
    
    # Finds gaussians corresponding to background for each pixel
    def find_background(self):
        self.background = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                bg_weights, bg_sum = [], 0
                for k in range(self.K):
                    bg_weights.append([self.weight[i][j][k] / np.sqrt(self.variance[i][j][k]), k])
                bg_weights.sort(reverse=True)
                for bg_weight in bg_weights:
                    self.background[i][j].append(bg_weight[1])
                    bg_sum += bg_weight[0]
                    if bg_sum > self.T:
                        break

    # Filter foreground-background pixels in given dataset
    #   dataset:         list: list of images
    #   output_folder: string: path to save images
    def inference(self, dataset, output_folder):
        for idx, image in enumerate(dataset):
            filtered_image = np.zeros((self.num_rows, self.num_cols, self.num_channels))
            for i in range(self.num_rows):
                for j in range(self.num_cols):
                    matching_gaussian = -1
                    for k in range(self.K):
                        if np.linalg.norm(image[i][j] - self.mean[i][j][k]) < 2.5 * self.variance[i][j][k]:
                            matching_gaussian = k
                            break
                    if matching_gaussian not in self.background[i][j]:
                        filtered_image[i][j][0], filtered_image[i][j][1], filtered_image[i][j][2] = 255, 255, 255
            print("Saving filtered image:", idx)
            cv2.imwrite(output_folder + "out" + str(idx).zfill(6) + ".png", filtered_image)
        

                    


