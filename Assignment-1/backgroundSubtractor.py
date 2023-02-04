import cv2
import numpy as np
from skimage.feature import peak_local_max

class BackgroundSubtractor:
    def __init__(self, K, alpha, T, m_data, p_data, filter, fill, non_max):
        """Initializes background subtractor module (Mixture of Gaussians).
        
        Function parameters:\\
        K:        int: number of gaussians in the mixture\\
        alpha:  float: learning rate\\
        T:      float: cumulative threshold for filtering background distributions\\
        m_data:  dict: model initialization data (weight, variance and mean/image)\\
        p_data:  dict: foreground patch data (dimension and threshold)\\
        filter:  bool: flag for switching on filtering of image after prediction\\
        fill:    bool: flag for switching on filling of false negatives (integral images)\\
        non_max: bool: flag for switching on non-maximum suppression (integral images)
        
        Returns nothing.
        """
        # set model parameters
        self.K = K
        self.alpha = alpha
        self.T = T
        self.initial_weight = m_data["weight"]
        self.initial_variance = m_data["variance"]
        self.initial_mean = m_data["mean"]
        self.num_rows, self.num_cols = self.initial_mean.shape[:2]
        if len(self.initial_mean.shape) > 2:
            self.num_channels = self.initial_mean.shape[2]
        else:
            self.num_channels = 1
            self.initial_mean = self.initial_mean.reshape([self.num_rows, self.num_cols, self.num_channels])
        self.filter = filter
        self.fill = fill
        self.non_max = non_max
        # set foreground patch data
        self.patch_dim = p_data["dim"]
        self.patch_thresh = p_data["thresh"]
        # initialize MOG parameters
        self.weight = np.zeros((self.num_rows, self.num_cols, self.K), dtype=np.float32)
        self.weight[:, :, self.K - 1] = 1
        self.variance = np.full((self.num_rows, self.num_cols, self.K), self.initial_variance, dtype=np.float32)
        self.mean = np.zeros((self.num_rows, self.num_cols, self.K, self.num_channels), dtype=np.float32)
        self.mean[:, :, self.K - 1] = self.initial_mean.copy()

    def fit(self, gt_images, in_images, out_dir):
        """Fits background subtractor module (Mixture of Gaussians) on given dataset.
        
        Function parameters:\\
        gt_images: list: list of groundtruth images (except first one)\\
        in_images: list: list of input images (except first one)\\
        out_dir: string: directory path to save foreground video\\
        gt_dir:  string: directory path containing groundtruth images
        
        Returns nothing.
        """
        mIoU = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_name = "foreground"
        if self.filter:
            video_name += "_filter"
            if self.fill:
                video_name += "_fill"
                if self.non_max:
                    video_name += "_nonmax"
            else:
                if self.non_max:
                    video_name += "_nonmax"
        else:
            video_name += "_raw"
        video_name += ".mp4"
        video_writer = cv2.VideoWriter(out_dir + video_name, fourcc, 15, (self.num_cols, self.num_rows))
        for idx, image in enumerate(in_images):
            if self.num_channels == 1:
                image = image.reshape([self.num_rows, self.num_cols, self.num_channels])
            # fit MOG using K-means approximation of EM algorithm (Stauffer and Grimson)
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
            for c in range(self.num_channels):
                mean_copy[:, :, c] = np.take_along_axis(mean_copy[:, :, c], sort_weights, axis=-1)
            mean_copy[:, 0] = image[no_match_locations].copy()
            self.mean[no_match_locations] = mean_copy.copy()
            self.variance[no_match_locations] = np.take_along_axis(self.variance[no_match_locations], sort_weights, axis=-1)
            variance_copy = self.variance[no_match_locations].copy()
            variance_copy[:, 0] = self.initial_variance
            self.variance[no_match_locations] = variance_copy.copy()
            # find foreground pixels by sorting gaussians w.r.t their weight/sigma ratio (Stauffer and Grimson)
            sort_weights_by_std = np.argsort(self.weight / np.sqrt(self.variance), axis=-1)
            self.weight = np.take_along_axis(self.weight, sort_weights_by_std, axis=-1)
            mean_copy = self.mean.copy()
            for c in range(self.num_channels):
                mean_copy[:, :, :, c] = np.take_along_axis(mean_copy[:, :, :, c], sort_weights_by_std, axis=-1)
            self.mean = mean_copy.copy()
            self.variance = np.take_along_axis(self.variance, sort_weights_by_std, axis=-1)
            foreground = self.find_foreground(image)
            if self.filter:
                # primary filter (noise removal)
                foreground = self.clean_foreground(foreground, (15, 9), 70, False)
                # secondary filter (detect objects of certain size)
                foreground = self.clean_foreground(foreground, self.patch_dim, self.patch_thresh, self.fill)
            color_foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
            # calculate mIoU
            gt_mask = (gt_images[idx][:, :, 0] > 0)
            out_mask = (foreground > 0)
            union_mask = np.logical_or(gt_mask, out_mask)
            intersection_mask = np.logical_and(gt_mask, out_mask)
            if union_mask.sum() == 0:
                curr_mIoU = 1.00
            else:
                curr_mIoU = intersection_mask.sum() / union_mask.sum()
            mIoU += curr_mIoU
            # generate bounding box around detected objects
            contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < 400:
                    continue
                cv2.rectangle(color_foreground, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # save image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(color_foreground, 'mIoU: ' + str(round(curr_mIoU, 2)), (0, 13), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            video_writer.write(color_foreground)
            # cv2.imwrite(out_dir + "out" + str(0).zfill(6) + ".png", color_foreground)
            # cv2.waitKey(30)  
        video_writer.release()
        mIoU /= len(gt_images)
        print("Mean mIoU: ", mIoU)


    def find_foreground(self, image):
        """Finds pixels corresponding to foreground.

        Function parameters:\\
        image: array[R, C, num_channels]: input image

        Returns foreground mask (noisy)
        """
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
    
    def clean_foreground(self, foreground, dim, thresh, fill):
        """Cleans foreground by using integral images and thresholding

        Function parameters:\\
        foreground: array[R, C]: foreground mask (noisy)\\
        dim:     pair(int, int): integration patch dimension\\
        thresh:             int: integration threshold

        Returns cleaned foreground mask (noise-free)
        """
        height, width = dim
        fore_integral = cv2.integral(foreground / 255)
        patch_rows, patch_cols = self.num_rows - height + 1, self.num_cols - width + 1
        patch = fore_integral[height:, width:] - fore_integral[height:, :patch_cols] - fore_integral[:patch_rows, width:] + fore_integral[:patch_rows, :patch_cols]
        thresh_mask = patch > thresh
        if self.non_max:
            patch[np.where(np.logical_not(thresh_mask))] = 0
            max_indices = peak_local_max(patch, min_distance=1)
            max_mask = np.zeros_like(patch, dtype=bool)
            max_mask[tuple(max_indices.T)] = True
            total_mask = np.logical_and(max_mask, thresh_mask)
            indices = np.argwhere(total_mask)
        else:
            indices = np.argwhere(thresh_mask)
        cleaned_foreground = np.zeros_like(foreground, dtype=np.uint8)
        for i in range(indices.shape[0]):
            if fill:
                cleaned_foreground[indices[i][0]:indices[i][0] + height, indices[i][1]:indices[i][1] + width] = 255
            else:
                cleaned_foreground[indices[i][0]:indices[i][0] + height, indices[i][1]:indices[i][1] + width] = foreground[indices[i][0]:indices[i][0] + height, indices[i][1]:indices[i][1] + width]
        return cleaned_foreground
 
    def gaussian_pdf(self, X, mean, variance):
        """Computes multi-dimensional gaussian pdf (scalar covariance matrix).
        
        Function parameters:\\
        X:        array[B, num_channels]: input array\\
        mean:     array[B, num_channels]: mean array\\
        variance:              array[B,]: variance vector

        Returns corresponding multi-dimensional gaussian pdf
        """
        return np.exp(-0.5 * (np.linalg.norm(X - mean, axis=-1) ** 2) / variance) / ((np.sqrt(2 * np.pi * variance)) ** self.num_channels)


        

                    


