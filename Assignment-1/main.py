import os
import cv2
import sys
from backgroundSubtractor import BackgroundSubtractor

def main():
    folder = sys.argv[1]
    K = int(sys.argv[2])
    alpha = float(sys.argv[3])
    T = float(sys.argv[4])
    input_folder = os.getcwd() + "/" + "Dataset/" + folder + "/" + "input/"
    input_files = [(input_folder + f) for f in os.listdir(input_folder)]
    input_files.sort()
    input_images = [cv2.imread(file) for file in input_files]
    initial_image = input_images[0]
    num_rows, num_cols, num_channels = initial_image.shape
    model = BackgroundSubtractor(K, alpha, T, num_rows, num_cols, num_channels, 220.0, initial_image)
    model.train(input_images)

if __name__ == "__main__":
    main()



