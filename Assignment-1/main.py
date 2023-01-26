import os
import cv2
import sys
from backgroundSubtractor import BackgroundSubtractor
import numpy as np

def norm_pdf(x,mean,sigma):
        return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

def main():
    folder = sys.argv[1]
    K = int(sys.argv[2])
    alpha = float(sys.argv[3])
    T = float(sys.argv[4])
    initial_weight = float(sys.argv[5])
    initial_variance = float(sys.argv[6])
    input_folder = os.getcwd() + "/" + "Dataset/" + folder + "/" + "input/"
    output_folder = os.getcwd() + "/" + "Dataset/" + folder + "/" + "output/"
    input_files = [(input_folder + f) for f in os.listdir(input_folder)]
    input_files.sort()
    # input_images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in input_files]
    input_images = [cv2.imread(file) for file in input_files]
    initial_image = input_images[0]
    initial_data = {
        "weight": initial_weight,
        "variance": initial_variance,
        "image": initial_image
    }
    model = BackgroundSubtractor(K, alpha, T, initial_data)
    model.train(input_images[1:], output_folder)


if __name__ == "__main__":
    main()



