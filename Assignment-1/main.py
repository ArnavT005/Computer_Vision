import os
import cv2
import argparse
from backgroundSubtractor import BackgroundSubtractor

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="IBMtest2")
parser.add_argument("--K", type=int, default=3)
parser.add_argument("--A", type=float, default=0.03)
parser.add_argument("--T", type=float, default=0.7)
parser.add_argument("--wt", type=float, default=0.08)
parser.add_argument("--var", type=float, default=600)
parser.add_argument("--height", type=int, default=15)
parser.add_argument("--width", type=int, default=9)
parser.add_argument("--thresh", type=int, default=70)
parser.add_argument("--filter", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--fill", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--nonmax", default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()


def main():
    data_dir = args.dir
    K = args.K
    alpha = args.A
    T = args.T
    initial_weight = args.wt
    initial_variance = args.var
    patch_dim = (args.height, args.width)
    patch_thresh = args.thresh
    patch_data = {
        "dim": patch_dim,
        "thresh": patch_thresh
    }
    gt_dir = os.getcwd() + "/" + "Dataset/" + data_dir + "/" + "groundtruth/"
    in_dir = os.getcwd() + "/" + "Dataset/" + data_dir + "/" + "input/"
    out_dir = os.getcwd() + "/" + "Dataset/" + data_dir + "/"
    gt_files = [(gt_dir + f) for f in os.listdir(gt_dir)]
    gt_files.sort()
    in_files = [(in_dir + f) for f in os.listdir(in_dir)]
    in_files.sort()
    gt_images = [cv2.imread(file) for file in gt_files]
    in_images = [cv2.imread(file) for file in in_files]
    initial_mean = in_images[0]
    initial_data = {
        "weight": initial_weight,
        "variance": initial_variance,
        "mean": initial_mean
    }
    model = BackgroundSubtractor(K, alpha, T, initial_data, patch_data, args.filter, args.fill, args.nonmax)
    model.fit(gt_images[1:], in_images[1:], out_dir)


if __name__ == "__main__":
    main()