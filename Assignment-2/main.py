import os
import cv2
import argparse
import numpy as np
from imageStitcher import ImageStitcher
from cornerDetector import CornerDetector

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="1")
parser.add_argument("--p_radius", type=int, default=300)
parser.add_argument("--n_radius", type=int, default=2)
parser.add_argument("--sample_count", type=int, default=100)
parser.add_argument("--window_size", type=int, default=5)
parser.add_argument("--k", type=float, default=0.04)
parser.add_argument("--thresh", type=float, default=0.01)
parser.add_argument("--top_k", type=int, default=3000)
parser.add_argument("--nms_radius", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--tolerance", type=int, default=10)
parser.add_argument("--gaussian", default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()


def main():
    imageStitcher = ImageStitcher(args.p_radius, args.n_radius, args.sample_count, args.seed)
    window = {
        "type": "gaussian" if args.gaussian else "box",
        "size": args.window_size
    }
    cornerDetector = CornerDetector(window, args.k, args.thresh, args.top_k, args.nms_radius)
    data_dir = os.getcwd() + "/" + "Dataset/" + args.dir + "/"
    img_files = [(data_dir + "image " + str(i) + ".jpg") for i in range(len(os.listdir(data_dir)))]
    img_list = [cv2.imread(file) for file in img_files]
    img_left = img_list[0].copy()
    pts_left = cornerDetector.find(img_left)
    n = len(img_list)
    for i in range(1, n):
        img_right = img_list[i]
        pts_right = cornerDetector.find(img_right)
        src, dest = imageStitcher.match((img_right, img_left), (pts_right, pts_left))
        affine = imageStitcher.getAffineTransform2D(src, dest)
        result = cv2.warpAffine(img_right, affine, (img_right.shape[1] + img_left.shape[1], img_left.shape[0]))
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left.copy()    
        zero_idx = np.argwhere(np.all(result[..., :] == 0, axis=0))
        result = result[:, :zero_idx[0][0] - args.tolerance].copy() if i < n - 1 else result[:, :zero_idx[0][0]].copy()
        img_left = result.copy()
        pts_left = cornerDetector.find(img_left)
        print(f"Stitched image {i} to base...")
    print("Image stitching complete. Saving final image as stitched-" + args.dir + ".jpg")
    cv2.imwrite("stitched-" + args.dir + ".jpg", img_left)


if __name__ == "__main__":
    main()