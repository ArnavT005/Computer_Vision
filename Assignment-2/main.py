import cv2
from imageStitcher import ImageStitcher
from cornerDetector import CornerDetector
import numpy as np

imageStitcher = ImageStitcher(1000, 2, 100)
window = {
    "type": "gaussian",
    "size": 5,
}
cornerDetector = CornerDetector(window, 0.04, 0.01, 2)
img_list = [cv2.imread(f"Dataset/1/image {i}.jpg") for i in range(15)]
while len(img_list) > 1:
    temp_img_list = []
    for i in range(len(img_list) // 2):
        img = img_list[2 * i]
        dst = cornerDetector.find(img)
        img_ = img_list[2 * i + 1]
        dst_ = cornerDetector.find(img_)
        src, dest = imageStitcher.match((img_, img), (dst_, dst))
        h = imageStitcher.getAffineTransform(src, dest)
        result = cv2.warpPerspective(img_, h, (img_.shape[1] + img.shape[1], img_.shape[0]))
        result[0:img.shape[0], 0:img.shape[1]] = img.copy()
        zero_idx = np.argwhere(np.all(result[..., :] == 0, axis=0))
        result = result[:, :zero_idx[0][0]].copy()
        temp_img_list.append(result)
        # breakpoint()
    if len(img_list) % 2 == 1:
        temp_img_list.append(img_list[-1].copy())
    img_list = temp_img_list
breakpoint()
# dst = cornerDetector.find(img)
# dst = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5, 3, 0.2)
# dst = cv2.dilate(dst * 1.0,None)
# img[dst > 0] = [0, 0, 255]
# for i in range(1, 15):
#     img_ = cv2.imread(f"Dataset/2/image {6}.jpg")
#     dst_ = cornerDetector.find(img_)
#     src, dest = imageStitcher.match((img_, img), (dst_, dst))
#     h = imageStitcher.getAffineTransform(src, dest)
#     result = cv2.warpPerspective(img_, h, (img_.shape[1] + img.shape[1], img_.shape[0]))
#     # cv2.imwrite("persp.png", result)
#     # src_array = np.concatenate((np.array(src), np.ones((len(src), 1))), axis=-1)
#     # dest_array = np.array(dest)
#     # affine = (np.linalg.inv(src_array.T @ src_array) @ src_array.T @ dest_array).T
#     # result = cv2.warpAffine(img_, affine, (img_.shape[1] + img.shape[1], img_.shape[0]))
#     # cv2.imwrite("affine.png", result)
#     result[0:img.shape[0], 0:img.shape[1]] = img.copy()    
#     cv2.imwrite("sample.png", result)
#     breakpoint()
#     zero_idx = np.argwhere(np.all(result[..., :] == 0, axis=0))
#     # result = np.delete(result, zero_idx, axis=1)
#     result = result[:, :zero_idx[0][0]].copy()
#     cv2.imwrite("crop.png", result)
#     breakpoint()
#     img = result.copy()
#     dst = cornerDetector.find(img)
# breakpoint()
# print(((dst > 0) * 1).sum())
# dst = cv2.dilate(dst,None)
# img[dst > 0] = [0, 0, 255]
# cv2.imwrite("harris_our.png", img)
