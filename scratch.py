import cv2
import numpy as np
import main
from numba import cuda
from skimage import metrics
# print(cuda.gpus)

# shape = np.array([500, 500, 3])
# img = np.ones(shape)
# blank = np.zeros([500, 500, 4])
# cv2.imshow("blank", blank)
# cv2.waitKey()
#


targetimg = cv2.imread("images\\manatee.png")
guessimg = cv2.imread("images\\20191209_2317_GA1088.png")

calcsize = 300
shape_orig = targetimg.shape
calcshape = (calcsize, int(calcsize * shape_orig[0] / shape_orig[1]))

img_calc = cv2.resize(targetimg, calcshape)
img_gues = cv2.resize(guessimg, calcshape)

cv2.imshow("target", img_calc)
cv2.imshow("guess", img_gues)
cv2.waitKey()

print(metrics.mean_squared_error(img_calc, img_gues))
print(metrics.normalized_root_mse(img_calc, img_gues))
# targetimg = cv2.imread("green.png")
# targetimg = cv2.resize(targetimg, (100, 100))
#
# img = cv2.imread(r"images\fkm8s0.png")
# cv2.imshow("original", img)
# cv2.imshow("target", targetimg)
# cv2.waitKey()
#
# R = img[:,:,0]
# G = img[:,:,1]
# B = img[:,:,2]
#
# err = main.rico_mse(targetimg, img)
# err2 = main.rico_mse(img, targetimg)
# print(err, err2)

# shape = img.shape
# vcat = img
# numcols = 4
#
# for i in range(0, numcols):
#     vcat = np.concatenate((vcat, img))
#
# vcat = cv2.resize(vcat, (int(500*shape[1]/(shape[0]*numcols)), 500))
#
# cv2.imshow("hcat", vcat)
# cv2.imwrite("wtf.jpg", vcat)
# cv2.waitKey()