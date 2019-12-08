import cv2
import numpy as np
import main


# shape = np.array([500, 500, 3])
# img = np.ones(shape)
# blank = np.zeros([500, 500, 4])
# cv2.imshow("blank", blank)
# cv2.waitKey()
#

targetimg = cv2.imread("green.png")
targetimg = cv2.resize(targetimg, (100, 100))

img = cv2.imread(r"images\fkm8s0.png")
cv2.imshow("original", img)
cv2.imshow("target", targetimg)
cv2.waitKey()

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

err = main.rico_mse(targetimg, img)
err2 = main.rico_mse(img, targetimg)
print(err, err2)

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