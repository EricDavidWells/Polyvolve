import cv2
import numpy as np
from skimage import metrics


def rico_ssim(img1, img2):
    ssim = metrics.structural_similarity(img1, img2, multichannel=True)
    return ssim


def rico_mse(image1, image2, displayFlag=False):
    diff = np.subtract(image1, image2)
    err = np.sum(np.square(diff)) / np.prod(image1.shape)
    return err


def rico_mse_hsv(image1, image2):
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
    diff = np.subtract(hsv1, hsv2)
    err = np.sum(np.square(diff)) / np.prod(image1.shape)
    return err


def rico_mse_lab(image1, image2):
    lab1 = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)
    lab2 = cv2.cvtColor(image2, cv2.COLOR_RGB2LAB)
    diff = np.subtract(lab1, lab2)
    err = np.sum(np.square(diff)) / np.prod(image1.shape)
    return err

