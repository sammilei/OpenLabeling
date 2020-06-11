import cv2
import numpy as np


def enhance_img(img):
    # using gamma correction + brightness increase on V-channel

    # 1. get the aver intensity of the whole img.
    # it will be better to get the bbox intensity and ave both
    blue_medium = int(np.median(img[:, :, 0]))
    green_medium = int(np.median(img[:, :, 1]))
    red_medium = int(np.median(img[:, :, 2]))
    intensity = int((blue_medium + green_medium + red_medium) / 3)

    # 2. calculate gamma by mapping
    gamma = get_gamma(intensity)

    # 3. gamma correction
    hsv_gamma = cv2.cvtColor(gamma_correction(
        img, gamma=gamma), cv2.COLOR_BGR2HSV)

    # 4.brighten the img a bit
    return brighten_img(hsv_gamma, enhanced_init=gamma_to_brightness(gamma))


def brighten_img(hsv, enhanced_init=10):
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.minimum(
        enhanced_init + hsv[:, :, 2].astype(np.float) * 2, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_gamma(intensity, gamma_min=10,
              gamma_max=120,
              brightness_min=0,
              brightness_max=120):  # orignal: 120
    if intensity < brightness_max:
        gamma = float(gamma_max - gamma_min) / float(brightness_max -
                                                     brightness_min) * float(intensity - brightness_min) + gamma_min
        return gamma
    return gamma_max


def gamma_correction(img, gamma=90, gamma_max=200):
    lookUpTable = np.empty((1, 256), np.uint8)
    gamma = min(gamma, gamma_max) / 100
    lookUpTable[0, :] = np.clip((np.arange(256) / 255.) ** gamma * 255, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res


def gamma_to_brightness(gamma, gamma_min=10, gamma_max=120, brighten_min=0, brighten_max=10):
    return (brighten_max - brighten_min) / (gamma_max -
                                            gamma_min) * (gamma - gamma_min) + brighten_min
