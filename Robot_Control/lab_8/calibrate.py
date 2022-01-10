#!/usr/bin/env python3

import cv2
from os import listdir
from os.path import isfile, join
import numpy as np


def calibrate(nx, ny, dir, drawcorner=False):

    object_points = np.zeros((nx*ny,3), np.float32)
    object_points[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d pionts in image plane.

    img_size = (0, 0)

    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    for file in files:
        print(file)
        img = cv2.imread(dir + '\\' + file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(grey, (8, 5),
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                         cv2.CALIB_CB_FAST_CHECK +
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            objpoints.append(object_points)
            imgpoints.append(corners)
            if drawcorner:
                fnl = cv2.drawChessboardCorners(img, (8, 5), corners, found)
                cv2.imshow("fnl", fnl)
                cv2.waitKey(0)

        else:
            print("No Chessboard Found")

        img_size = (img.shape[1], img.shape[0])

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist, img_size


def undistort(mtx, dist, img_size, alpha=1):
    new_cam_mtx, valid_roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, alpha, img_size)
    map_x, map_y = cv2.initUndistortRectifyMap(mtx, dist, None, new_cam_mtx, img_size, cv2.CV_16SC2)
    return map_x, map_y


def undistort_img(image, dir):
    mtx, dist, img_size = calibrate(8, 5, dir)
    map_x, map_y = undistort(mtx, dist, img_size)
    img = cv2.imread(image)
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow("undistorted", img)
    cv2.waitKey(0)
    return img
