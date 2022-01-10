#!/usr/bin/env python3

import cv2
from os import listdir
from os.path import isfile, join


def find_chessboard(dir):
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
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), term)

            fnl = cv2.drawChessboardCorners(img, (8, 5), corners, found)
            cv2.imshow("fnl", fnl)
            cv2.waitKey(0)
        else:
            print("No Chessboard Found")


find_chessboard('robot_control\\camera_calibration\\calib_photos')
